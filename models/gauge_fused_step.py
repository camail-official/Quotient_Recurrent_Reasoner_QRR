"""Fused z-step update for QuotientIntegrator._core (used for fp32 CUDA inputs).

Replaces QuotientIntegrator._core's ~25 eager elementwise passes over [B,S,H] with one
triton kernel per direction (fwd/bwd), program-per-token, everything in registers.
EXACT same math as the eager QuotientIntegrator._core (order 2), including:
  - fp32 split with den_eps clamp:  a = Im<y, F> / max(|y|^2, den_eps),  F = z - y
  - F_H = F - i a y if PROJ (a carries gradient -- quotient rule in the VJP), else F_H = F
  - prop = beta m + (1-beta) F_H,  m' = eta prop,  y_n = y + eta prop
  - rotation R_theta with theta = remainder(a.detach(), 2pi)  (theta DETACHED, as eager)
  - r_prop / r_exec per-token L-inf complex maxes (no_grad), board-max taken outside
Backward returns (gy, gm, gz, g_bH); eta is treated as a constant (eager path's eta enters
via _finish's torch.where bookkeeping, not through _core's graph -- matches use_ckpt=True
training where eta is an input to the checkpointed pure fn and gets no grad).
Parity-tested against the eager _core (fp32 tolerance 5e-5) on sudoku and maze.
projection_mode 'c1_only' falls back to eager (see integrator.py).
"""
import math
import torch
import triton
import triton.language as tl

FUSED = True                  # fused z-step enabled


@triton.jit
def _fused_fwd(Y, M, Z, ETA, BH,
               YN, MN, ASAVE, RPROP, REXEC,
               DEN_EPS, EPS_R, APPLY_R: tl.constexpr, PROJ: tl.constexpr,
               HC: tl.constexpr):
    """One program per (b*s) token row.  HC = H//2 complex pairs (compile-time)."""
    pid = tl.program_id(0)
    off = pid * (2 * HC) + tl.arange(0, HC)
    re_i = pid * (2 * HC) + 2 * tl.arange(0, HC)
    im_i = re_i + 1
    yr = tl.load(Y + re_i); yi = tl.load(Y + im_i)
    zr = tl.load(Z + re_i); zi = tl.load(Z + im_i)
    mr = tl.load(M + re_i); mi = tl.load(M + im_i)
    eta = tl.load(ETA + pid)                                     # per-token eta (host expands)
    bh = tl.load(BH)
    beta = 1.0 / (1.0 + tl.exp(-bh))

    fr = zr - yr; fi = zi - yi
    n2 = tl.sum(yr * yr + yi * yi, 0)
    n2 = tl.maximum(n2, DEN_EPS)
    # Im<y, F> = sum(yr*fi - yi*fr)
    aip = tl.sum(yr * fi - yi * fr, 0)
    a = aip / n2
    # F_H = F - i a y  ->  re: fr + a*yi ; im: fi - a*yr
    # PROJ=False (projection_mode: none) is the UNPROJECTED law, F_H = F.  `a` is still
    # computed above because APPLY_R needs it as the rotation angle -- projection-off is
    # not rotation-off.
    if PROJ:
        fhr = fr + a * yi
        fhi = fi - a * yr
    else:
        fhr = fr
        fhi = fi
    pr = beta * mr + (1.0 - beta) * fhr
    pi = beta * mi + (1.0 - beta) * fhi
    mnr = eta * pr; mni = eta * pi
    ynr = yr + eta * pr; yni = yi + eta * pi

    # per-token L-inf pieces; the RATIO is taken on the host from BOARD maxes
    # (ratio-of-maxes, exactly _linf_c semantics -- NOT max-of-ratios)
    pmag = tl.sqrt(pr * pr + pi * pi)
    ypmag = tl.sqrt((yr + pr) * (yr + pr) + (yi + pi) * (yi + pi))
    ymag = tl.sqrt(yr * yr + yi * yi)
    tl.store(RPROP + pid, tl.max(pmag, 0))
    tl.store(REXEC + pid, tl.max(ypmag, 0))
    tl.store(ASAVE + pid, a)
    tl.store(RPROP + tl.num_programs(0) + pid, eta * tl.max(pmag, 0))
    tl.store(REXEC + tl.num_programs(0) + pid, tl.max(ymag, 0))

    if APPLY_R:
        TWO_PI = 6.283185307179586
        th = a - tl.floor(a / TWO_PI) * TWO_PI
        c = tl.cos(th); s = tl.sin(th)
        or_ = c * ynr - s * yni
        oi_ = s * ynr + c * yni
        mr2 = c * mnr - s * mni
        mi2 = s * mnr + c * mni
        tl.store(YN + re_i, or_); tl.store(YN + im_i, oi_)
        tl.store(MN + re_i, mr2); tl.store(MN + im_i, mi2)
    else:
        tl.store(YN + re_i, ynr); tl.store(YN + im_i, yni)
        tl.store(MN + re_i, mnr); tl.store(MN + im_i, mni)


@triton.jit
def _fused_bwd(Y, M, Z, ETA, BH, ASAVE,
               GYN, GMN,
               GY, GM, GZ, GBH_PART,
               DEN_EPS, APPLY_R: tl.constexpr, PROJ: tl.constexpr,
               HC: tl.constexpr):
    """VJP of _fused_fwd (r stats are no_grad; theta detached exactly as eager)."""
    pid = tl.program_id(0)
    re_i = pid * (2 * HC) + 2 * tl.arange(0, HC)
    im_i = re_i + 1
    yr = tl.load(Y + re_i); yi = tl.load(Y + im_i)
    zr = tl.load(Z + re_i); zi = tl.load(Z + im_i)
    mr = tl.load(M + re_i); mi = tl.load(M + im_i)
    gyr = tl.load(GYN + re_i); gyi = tl.load(GYN + im_i)
    gmr = tl.load(GMN + re_i); gmi = tl.load(GMN + im_i)
    eta = tl.load(ETA + pid)
    bh = tl.load(BH)
    beta = 1.0 / (1.0 + tl.exp(-bh))
    a = tl.load(ASAVE + pid)

    if APPLY_R:
        TWO_PI = 6.283185307179586
        th = a - tl.floor(a / TWO_PI) * TWO_PI
        c = tl.cos(th); s = tl.sin(th)
        # rotate incoming grads back: g <- R(-theta) g
        t = c * gyr + s * gyi
        gyi = -s * gyr + c * gyi; gyr = t
        t = c * gmr + s * gmi
        gmi = -s * gmr + c * gmi; gmr = t

    # y_n = y + eta*prop ; m' = eta*prop
    gpr = eta * (gyr + gmr)
    gpi = eta * (gyi + gmi)
    # prop = beta m + (1-beta) F_H
    gmr_out = beta * gpr
    gmi_out = beta * gpi
    gfhr = (1.0 - beta) * gpr
    gfhi = (1.0 - beta) * gpi
    # d beta (scalar): sum((m - F_H) . gprop) * beta*(1-beta)
    fr = zr - yr; fi = zi - yi
    if PROJ:
        n2r = tl.sum(yr * yr + yi * yi, 0)
        n2 = tl.maximum(n2r, DEN_EPS)
        aip = tl.sum(yr * fi - yi * fr, 0)
        aa = aip / n2
        fhr = fr + aa * yi
        fhi = fi - aa * yr
    else:
        fhr = fr
        fhi = fi
    dbeta = tl.sum((mr - fhr) * gpr + (mi - fhi) * gpi, 0)
    tl.store(GBH_PART + pid, dbeta * beta * (1.0 - beta))

    if PROJ:
        # F_H = F - i a y : gF = gFH ; ga = sum(yi*gfhr - yr*gfhi) ; gy += a*(i-part...)
        ga = tl.sum(yi * gfhr - yr * gfhi, 0)
        gfr = gfhr
        gfi = gfhi
        # d(F_H)/dy at fixed a: fhr has +a*yi -> d/dyi = +a ; fhi has -a*yr -> d/dyr = -a
        gyr_l = -aa * gfhi
        gyi_l = aa * gfhr
        # a = aip/n2 (with clamp: gradient through n2 only when n2r > DEN_EPS)
        live = n2r > DEN_EPS
        inv_n2 = 1.0 / n2
        ga_ip = ga * inv_n2
        ga_n2 = tl.where(live, -ga * aip * inv_n2 * inv_n2, 0.0)
        # aip = sum(yr*fi - yi*fr):  d/dyr = fi + ... F = z - y couples too; handle F first:
        #   d aip/d fr = -yi ; d aip/d fi = +yr
        gfr += ga_ip * (-yi)
        gfi += ga_ip * (yr)
        #   d aip/d yr = fi ; d aip/d yi = -fr
        gyr_l += ga_ip * fi
        gyi_l += ga_ip * (-fr)
        # n2 = sum(yr^2 + yi^2): d/dyr = 2 yr
        gyr_l += ga_n2 * 2.0 * yr
        gyi_l += ga_n2 * 2.0 * yi
    else:
        # F_H = F exactly, so there is NO a-dependence in the graph: `a` reaches the output
        # only through the DETACHED rotation angle (QuotientIntegrator._core), matching eager,
        # where autograd likewise builds no a->y/z path.  The whole ga chain is dead code
        # and the two reductions that produce `aa` are not needed here at all.
        gfr = gfhr
        gfi = gfhi
        gyr_l = gfhr * 0.0
        gyi_l = gfhi * 0.0
    # F = z - y
    gzr = gfr; gzi = gfi
    gyr_l += gyr + (-gfr)
    gyi_l += gyi + (-gfi)
    tl.store(GY + re_i, gyr_l); tl.store(GY + im_i, gyi_l)
    tl.store(GM + re_i, gmr_out); tl.store(GM + im_i, gmi_out)
    tl.store(GZ + re_i, gzr); tl.store(GZ + im_i, gzi)


class _FusedStepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, m, eta_tok, b_H, z, den_eps, eps_r, apply_R, proj=True):
        B, S, H = y.shape
        yc, mc, zc = y.contiguous(), m.contiguous(), z.contiguous()
        et = eta_tok.contiguous()
        yn = torch.empty_like(yc); mn = torch.empty_like(mc)
        a = torch.empty(B * S, device=y.device, dtype=torch.float32)
        rp = torch.empty(2 * B * S, device=y.device, dtype=torch.float32)
        re = torch.empty(2 * B * S, device=y.device, dtype=torch.float32)
        _fused_fwd[(B * S,)](yc, mc, zc, et, b_H, yn, mn, a, rp, re,
                             den_eps, eps_r, APPLY_R=apply_R, PROJ=proj, HC=H // 2)
        ctx.save_for_backward(yc, mc, zc, et, b_H, a)
        ctx.meta = (den_eps, apply_R, proj, B, S, H)
        return yn, mn, rp.view(2, B, S), re.view(2, B, S)

    @staticmethod
    def backward(ctx, gyn, gmn, _grp, _gre):
        yc, mc, zc, et, b_H, a = ctx.saved_tensors
        den_eps, apply_R, proj, B, S, H = ctx.meta
        gy = torch.empty_like(yc); gm = torch.empty_like(mc); gz = torch.empty_like(zc)
        gbp = torch.empty(B * S, device=yc.device, dtype=torch.float32)
        _fused_bwd[(B * S,)](yc, mc, zc, et, b_H, a,
                             gyn.contiguous(), gmn.contiguous(),
                             gy, gm, gz, gbp,
                             den_eps, APPLY_R=apply_R, PROJ=proj, HC=H // 2)
        return gy, gm, None, gbp.sum().reshape(1), gz, None, None, None, None


def fused_core(integ, y_flat, m_flat, eta, z_new):
    """Drop-in for QuotientIntegrator._core when FUSED and order==2 and grad-safe dtypes.
    Returns (y_n, m_out, r_prop, r_exec) with r_* board L-inf maxes matching _linf_c."""
    B, S, H = y_flat.shape
    eta_tok = eta.expand(B, S, 1).reshape(B * S).to(torch.float32).contiguous()
    yn, mn, rp, re = _FusedStepFn.apply(
        y_flat.float(), m_flat.float(), eta_tok, integ.b_H.float().reshape(1),
        z_new.float(), float(integ.den_eps), float(integ.eps_r), bool(integ.apply_R),
        bool(getattr(integ, 'proj_force', True)))
    # ratio of board L-inf maxes, matching _core/_linf_c exactly
    r_prop = rp[0].amax(-1) / (re[0].amax(-1) + integ.eps_r)
    r_exec = rp[1].amax(-1) / (re[1].amax(-1) + integ.eps_r)
    return yn, mn, r_prop, r_exec
