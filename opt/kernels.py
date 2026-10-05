"""QRR release opt kernels (inference-only, no backward).

K1: fused gauge conv2d --
Math identical to gauge_layers.GaugeConv.forward (conv2d branch):
    out[b,s,c] = sum_j kern[j,c] * u_j(b,s) * h[b, src(j,s), c]
    u_j(b,s)   = g[b,s] * conj(g[b,src(j,s)]) * ok(j,s) * (mask[b,s,src] if given)
Index tables (src/ok) cached per (Sg,k) on device; single triton kernel, C chunked."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gconv_fwd(H, G, K_, SRC, OK, MASK, OUT,
               Sg: tl.constexpr, C: tl.constexpr, NT: tl.constexpr,
               HAS_MASK: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)                       # one program per (b, s)
    b = pid // Sg
    s = pid % Sg
    gr_s = tl.load(G + (b * Sg + s) * 2 + 0)
    gi_s = tl.load(G + (b * Sg + s) * 2 + 1)
    for c0 in range(0, C, BC):
        cs = c0 + tl.arange(0, BC)
        cm = cs < C
        accr = tl.zeros([BC], dtype=tl.float32)
        acci = tl.zeros([BC], dtype=tl.float32)
        for j in range(NT):
            src = tl.load(SRC + j * Sg + s)
            ok = tl.load(OK + j * Sg + s).to(tl.float32)
            if HAS_MASK:
                mk = tl.load(MASK + (b * Sg + s) * Sg + src).to(tl.float32)
                ok = ok * mk
            gr_t = tl.load(G + (b * Sg + src) * 2 + 0)
            gi_t = tl.load(G + (b * Sg + src) * 2 + 1)
            # u = g_s * conj(g_t) * ok
            ur = (gr_s * gr_t + gi_s * gi_t) * ok
            ui = (gi_s * gr_t - gr_s * gi_t) * ok
            hb = H + ((b * Sg + src) * C + cs) * 2
            hr = tl.load(hb + 0, mask=cm, other=0.0)
            hi = tl.load(hb + 1, mask=cm, other=0.0)
            kb = K_ + (j * C + cs) * 2
            kr = tl.load(kb + 0, mask=cm, other=0.0)
            ki = tl.load(kb + 1, mask=cm, other=0.0)
            # t = u * h ; out += kern * t
            tr = ur * hr - ui * hi
            ti = ur * hi + ui * hr
            accr += kr * tr - ki * ti
            acci += kr * ti + ki * tr
        ob = OUT + ((b * Sg + s) * C + cs) * 2
        tl.store(ob + 0, accr, mask=cm)
        tl.store(ob + 1, acci, mask=cm)


_TABLES = {}


def _tables(Sg, offsets, device):
    key = (Sg, tuple(offsets), device)
    if key not in _TABLES:
        hw = int(math.isqrt(Sg))
        import numpy as np
        src = np.zeros((len(offsets), Sg), dtype=np.int32)
        ok = np.zeros((len(offsets), Sg), dtype=np.int8)
        yy, xx = np.divmod(np.arange(Sg), hw)
        for j, (dy, dx) in enumerate(offsets):
            sy, sx = yy + dy, xx + dx
            good = (sy >= 0) & (sy < hw) & (sx >= 0) & (sx < hw)
            src[j] = np.clip(sy, 0, hw - 1) * hw + np.clip(sx, 0, hw - 1)
            ok[j] = good
        _TABLES[key] = (torch.from_numpy(src).to(device),
                        torch.from_numpy(ok).to(device))
    return _TABLES[key]


def gauge_conv2d_fused(conv_mod, hc_grid, g_tok, mask_grid):
    """Drop-in for GaugeConv.forward, conv2d branch."""
    B, Sg, C = hc_grid.shape
    src, ok = _tables(Sg, conv_mod.offsets, hc_grid.device)
    H = torch.view_as_real(hc_grid.contiguous()).float().contiguous()
    G = torch.view_as_real(g_tok.contiguous()).float().contiguous()
    K_ = torch.view_as_real(torch.view_as_complex(conv_mod.kern).to(hc_grid.dtype)
                            .contiguous()).float().contiguous()
    out = torch.empty_like(H)
    if mask_grid is not None:
        MASK = mask_grid.to(torch.int8).contiguous()
        has_mask = True
    else:
        MASK = torch.empty(1, dtype=torch.int8, device=hc_grid.device)
        has_mask = False
    _gconv_fwd[(B * Sg,)](H, G, K_, src, ok, MASK, out,
                          Sg=Sg, C=C, NT=len(conv_mod.offsets),
                          HAS_MASK=has_mask, BC=64)
    return torch.view_as_complex(out)


# ---------------------------------------------------------------- K2: fused re_defect
@triton.jit
def _redefect_tok(PY, PM, Y1, M1, N2, Q2, C: tl.constexpr, BC: tl.constexpr,
                  HAS_M: tl.constexpr):
    pid = tl.program_id(0)                    # one program per (b*s) token
    ipr = 0.0
    ipi = 0.0
    q2 = 0.0
    for c0 in range(0, C, BC):
        cs = c0 + tl.arange(0, BC)
        cm = cs < C
        pb = PY + (pid * C + cs) * 2
        pyr = tl.load(pb + 0, mask=cm, other=0.0)
        pyi = tl.load(pb + 1, mask=cm, other=0.0)
        yb = Y1 + (pid * C + cs) * 2
        yr = tl.load(yb + 0, mask=cm, other=0.0)
        yi = tl.load(yb + 1, mask=cm, other=0.0)
        ipr += tl.sum(pyr * yr + pyi * yi, 0)      # Re(conj(py)*y1)
        ipi += tl.sum(pyr * yi - pyi * yr, 0)      # Im(conj(py)*y1)
        q2 += tl.sum(pyr * pyr + pyi * pyi, 0)
        if HAS_M:
            mb = PM + (pid * C + cs) * 2
            pmr = tl.load(mb + 0, mask=cm, other=0.0)
            pmi = tl.load(mb + 1, mask=cm, other=0.0)
            m1b = M1 + (pid * C + cs) * 2
            mr = tl.load(m1b + 0, mask=cm, other=0.0)
            mi = tl.load(m1b + 1, mask=cm, other=0.0)
            ipr += tl.sum(pmr * mr + pmi * mi, 0)
            ipi += tl.sum(pmr * mi - pmi * mr, 0)
            q2 += tl.sum(pmr * pmr + pmi * pmi, 0)
    # ph = conj(ip)/(|ip| + 1e-30)
    mag = tl.sqrt(ipr * ipr + ipi * ipi) + 1e-30
    phr = ipr / mag
    phi = -ipi / mag
    n2 = 0.0
    for c0 in range(0, C, BC):
        cs = c0 + tl.arange(0, BC)
        cm = cs < C
        pb = PY + (pid * C + cs) * 2
        pyr = tl.load(pb + 0, mask=cm, other=0.0)
        pyi = tl.load(pb + 1, mask=cm, other=0.0)
        yb = Y1 + (pid * C + cs) * 2
        yr = tl.load(yb + 0, mask=cm, other=0.0)
        yi = tl.load(yb + 1, mask=cm, other=0.0)
        br = yr * phr - yi * phi - pyr
        bi = yr * phi + yi * phr - pyi
        n2 += tl.sum(br * br + bi * bi, 0)
        if HAS_M:
            mb = PM + (pid * C + cs) * 2
            pmr = tl.load(mb + 0, mask=cm, other=0.0)
            pmi = tl.load(mb + 1, mask=cm, other=0.0)
            m1b = M1 + (pid * C + cs) * 2
            mr = tl.load(m1b + 0, mask=cm, other=0.0)
            mi = tl.load(m1b + 1, mask=cm, other=0.0)
            cr = mr * phr - mi * phi - pmr
            ci = mr * phi + mi * phr - pmi
            n2 += tl.sum(cr * cr + ci * ci, 0)
    tl.store(N2 + pid, n2)
    tl.store(Q2 + pid, q2)


def redefect_fused(py, pm, y1, m1):
    """Per-board dre matching the re_defect block in QRRModelInner.forward: [B] fp32.
    py/pm are the PRE-step real-view [B,S,H] tensors; y1/m1 post-step; m may be None."""
    B, S, H = y1.shape
    C = H // 2
    n2 = torch.empty(B * S, device=y1.device, dtype=torch.float32)
    q2 = torch.empty(B * S, device=y1.device, dtype=torch.float32)
    args = [py.float().contiguous(), (pm.float().contiguous() if pm is not None else py),
            y1.float().contiguous(), (m1.float().contiguous() if m1 is not None else y1),
            n2, q2]
    _redefect_tok[(B * S,)](*args, C=C, BC=64, HAS_M=pm is not None)
    n2 = n2.view(B, S).sum(1)
    q2 = q2.view(B, S).sum(1)
    return torch.sqrt(n2) / torch.sqrt(q2).clamp_min(1e-30)
