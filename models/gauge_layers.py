"""Token-local U(1) gauge-equivariant primitives.

Group action (h-ONLY; x neutral):  h_s -> e^{i phi_s} h_s  per token s, the phase acting on
the H/2 complex channels formed by ADJACENT real pairs (2c, 2c+1) -- the same pairing as the
state's n-cells (cell i owns 4i..4i+3, pairs (4i,4i+1),(4i+2,4i+3)).

Conventions:
- complex-linear maps, NO bias (a bias is not phase-covariant)
- params stored real [out_c, in_c, 2], used via view_as_complex
- degenerate connection entries => message masked to ZERO (a U=1 fallback breaks the local law)
- everything fp32 (complex ops are not autocast-eligible)
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.layers import rms_norm
from models.rng_streams import new_param

# Fixed implementation choices:
#   * every complex GLinear GEMM runs as ONE interleaved block-real fp32 GEMM
#     (the same linear map as the complex matmul exactly; only roundoff differs)
#   * the pure-gauge connection U = g g^H is consumed as DRESSED STANDARD sdpa:
#       scores = Re(g_s conj(g_t) <q_s,k_t>) = <view_r(conj(g) q), view_r(conj(g) k)>
#       out_s  = g_s * (att @ (conj(g) v))
#     -- the exact same map, zero S^2 work inside attention, flash-eligible.
DRESSED_BF16 = False          # fp32 SDPA; opt.install(bf16=True) flips this for inference

_compiled = {}
def _get_compiled(name, fn):
    if name not in _compiled:
        try:
            _compiled[name] = torch.compile(fn, dynamic=False)
        except Exception:
            _compiled[name] = fn
    return _compiled[name]

# The dressed-attention pre/post pointwise chains (rope phase mul, conj(g) dressing,
# bf16 cast, layout permute / float+nan_to_num, re-dressing) run as torch.compile'd
# REAL-VIEW leaves. The complex multiplies are expanded to real components (dynamo cannot
# trace view_as_complex), and the combined phase w = rope*conj(g) is applied as ONE complex
# mul instead of two sequential ones -- same map up to fp32 reassociation.
ATTN_FUSEPRE = True           # fused dress pre/post chains

def _dress_pre(qkvR, phR, phI, gR, gI, to_bf16: bool):
    # qkvR [B,S,3,h,dc,2] fp32; ph [S,dc]; g [B,S]. Returns q,k,v [B,h,S,2dc].
    a, b = qkvR[..., 0], qkvR[..., 1]                     # [B,S,3,h,dc]
    ph_r = phR[None, :, None, :]                          # [1,S,1,dc]
    ph_i = phI[None, :, None, :]
    g_r = gR[:, :, None, None]                            # [B,S,1,1]
    g_i = gI[:, :, None, None]
    wqk_r = ph_r * g_r + ph_i * g_i                       # rope * conj(g)
    wqk_i = ph_i * g_r - ph_r * g_i
    wv_r, wv_i = g_r.expand_as(wqk_r), (-g_i).expand_as(wqk_i)
    wr = torch.stack([wqk_r, wqk_r, wv_r], dim=2)         # [B,S,3,1,dc], bcast over h
    wi = torch.stack([wqk_i, wqk_i, wv_i], dim=2)
    o_r = a * wr - b * wi                                 # [B,S,3,h,dc]
    o_i = a * wi + b * wr
    out = torch.stack([o_r, o_i], dim=-1)                 # [B,S,3,h,dc,2]
    out = out.reshape(*out.shape[:-2], out.shape[-2] * 2) # [B,S,3,h,2dc]
    if to_bf16:
        out = out.to(torch.bfloat16)
    out = out.permute(0, 2, 3, 1, 4)                      # [B,3,h,S,2dc]
    return out[:, 0].contiguous(), out[:, 1].contiguous(), out[:, 2].contiguous()

def _dress_post(o, gR, gI):
    # o [B,h,S,2dc] (bf16 or fp32) -> re-dressed real view [B,S,h,dc,2] fp32.
    o = torch.nan_to_num(o.float(), nan=0.0)
    o = o.permute(0, 2, 1, 3)                             # [B,S,h,2dc]
    re, im = o[..., 0::2], o[..., 1::2]                   # interleaved pairs
    g_r = gR[:, :, None, None]
    g_i = gI[:, :, None, None]
    rr = re * g_r - im * g_i                              # (re+i im)*(gR+i gI)
    ri = re * g_i + im * g_r
    return torch.stack([rr, ri], dim=-1).contiguous()     # [B,S,h,dc,2]


def _swiglu_gate_pre(preR, vR):
    """silu(pre) * v on the real view; the signed gate pre-activation is computed by the
    caller (GaugeSwiGLU._signed_pre)."""
    return F.silu(preR).unsqueeze(-1) * vR


# ---------------------------------------------------------------- complex view
def to_c(h: torch.Tensor) -> torch.Tensor:
    """[..., 2n] real -> [..., n] complex via adjacent pairs (2c, 2c+1). Preserves fp64."""
    if h.dtype not in (torch.float32, torch.float64):
        h = h.float()
    return torch.view_as_complex(h.reshape(*h.shape[:-1], h.shape[-1] // 2, 2).contiguous())


def to_r(hc: torch.Tensor) -> torch.Tensor:
    """[..., n] complex -> [..., 2n] real, inverse of to_c."""
    return torch.view_as_real(hc).reshape(*hc.shape[:-1], hc.shape[-1] * 2)


def apply_phase(hc: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """h_s -> e^{i phi_s} h_s.  hc [B,S,C] complex, phi [B,S] or [S] real."""
    return hc * torch.exp(1j * phi.to(hc.real.dtype)).unsqueeze(-1)


def gauge_quotient_d2(ya: torch.Tensor, yb: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
    """min_phi ||a_s - e^{i phi_s} b_s||^2 per TOKEN -- the phase-independent distance
    under this model's token-local U(1)^S.  ya, yb real [B,S,H] -> [B,S].

        d_G^2([a],[b])_s = |a_s|^2 + |b_s|^2 - 2 |<a_s, b_s>_C|

    The phase is ONE scalar per token shared across ALL H/2 complex channels (see this
    module's header), NOT one per 4-cell, so the inner product is summed over the WHOLE token
    before taking |.|.  Summing per-cell instead would quotient a much larger group than the
    model actually has and would discard real information.

    `normalize` divides by 2|a|^2, which maps to [0, 1] when |a| = |b|.  No sqrt: this is a
    squared distance, so it is smooth at coincidence.
    """
    ca, cb = to_c(ya), to_c(yb)
    ip = (ca.conj() * cb).sum(-1)                     # [B,S] complex -- one phase per token
    na = ca.abs().pow(2).sum(-1)
    d2 = na + cb.abs().pow(2).sum(-1) - 2.0 * ip.abs()
    return d2 / (2.0 * na).clamp_min(1e-12) if normalize else d2


# ---------------------------------------------------------------- linear
class GLinear(nn.Module):
    """Complex-linear map, no bias. Commutes with any per-token phase exactly."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # LeCun-style init on the underlying real dof (2*in_c real inputs per complex out)
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_c, in_c, 2), std=1.0 / math.sqrt(2 * in_c)))

    def __deepcopy__(self, memo):
        # The blockreal P cache (_P) is a NON-LEAF tensor carrying autograd graph;
        # deepcopy (EMA switch at eval steps) refuses those. Drop cache keys and
        # deepcopy the rest normally — P is rebuilt on demand (keyed by
        # weight._version + grad mode, so the copy regenerates its own).
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("_P", "_p_ver", "_p_shape")}
        new.__dict__.update(_copy.deepcopy(d, memo))
        return new

    def wc(self) -> torch.Tensor:
        return torch.view_as_complex(self.weight)

    def forward(self, hc: torch.Tensor) -> torch.Tensor:
        if not hc.is_complex():
            return hc @ self.wc().t().to(hc.dtype)
        # Interleaved block-real GEMM. The packed weight P depends only on
        # the weight tensor, so it is cached per weight._version (rebuilt after any
        # optimizer step; grad mode is part of the key so an eval-built P is never
        # reused by a training forward).
        ver = (self.weight._version, hc.real.dtype, False,
               torch.is_grad_enabled() and self.weight.requires_grad)
        if getattr(self, "_p_ver", None) != ver:
            w = self.weight.to(hc.real.dtype)                 # [out_c, in_c, 2]
            out_c, in_c, _ = w.shape
            w_re, w_im = w[..., 0], w[..., 1]
            packed = torch.stack([torch.stack([w_re, w_im], -1),
                                  torch.stack([-w_im, w_re], -1)], -2)
            packed = packed.permute(1, 2, 0, 3).reshape(2 * in_c, 2 * out_c)
            self._P = packed
            self._p_shape = (out_c, in_c)
            self._p_ver = ver
        out_c, in_c = self._p_shape
        x_real = torch.view_as_real(hc).reshape(*hc.shape[:-1], 2 * in_c)
        # fp32 GEMM pinned regardless of any surrounding autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            y_real = x_real.float() @ self._P
        return torch.view_as_complex(y_real.view(*hc.shape[:-1], out_c, 2))


# ---------------------------------------------------------------- connection
def gram_connection_parts(hc: torch.Tensor, eps: float):
    """Normalised Gram connection U[s,t] = <h_t,h_s> / |<h_t,h_s>| (zero where |.| <= eps),
    returned as real parts (ur, ui) and the live mask.
    Gr[s,t] = Re<h_t,h_s> = sum hr_s hr_t + hi_s hi_t  -> packed real GEMM;
    Gi[s,t] = Im<h_t,h_s> = sum hi_s hr_t - hr_s hi_t  -> packed real GEMM with (hi, -hr)."""
    B, S, C = hc.shape
    hR = torch.view_as_real(hc)                               # [B,S,C,2] contiguous
    h2 = hR.reshape(B, S, 2 * C)
    ht = torch.stack([hR[..., 1], -hR[..., 0]], -1).reshape(B, S, 2 * C)
    Gr = h2 @ h2.transpose(1, 2)
    Gi = ht @ h2.transpose(1, 2)
    # clamp keeps backward finite at exactly-zero Gram entries (0-grad x inf-slope = NaN
    # without it; complex abs() avoids this via its sgn subgradient). Forward: 1e-15 << eps,
    # so masking is unchanged.
    mag = torch.sqrt((Gr.square() + Gi.square()).clamp_min(1e-30))
    mask = mag > eps
    inv = torch.where(mask, 1.0 / mag.clamp_min(eps), torch.zeros((), dtype=mag.dtype, device=mag.device))
    return Gr * inv, Gi * inv, mask


# ---------------------------------------------------------------- rope (interleaved)
class PairRope(nn.Module):
    """RoPE as a diagonal complex phase in the SAME complex structure as the gauge.
    Commutes exactly with per-token phases (both are diagonal complex scalars)."""

    def __init__(self, dim_c: int, max_pos: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_c, dtype=torch.float32) / dim_c))
        ang = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
        self.phase = nn.Buffer(torch.polar(torch.ones_like(ang), ang), persistent=False)  # [P, dim_c]

    def forward(self, qc: torch.Tensor) -> torch.Tensor:
        """qc [B,S,H,dc] complex -> rotated."""
        S = qc.shape[1]
        return qc * self.phase[:S].to(qc.dtype)[None, :, None, :]


# ---------------------------------------------------------------- attention
class GaugeAttention(nn.Module):
    """Transported attention: scores Re(U_{s<-t} (q_s^dag k_t))/sqrt(d); values sum_t p U v.
    Scores are gauge-INVARIANT, the output is gauge-COVARIANT."""

    def __init__(self, hidden_c: int, num_heads: int, max_pos: int, rope_theta: float,
                 inner_c: int = None, xroute_dim: int = 0,
                 xroute_pel: int = 0, xroute_grid: int = 0, xroute_tag: str = "b0"):
        super().__init__()
        # inner width decoupled from state width: inner_c=2*hidden_c (512_c) makes each
        # projection 2*256*512 = 262,144 real params == a real 512x512 projection exactly.
        # Symmetry unaffected (any complex-linear map commutes with the token phase);
        # state dimension unchanged.
        inner_c = inner_c or hidden_c
        self.h = num_heads
        self.dc = inner_c // num_heads                        # complex dims per head
        self.qkv = GLinear(hidden_c, 3 * inner_c)
        self.o = GLinear(inner_c, hidden_c)
        self.rope = PairRope(self.dc, max_pos, rope_theta)
        self.scale = 1.0 / math.sqrt(2 * self.dc)             # 1/sqrt(head_dim_real)
        # Input-routed attention (xroute_dim > 0):
        #   l_st = l_st^state + (Q_x x_s)^T (K_x x_t)/sqrt(d_x) + b_{dr,dc}
        # Both added terms are built from the NEUTRAL x and from positions, so both are
        # gauge-invariant -- the score stays invariant and the value aggregation is
        # untouched (it still uses the transported charged values). Sparse-edit tasks need
        # routing decided by small differences in the RAW input, which the current scores
        # can only see after several nonlinear state updates.
        self.xdim = int(xroute_dim)
        if self.xdim > 0:
            self.x_pel = int(xroute_pel)
            self.x_grid = int(xroute_grid)
            hw = int(round(self.x_grid ** 0.5))
            assert hw * hw == self.x_grid, (self.x_grid, hw)
            self.x_hw = hw
            # both factors of the product get a small NONZERO init: zeroing both would give
            # identically zero gradient on both and the branch would never leave the origin.
            self.x_q = new_param(self.h * self.xdim, 2 * hidden_c,
                                 init=lambda p, g: p.normal_(0.0, (2 * hidden_c) ** -0.5,
                                                             generator=g),
                                 name=f"xroute.{xroute_tag}.q")
            self.x_k = new_param(self.h * self.xdim, 2 * hidden_c,
                                 init=lambda p, g: p.normal_(0.0, (2 * hidden_c) ** -0.5,
                                                             generator=g),
                                 name=f"xroute.{xroute_tag}.k")
            # 2D relative-position bias over the GRID; one extra bucket for any pair that
            # touches the puzzle-emb prefix (no (dr,dc) is defined there). Zero-init is safe
            # here: it is an additive term, not a factor, so its gradient is nonzero at 0.
            self.x_rel = new_param(self.h, (2 * hw - 1) ** 2 + 1,
                                   init=lambda p, g: p.zero_(),
                                   name=f"xroute.{xroute_tag}.rel")
            self.register_buffer("_x_bucket", torch.empty(0, dtype=torch.long),
                                 persistent=False)

    def _x_buckets(self, S, device):
        """[S,S] long: 2D relative-position bucket, with one catch-all for prefix pairs."""
        if self._x_bucket.numel() == S * S:
            return self._x_bucket
        hw, pel = self.x_hw, self.x_pel
        n = (2 * hw - 1) ** 2
        b = torch.full((S, S), n, dtype=torch.long, device=device)   # prefix pairs -> last
        idx = torch.arange(pel, min(S, pel + self.x_grid), device=device)
        r = (idx - pel) // hw
        c = (idx - pel) % hw
        dr = r[:, None] - r[None, :] + (hw - 1)
        dc = c[:, None] - c[None, :] + (hw - 1)
        b[pel:pel + len(idx), pel:pel + len(idx)] = dr * (2 * hw - 1) + dc
        self._x_bucket = b
        return b

    def _x_factors(self, xc):
        """The rank-`xdim` factors of `_x_logits`, WITHOUT forming [B,h,S,S].

        Returns (qx, kx, x_rel, buckets, scale) with qx/kx [B,h,S,xdim]. At the maze shape
        the assembled logits would be 2.4 GiB per call at B=96 while these factors are ~45 MB, which
        is the entire reason the sdpa path takes the factors instead.

        `_x_logits` is defined as the composition of these below, so the eager reference and
        the sdpa path cannot drift: a change here changes both."""
        xr = torch.view_as_real(xc).flatten(-2)                      # [B,S,2C], x neutral
        B, S = xr.shape[0], xr.shape[1]
        xr = xr.to(self.x_q.dtype)
        qx = F.linear(xr, self.x_q).view(B, S, self.h, self.xdim).transpose(1, 2)
        kx = F.linear(xr, self.x_k).view(B, S, self.h, self.xdim).transpose(1, 2)
        return qx, kx, self.x_rel, self._x_buckets(S, xr.device), self.xdim ** -0.5

    def _x_logits(self, xc):
        """(Q_x x_s)^T (K_x x_t)/sqrt(d_x) + b_{dr,dc} -> [B,h,S,S], gauge-INVARIANT."""
        qx, kx, xrel, buck, xsc = self._x_factors(xc)
        return (qx @ kx.transpose(-1, -2)) * xsc + xrel[:, buck].unsqueeze(0)

    def _x_sdpa_extras(self, xc, dt):
        """Input routing for the DRESSED sdpa path: (q_extra, k_extra, mask_add), exactly.

        The routing logit decomposes into two sdpa-native pieces, so the dressed path
        carries it with NO custom attention kernel and NO [B,h,S,S] tensor:
          * rank-xdim dot   -> extra head dims concatenated onto the DRESSED q/k. Appended
            AFTER dressing because x is neutral: conj(g) must never touch the routing factors,
            and this way it cannot. sdpa multiplies q.k by self.scale BEFORE adding the mask,
            while the eager reference adds this term post-scale -- so xsc/scale is folded
            into the q factor here, cancelling sdpa's scale exactly.
          * relative table  -> xrelt[h,m,n] = x_rel[h, bucket] as a [1,h,S,S] additive
            attn_mask, batch-free (26.8 MB at the maze shape), broadcast by sdpa; d(x_rel)
            flows through torch's index backward.
        Both pieces are invariant, so the dressed path's equivariance argument is unchanged."""
        qx, kx, xrel, bk, xsc = self._x_factors(xc)
        return ((qx * (xsc / self.scale)).to(dt), kx.to(dt),
                xrel[:, bk].unsqueeze(0).to(dt))

    def forward(self, hc, Uri=None, xc=None):
        # Input routing rides the dressed sdpa exactly (see _x_sdpa_extras): extra head
        # dims + an additive relative-position mask; refuse a silent drop of the routing term.
        if self.xdim > 0 and xc is None:
            raise ValueError("gauge_xroute_dim>0 but xc was not passed to attention")
        B, S, C = hc.shape
        qkv = self.qkv(hc).view(B, S, 3, self.h, self.dc)
        if (ATTN_FUSEPRE and Uri is not None and len(Uri) == 4
                and Uri[3] is not None):
            # fused real-view pre/post chains (compiled leaves): rope phase, conj(g)
            # dressing, bf16 cast and the [B,h,S,2dc] layout land in ONE pass over
            # the raw qkv projection (the largest activations in the model), instead
            # of ~14 separate full-tensor passes. Combined phase w = rope*conj(g) is
            # one complex mul (differs from the sequential form only by fp32 reassociation).
            g = Uri[3]                                        # [B,S] unit phases, stop-grad
            am = None if Uri[2] is None else Uri[2].unsqueeze(1)
            ph = self.rope.phase[:S]
            fn = _get_compiled("dress_pre", _dress_pre)
            qh, kh, vh = fn(torch.view_as_real(qkv), ph.real.contiguous(),
                            ph.imag.contiguous(), g.real.contiguous(),
                            g.imag.contiguous(), DRESSED_BF16)
            if self.xdim > 0:
                # input routing: extra head dims + batch-free additive mask
                qe, ke, xm = self._x_sdpa_extras(xc, qh.dtype)
                qh = torch.cat([qh, qe], -1)
                kh = torch.cat([kh, ke], -1)
                am = xm if am is None else am + xm
            o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=am, scale=self.scale)
            fn2 = _get_compiled("dress_post", _dress_post)
            out = torch.view_as_complex(fn2(o, g.real.contiguous(), g.imag.contiguous()))
            return self.o(out.reshape(B, S, self.h * self.dc))
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(q); k = self.rope(k)
        if Uri is not None and len(Uri) == 4 and Uri[3] is not None:
            g = Uri[3]                                        # [B,S] unit phases, stop-grad
            am = None if Uri[2] is None else Uri[2].unsqueeze(1)
            gc = g.conj().unsqueeze(-1).unsqueeze(-1)         # [B,S,1,1]
            qh = torch.view_as_real(q * gc).reshape(B, S, self.h, 2 * self.dc)
            kh = torch.view_as_real(k * gc).reshape(B, S, self.h, 2 * self.dc)
            vh = torch.view_as_real(v * gc).reshape(B, S, self.h, 2 * self.dc)
            qh, kh, vh = (t.permute(0, 2, 1, 3) for t in (qh, kh, vh))
            # all-live mask -> None (flash-eligible); checked once per U build upstream
            if DRESSED_BF16:
                qh, kh, vh = qh.to(torch.bfloat16), kh.to(torch.bfloat16), vh.to(torch.bfloat16)
            if self.xdim > 0:
                # input routing: extra head dims + batch-free additive mask
                qe, ke, xm = self._x_sdpa_extras(xc, qh.dtype)
                qh = torch.cat([qh, qe], -1)
                kh = torch.cat([kh, ke], -1)
                am = xm if am is None else am + xm
            o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=am, scale=self.scale)
            o = torch.nan_to_num(o.float(), nan=0.0)          # fully-masked rows -> zero
            o = o.permute(0, 2, 1, 3).reshape(B, S, self.h, self.dc, 2)
            out = torch.view_as_complex(o.contiguous()) * g.unsqueeze(-1).unsqueeze(-1)
            return self.o(out.reshape(B, S, self.h * self.dc))
        raise RuntimeError(
            "GaugeAttention (release) runs only the dressed pure-gauge transport "
            "Uri=(ur, ui, amask, g); no dense-U path is shipped.")


# ---------------------------------------------------------------- conv
class GaugeConv(nn.Module):
    """Transported depthwise complex conv on the grid tokens (prefix bypass handled by caller).
    conv1d: causal K taps on raster order (maze, Mini-ARC).  conv2d: k x k on the hw x hw grid (sudoku).
    Message from a degenerate pair is zero via U's built-in mask."""

    def __init__(self, hidden_c: int, conv_type: str, k: int):
        super().__init__()
        self.type = conv_type
        self.k = k
        if conv_type == "conv1d":
            self.offsets = [(j,) for j in range(k)]           # taps s-j, causal
        else:
            r = k // 2
            self.offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.kern = nn.Parameter(
            trunc_normal_init_(torch.empty(len(self.offsets), hidden_c, 2), std=1.0 / math.sqrt(len(self.offsets))))

    def forward(self, hc_grid, g_tok, mask_grid=None):
        """hc_grid [B,Sg,C]; g_tok [B,Sg] unit phases; mask_grid [B,Sg,Sg] bool or None.
        Taps use U[s,t] = g_s conj(g_t) mask[s,t] without materializing the dense matrix."""
        B, Sg, C = hc_grid.shape
        kern = torch.view_as_complex(self.kern).to(hc_grid.dtype)     # [K, C]
        out = torch.zeros_like(hc_grid)
        def uoff(idx, src, ok):
            u = g_tok[:, idx] * g_tok[:, src].conj()
            if mask_grid is not None:
                u = u * mask_grid[:, idx, src]                # None => established all-live
            return u * ok
        if self.type == "conv1d":
            idx = torch.arange(Sg, device=hc_grid.device)
            for j, (off,) in enumerate(self.offsets):
                src = idx - off
                ok = src >= 0
                srcc = src.clamp_min(0)
                Uoff = uoff(idx, srcc, ok)                     # [B,Sg]; 0 where invalid/masked
                out = out + kern[j] * (Uoff.unsqueeze(-1) * hc_grid[:, srcc, :])
        else:
            hw = int(math.isqrt(Sg))
            yy, xx = torch.meshgrid(torch.arange(hw, device=hc_grid.device),
                                    torch.arange(hw, device=hc_grid.device), indexing="ij")
            yy = yy.reshape(-1); xx = xx.reshape(-1)
            idx = torch.arange(Sg, device=hc_grid.device)
            for j, (dy, dx) in enumerate(self.offsets):
                sy = yy + dy; sx = xx + dx
                ok = (sy >= 0) & (sy < hw) & (sx >= 0) & (sx < hw)
                src = (sy.clamp(0, hw - 1) * hw + sx.clamp(0, hw - 1))
                Uoff = uoff(idx, src, ok)
                out = out + kern[j] * (Uoff.unsqueeze(-1) * hc_grid[:, src, :])
        return out


# ---------------------------------------------------------------- invariant features
def neutral_feats(hc_proj: torch.Tensor, xc_proj: Optional[torch.Tensor]) -> torch.Tensor:
    """Charge-neutral per-token scalars from small projections.
    hc_proj [B,S,r] (charged), xc_proj [B,S,r] (neutral) or None.
    Returns [B,S,F] real: within-token channel Gram Re/Im(h_a conj(h_b)) (r^2 each)
    + if x given: t = conj(x)*h per channel, then Re/Im(t_a conj(t_b)) (r^2 each) --
    h-charge cancels in every term."""
    g = hc_proj.unsqueeze(-1) * hc_proj.conj().unsqueeze(-2)          # [B,S,r,r]
    feats = [g.real.flatten(-2), g.imag.flatten(-2)]
    if xc_proj is not None:
        t = xc_proj.conj() * hc_proj                                   # [B,S,r] charged e^{i phi}
        tg = t.unsqueeze(-1) * t.conj().unsqueeze(-2)                  # neutral
        feats += [tg.real.flatten(-2), tg.imag.flatten(-2)]
    return torch.cat(feats, dim=-1)


def soft_dressing(xc: torch.Tensor, hc: torch.Tensor, Wd: "GLinear", eps_soft: float,
                  wx: torch.Tensor = None, groups: int = 1):
    """Charge-dressing factor for the input injection, singularity-free.

    c_s = <W_d x_s, h_s> = sum_c conj(W_d x)_c h_c   (x neutral, h charge +1  =>  c charge +1)
    D_s = c_s / (|c_s| + eps_soft)

    - covariant: D_s -> e^{i phi_s} D_s exactly
    - |D_s| ~= 1 away from degeneracy, -> 0 smoothly as |c_s| -> 0 (injection auto-vanishes;
      NO undefined phase, NO U=1 fallback, bounded gradients)
    Returns (D [B,S] complex, |c| [B,S] real, kept as a diagnostic).
    """
    prod = (Wd(xc) if wx is None else wx).conj() * hc
    if groups == 1:
        c = prod.sum(-1)                                   # [B,S] (exact original path)
    else:
        # G per-channel-group gates: each c^g is charge +1 (same proof as G=1, the
        # sum just runs over a channel subset), so equivariance is exact per token.
        C = prod.shape[-1]
        c = prod.reshape(*prod.shape[:-1], groups, C // groups).sum(-1)   # [B,S,G]
    mag = c.abs()
    D = c / (mag + eps_soft)
    return D, mag


class GaugeSwiGLU(nn.Module):
    """Gauge-SwiGLU with a signed relative-phase gate.

    The standard SwiGLU's learned all-to-all gate projection is kept; the gate input is the
    INVARIANT signed pre-activation u (see `_signed_pre`), so the gate is real+invariant
    and the gated value branch stays exactly equivariant.  At inter_c=1536 each of
    g/v/down carries 2*256*1536 = 786,432 real params -- identical to a real SwiGLU's
    gate/up [1536,512] and down [512,1536].
    """

    def __init__(self, hidden_c: int, inter_c: int, tag: str = "b0",
                 gate_anchors: int = 0, gate_anchor_eps: float = 1e-3):
        super().__init__()
        self.g = GLinear(hidden_c, inter_c)
        self.v = GLinear(hidden_c, inter_c)
        self.down = GLinear(inter_c, hidden_c)
        # Signed relative-phase gate. |W_g h| is invariant but non-negative -- using only
        # the magnitude is NOT something the symmetry forces. With complex anchors a_r,
        #     u_j = Re[conj(c_{r(j)}) (W_g h)_j] / sqrt(|c_{r(j)}|^2 + eps^2),  c_r = a_r^H h
        # is invariant (h -> g h sends both c and W_g h to g(.), so conj(g)g = 1 cancels) and
        # SIGNED, so the nonlinearity gets strictly more information inside the same symmetry.
        # Channels are split into `gate_anchors` contiguous groups, one anchor each.
        self.gate_anchors = int(gate_anchors)
        self.gate_anchor_eps = float(gate_anchor_eps)
        assert self.gate_anchors > 0, "release ships the V03 signed relative-phase gate only"
        assert inter_c % self.gate_anchors == 0, (inter_c, self.gate_anchors)
        self.gate_anchor = new_param(
            self.gate_anchors, hidden_c, 2,
            init=lambda p, g: p.normal_(0.0, hidden_c ** -0.5, generator=g),
            name=f"gate.{tag}.anchor")

    def _signed_pre(self, hc):
        """u_j = Re[conj(c_{r(j)}) (W_g h)_j] / sqrt(|c_{r(j)}|^2 + eps^2)."""
        gcx = self.g(hc)                                            # [.., IC] complex
        a = torch.view_as_complex(self.gate_anchor.to(hc.real.dtype))   # [R, C]
        c = torch.einsum("rc,...c->...r", a.conj(), hc)              # [.., R]
        den = torch.sqrt(c.abs().square() + self.gate_anchor_eps ** 2)
        gg = gcx.reshape(*gcx.shape[:-1], self.gate_anchors, -1)     # [.., R, IC/R]
        u = (c.conj().unsqueeze(-1) * gg).real / den.unsqueeze(-1)
        return u.reshape(*gcx.shape)                                 # [.., IC] real

    def forward(self, hc):
        # The pre-activation is already real and signed; one compiled leaf fuses the
        # SiLU-gate * value elementwise chain.
        pre = self._signed_pre(hc)
        v = self.v(hc)
        fn = _get_compiled("swiglu_pre", _swiglu_gate_pre)
        gvR = fn(pre.to(v.real.dtype), torch.view_as_real(v))
        return self.down(torch.view_as_complex(gvR.contiguous()))


# ---------------------------------------------------------------- readout
class GaugeReadout(nn.Module):
    """Invariant readout: AKOrN-style readout norms + channel-Gram + x-referenced neutral
    pairs.  r(g h; x) = r(h; x) exactly."""

    def __init__(self, hidden_c: int, out_dim: int, k_ro: int = 8, r: int = 8,
                 hidden: int = 256, final_bias: Optional[float] = None,
                 feat_norm: bool = False, norm_eps: float = 1e-5, tag: str = "lm"):
        super().__init__()
        self.out_dim = out_dim
        self.k_ro = k_ro
        # feat_norm: gain-free RMSNorm on the INVARIANT FEATURE VECTOR before the MLP.
        #   Keeps every rich feature (readout norms + channel Gram + x-referenced pairs)
        #   but removes their scale, which otherwise grows until stablemax CE saturates
        #   (dL/dlogits ~ 0 => whole trunk starved).
        #   Gain-free => unlike a learned temperature, no weight can re-inflate the scale.
        #   RMSNorm of invariants is invariant, so the gauge property is untouched.
        self.feat_norm = feat_norm
        self.norm_eps = norm_eps
        self.ak = GLinear(hidden_c, out_dim * k_ro)
        self.ph = GLinear(hidden_c, r)
        self.px = GLinear(hidden_c, r)
        self.feat_dim = out_dim + 4 * r * r
        self.mlp = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out_dim))
        if final_bias is not None:
            with torch.no_grad():
                self.mlp[-1].weight.zero_()
                self.mlp[-1].bias.fill_(final_bias)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """h, x [.., 2C] real -> logits [.., out_dim] fp32."""
        hc, xc = to_c(h), to_c(x)
        a = self.ak(hc).view(*hc.shape[:-1], self.out_dim, self.k_ro)
        norms = a.abs().square().sum(-1)                        # [.., out_dim] invariant
        feats = torch.cat([norms, neutral_feats(self.ph(hc), self.px(xc))], dim=-1)
        if self.feat_norm:
            feats = rms_norm(feats, variance_epsilon=self.norm_eps)
        return self.mlp(feats.to(self.mlp[0].weight.dtype))
