"""Fused pointwise kernels for the gauge trunk (used for complex64 CUDA inputs).

Two exact fusions of the largest pointwise costs in the trunk (the RMS-norm chain and the
residual update):

  norm_c              rms_norm(to_r(hc)) : pow -> mean -> add eps -> rsqrt -> mul   (5+ passes)
                      -> one kernel, one pass, fp32 accumulate.  Identical formula.
  residual_update     a*hc + b*f  (a,b real per COMPLEX channel)  : mul,mul,add     (3 passes)
                      -> one kernel, one pass.  Identical formula.

Both operate on the REAL view [N, D=2C] so autograd never sees a complex op: no Wirtinger /
conjugate convention to get wrong, and view_as_real/view_as_complex are views, not copies.
Backward is closed-form (no recompute):
  norm: dx = r*g - (r^3/D) * x * sum_j(g_j x_j)            with r = rsqrt(mean(x^2)+eps)
  res : dx = a*g, df = b*g, da = sum_rows(g*x), db = sum_rows(g*f)
Parameter grads use per-program partials (no atomics), finished with one small torch reduce.
"""
import torch
import triton
import triton.language as tl

FUSE_NORM = True              # fused complex rms_norm enabled
FUSE_RES = True               # fused residual update enabled
_ROWS = 16                    # rows/program in the residual-update backward (tuned on H100)
_WARPS = 4


# ------------------------------------------------------------------ rms norm
@triton.jit
def _rms_fwd(X, Y, R, D: tl.constexpr, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    x = tl.load(X + row * D + cols, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(var + EPS)
    tl.store(R + row, r)
    tl.store(Y + row * D + cols, x * r, mask=m)


@triton.jit
def _rms_bwd(G, X, R, DX, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    g = tl.load(G + row * D + cols, mask=m, other=0.0).to(tl.float32)
    x = tl.load(X + row * D + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.load(R + row).to(tl.float32)
    gx = tl.sum(g * x, axis=0)
    dx = r * g - (r * r * r / D) * x * gx
    tl.store(DX + row * D + cols, dx, mask=m)


class _RmsNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        xc = x.contiguous()
        shp = xc.shape
        D = shp[-1]
        x2 = xc.reshape(-1, D)
        N = x2.shape[0]
        y = torch.empty_like(x2)
        r = torch.empty(N, dtype=torch.float32, device=x2.device)
        BLOCK = triton.next_power_of_2(D)
        _rms_fwd[(N,)](x2, y, r, D, eps, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x2, r)
        ctx.D = D
        ctx.BLOCK = BLOCK
        return y.reshape(shp)

    @staticmethod
    def backward(ctx, g):
        x2, r = ctx.saved_tensors
        D, BLOCK = ctx.D, ctx.BLOCK
        g2 = g.contiguous().reshape(-1, D)
        dx = torch.empty_like(g2)
        _rms_bwd[(g2.shape[0],)](g2, x2, r, dx, D, BLOCK=BLOCK, num_warps=4)
        return dx.reshape(g.shape), None


def rms_norm_fused(h, variance_epsilon: float):
    """Drop-in for models.layers.rms_norm on fp32 CUDA tensors; falls back otherwise."""
    if not (h.is_cuda and h.dtype == torch.float32):
        from models.layers import rms_norm as _ref
        return _ref(h, variance_epsilon)
    return _RmsNormFn.apply(h, variance_epsilon)


# ------------------------------------------------------------------ residual update
@triton.jit
def _res_fwd(X, F, A, B, O, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    ch = cols // 2                                    # real view: 2 slots per complex channel
    x = tl.load(X + row * D + cols, mask=m, other=0.0)
    f = tl.load(F + row * D + cols, mask=m, other=0.0)
    a = tl.load(A + ch, mask=m, other=0.0)
    b = tl.load(B + ch, mask=m, other=0.0)
    tl.store(O + row * D + cols, a * x + b * f, mask=m)


@triton.jit
def _res_bwd(G, X, F, A, B, DX, DF, PA, PB, N, D: tl.constexpr,
             ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    ch = cols // 2
    a = tl.load(A + ch, mask=m, other=0.0)
    b = tl.load(B + ch, mask=m, other=0.0)
    sa = tl.zeros([BLOCK], dtype=tl.float32)
    sb = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        live = row < N
        off = row * D + cols
        g = tl.load(G + off, mask=m & live, other=0.0)
        x = tl.load(X + off, mask=m & live, other=0.0)
        f = tl.load(F + off, mask=m & live, other=0.0)
        tl.store(DX + off, a * g, mask=m & live)
        tl.store(DF + off, b * g, mask=m & live)
        sa += g * x
        sb += g * f
    tl.store(PA + pid * D + cols, sa, mask=m)
    tl.store(PB + pid * D + cols, sb, mask=m)


class _ResFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f, a, b):
        xc, fc = x.contiguous(), f.contiguous()
        shp = xc.shape
        D = shp[-1]
        x2, f2 = xc.reshape(-1, D), fc.reshape(-1, D)
        o = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(D)
        _res_fwd[(x2.shape[0],)](x2, f2, a, b, o, D, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x2, f2, a, b)
        ctx.D, ctx.BLOCK = D, BLOCK
        return o.reshape(shp)

    @staticmethod
    def backward(ctx, g):
        x2, f2, a, b = ctx.saved_tensors
        D, BLOCK = ctx.D, ctx.BLOCK
        g2 = g.contiguous().reshape(-1, D)
        N = g2.shape[0]
        dx, df = torch.empty_like(g2), torch.empty_like(g2)
        nprog = triton.cdiv(N, _ROWS)
        pa = torch.empty(nprog, D, dtype=torch.float32, device=g2.device)
        pb = torch.empty(nprog, D, dtype=torch.float32, device=g2.device)
        _res_bwd[(nprog,)](g2, x2, f2, a, b, dx, df, pa, pb, N, D,
                           ROWS=_ROWS, BLOCK=BLOCK, num_warps=_WARPS)
        # [nprog, 2C] -> [C]: both real slots of a complex channel share one scale
        da = pa.sum(0).reshape(-1, 2).sum(-1) if a.requires_grad else None
        db = pb.sum(0).reshape(-1, 2).sum(-1) if b.requires_grad else None
        return dx.reshape(g.shape), df.reshape(g.shape), da, db


def residual_update(hc, f, a, b):
    """a*hc + b*f with a,b real [C] and hc,f complex [...,C] -- one pass, exact."""
    if not (hc.is_cuda and hc.is_complex() and hc.dtype == torch.complex64):
        return a * hc + b * f
    xr = torch.view_as_real(hc).reshape(*hc.shape[:-1], hc.shape[-1] * 2)
    fr = torch.view_as_real(f).reshape(*f.shape[:-1], f.shape[-1] * 2)
    o = _ResFn.apply(xr, fr, a.contiguous(), b.contiguous())
    return torch.view_as_complex(o.reshape(*o.shape[:-1], o.shape[-1] // 2, 2).contiguous())


def norm_c(hc, eps: float):
    """rms_norm over the real view of a complex tensor, without the to_r/to_c round-trip."""
    if not (hc.is_cuda and hc.dtype == torch.complex64):
        from models.layers import rms_norm as _ref
        from models.gauge_layers import to_c, to_r
        return to_c(_ref(to_r(hc), variance_epsilon=eps))
    xr = torch.view_as_real(hc).reshape(*hc.shape[:-1], hc.shape[-1] * 2)
    y = _RmsNormFn.apply(xr, eps)
    return torch.view_as_complex(y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2).contiguous())
