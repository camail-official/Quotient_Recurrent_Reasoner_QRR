"""Token-local U(1) gauge-equivariant QRR trunk (h-only gauge, x neutral).

Mirrors a standard pre-norm transformer block flow exactly: [conv -> dressed inject ->
2x(attn, mlp) -> optional output rms_norm], with a pair-tied residual-scaling algebra.

Transport: the connection is the best pure-gauge factorization
U = g g^H of the state Gram matrix (spectral angular synchronization, PG_ITERS masked
power iterations). The dense S^2 transport is never materialized -- attention consumes
the rank-1 phase field g directly as dressed standard SDPA (see gauge_layers), and the
conv builds its taps from g_s conj(g_t). g is stop-grad by the pure-gauge law.
"""
import math
import torch
import torch.nn as nn

from models.layers import rms_norm
import models.gauge_fuse_pw as GFP
from models.gauge_layers import (to_c, to_r, GaugeAttention, GaugeConv, GaugeSwiGLU,
                                 GLinear, soft_dressing, gram_connection_parts)

PG_ITERS = 30      # power-iteration depth for the pure-gauge factorization


def _pg_start(hc, conn_eps, dtype):
    """Power-iteration start vector for the pure-gauge factorization U ~ g g^H.

    `ones` is NOT gauge-covariant.  Under h_s -> e^{i phi_s} h_s the connection transforms
    as M -> D M D^H (D = diag(e^{i phi})), so the iteration v <- normalize(M v) satisfies
    M'(D v) = D M D^H D v = D (M v): it is covariant at EVERY k **provided the start is**.
    With `ones` the eigenvector overlaps <u_i, ones> are not gauge-invariant, so a TRUNCATED
    iteration lands on a different eigenvector mixture in each gauge and g g^H stops being
    covariant.  Measured one-step defect 0.462 at PG_ITERS = 30, decaying only as
    |lam2/lam1|^k with |lam2/lam1| ~ 0.994-0.999.

    The anchor phase g_s = phase<h_0, h_s> carries charge +1 up to the GLOBAL phase
    e^{-i phi_0}, which cancels identically in g g^H.  Starting there makes the truncated
    iteration exactly covariant AND leaves it converging to the same dominant eigenvector,
    at the cost of one extra einsum.

    NOTE the degenerate-token fallback to `ones` (|ip| <= conn_eps) is itself not covariant;
    it is kept for parity with the trained checkpoints, on which it does not fire.
    """
    ip = torch.einsum("bc,bsc->bs", hc[:, 0].conj(), hc)
    v = ip / ip.abs().clamp_min(conn_eps)
    v = torch.where(ip.abs() <= conn_eps, torch.ones_like(v), v)
    return v.to(dtype)


class GaugeBlock(nn.Module):
    def __init__(self, cfg, hidden_c, max_pos, layer_idx: int = 0):
        super().__init__()
        # Input routing: xroute_grid is the number of GRID tokens (seq_len), xroute_pel the
        # prefix length -- both needed to build the 2D relative-position buckets.
        self.attn = GaugeAttention(hidden_c, cfg.num_heads, max_pos, cfg.rope_theta,
                                   inner_c=int(getattr(cfg, "gauge_attn_inner_c", 2 * hidden_c)),
                                   xroute_dim=int(getattr(cfg, "gauge_xroute_dim", 0)),
                                   xroute_pel=int(getattr(cfg, "puzzle_emb_len", 0) or 0),
                                   xroute_grid=int(cfg.seq_len),
                                   xroute_tag=f"b{layer_idx}")
        # Signed relative-phase gate (gate_anchors > 0 is required).
        self.mlp = GaugeSwiGLU(hidden_c, int(getattr(cfg, "gauge_inter_c", 1536)),
                               tag=f"b{layer_idx}",
                               gate_anchors=int(getattr(cfg, "gauge_gate_anchors", 0)),
                               gate_anchor_eps=float(getattr(cfg, "gauge_gate_anchor_eps", 1e-3)))
        self.norm_eps = cfg.rms_norm_eps

    def forward(self, hc, xc, a1, b1, Uri=None):
        # pre-norm on the REAL view (rms over all 512 dims: exactly invariant)
        _nrm = GFP.norm_c if GFP.FUSE_NORM else (
            lambda h, eps: to_c(rms_norm(to_r(h), variance_epsilon=eps)))
        _res = GFP.residual_update if GFP.FUSE_RES else (lambda h, f, a, b: a * h + b * f)
        hn = _nrm(hc, self.norm_eps)
        f_attn = self.attn(hn, Uri=Uri, xc=xc)
        hc = _res(hc, f_attn, a1, b1)
        hn = _nrm(hc, self.norm_eps)
        f_mlp = self.mlp(hn)
        hc = _res(hc, f_mlp, a1, b1)
        return hc


class GaugeFixedPointTransformer(nn.Module):
    def __init__(self, cfg, n_layers: int):
        super().__init__()
        self.cfg = cfg
        hc = cfg.hidden_size // 2
        max_pos = cfg.seq_len + (cfg.puzzle_emb_len if cfg.puzzle_emb_len else 16)
        self.conn_eps = float(getattr(cfg, "gauge_conn_eps", 1e-6))
        self.dress_eps = float(getattr(cfg, "gauge_dress_eps", 1e-3))
        # gauge_covstart: start the pure-gauge power iteration from the covariant anchor phase,
        # making the TRUNCATED iteration exactly equivariant (see _pg_start). The released
        # Sudoku and Mini-ARC configs set this ON; both Maze configs leave it OFF.
        self.covstart = bool(getattr(cfg, "gauge_covstart", False))
        self.conv = None
        if cfg.conv_type in ("conv1d", "conv2d"):
            self.conv = GaugeConv(hc, cfg.conv_type, cfg.conv_kernel_size)
        self.dress = GLinear(hc, hc)
        # c1_reduced_state: second charge+1 input (momentum m_H). m_H rotates exactly like y,
        # so it needs NO soft_dressing -- dressing exists only because x is neutral. A GLinear
        # is charge-preserving by construction, so equivariance is exact. ZERO-INIT => at init
        # the model is bit-identical to one without this input.
        self.c1 = bool(getattr(cfg, "c1_reduced_state", False))
        if self.c1:
            self.mh_embed = GLinear(hc, hc)
            with torch.no_grad():
                self.mh_embed.weight.zero_()
            self.c1_c_m = float(getattr(cfg, "c1_c_m", 4.3455))
        self.layers = nn.ModuleList(GaugeBlock(cfg, hc, max_pos, layer_idx=_i)
                                    for _i in range(n_layers))
        # pair-tied residual scaling (256 logits -> broadcast over complex channels)
        a1 = math.log(cfg.alpha_1_init / (1 - cfg.alpha_1_init))
        a2 = math.log(cfg.alpha_2_init / (1 - cfg.alpha_2_init))
        self.alpha_1_param = nn.Parameter(a1 * torch.ones(hc)); self.alpha_1_param._no_weight_decay = True
        self.alpha_2_param = nn.Parameter(a2 * torch.ones(hc)); self.alpha_2_param._no_weight_decay = True
        self.norm_placement = cfg.norm_placement
        self.norm_eps = cfg.rms_norm_eps
        self.last_dress_mag = None            # detached |c| of the last dressing (diagnostic)

    def make_seg_ctx(self, input_injection):
        """Per-segment precompute shared by every z-step of one segment: the dress
        GEMM on the (segment-constant) input embedding and the residual-scaling
        algebra. Forward values are bit-identical to the per-call recompute; the
        backward sums the 6 uses into one graph node instead of re-deriving 6
        identical subgraphs."""
        alpha_2 = torch.sigmoid(self.alpha_2_param)
        alpha_1 = torch.sigmoid(self.alpha_1_param)
        L = 2 * len(self.layers)
        a1L = alpha_1.pow(L)
        beta_2 = 1 - alpha_2 * a1L
        beta_1 = beta_2 * (1 - alpha_1) / (1 - a1L + 1e-5)
        return dict(wx=self.dress(to_c(input_injection)), alpha_2=alpha_2,
                    alpha_1=alpha_1, beta_2=beta_2, beta_1=beta_1)

    def forward(self, hidden_states, input_injection, m_h=None, **kwargs):
        # m_h: momentum trunk input, real [B,S,H], charge +1; None when c1_reduced_state
        # is off.
        pel = kwargs.pop("puzzle_emb_len", 0)
        seg_ctx = kwargs.pop("seg_ctx", None)
        kwargs.pop("cos_sin", None); kwargs.pop("outer_c", None)
        hc = to_c(hidden_states)
        xc = to_c(input_injection)
        # One connection per trunk call, from the INPUT state; shared by conv + both layers.
        # The gram carries no autograd graph -- g is stop-grad by the pure-gauge law.
        with torch.no_grad():
            ur, ui, mask = gram_connection_parts(hc, self.conn_eps)
            Mw = mask.to(hc.dtype) * torch.complex(ur, ui)
            v = (_pg_start(hc, self.conn_eps, hc.dtype) if self.covstart else
                 torch.ones(hc.shape[0], hc.shape[1], dtype=hc.dtype, device=hc.device))
            for _ in range(PG_ITERS):
                v = torch.einsum("bst,bt->bs", Mw, v)
                v = v / v.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-30).sqrt()
            del Mw
            g = v / v.abs().clamp_min(1e-30)
        # NEVER store a graph-carrying tensor on the module (deepcopy at the EMA/eval switch
        # refuses non-leaf tensors); the live tensor stays a LOCAL consumed inside this
        # forward, the attribute keeps a detached copy for diagnostics.
        _g_live = g
        self._dress_g = g.detach()
        # conv on grid tokens only (prefix bypass), identity residual around the
        # transported conv; taps read U[s,t] = g_s conj(g_t) mask[s,t] directly.
        if self.conv is not None:
            gr = hc[:, pel:]
            hc = torch.cat([hc[:, :pel], gr + self.conv(gr, g_tok=_g_live[:, pel:],
                                                        mask_grid=mask[:, pel:, pel:])], dim=1)
        # residual scaling algebra, pair-tied. Values are identical across every z-step of
        # a segment (params change only at the optimizer step), so a caller-provided
        # seg_ctx shares one graph node.
        if seg_ctx is not None:
            alpha_2 = seg_ctx["alpha_2"].to(hc.real.dtype)
            alpha_1 = seg_ctx["alpha_1"].to(hc.real.dtype)
            beta_2, beta_1 = seg_ctx["beta_2"].to(hc.real.dtype), seg_ctx["beta_1"].to(hc.real.dtype)
        else:
            alpha_2 = torch.sigmoid(self.alpha_2_param).to(hc.real.dtype)
            alpha_1 = torch.sigmoid(self.alpha_1_param).to(hc.real.dtype)
            L = 2 * len(self.layers)
            a1L = alpha_1.pow(L)
            beta_2 = 1 - alpha_2 * a1L
            beta_1 = beta_2 * (1 - alpha_1) / (1 - a1L + 1e-5)
        # dressed sdpa transport: (ur, ui, mask-or-None, g). The all-live check syncs once
        # per module lifetime; an all-live mask hands sdpa attn_mask=None (flash-eligible).
        if not hasattr(self, "_mask_all_live"):
            self._mask_all_live = bool(mask.all())
        mk = None if self._mask_all_live else mask
        Uri = (ur, ui, mk, _g_live)
        # dressed injection: h <- a2 h + b2 (D_s * x_s); D covariant, smooth at degeneracy
        D, mag = soft_dressing(xc, hc, self.dress, self.dress_eps,
                               wx=None if seg_ctx is None else seg_ctx["wx"])
        self.last_dress_mag = mag.detach()
        Dx = D.unsqueeze(-1)
        hc = (GFP.residual_update(hc, Dx * xc, alpha_2, beta_2)
              if GFP.FUSE_RES else alpha_2 * hc + beta_2 * (Dx * xc))
        # Add the momentum input. Charge +1 like hc, so it joins additively with no
        # dressing; GLinear commutes with the per-token phase exactly. Fixed global rescale
        # (NOT per-token RMSNorm -- ||m_H|| is the speed and is part of the state).
        if self.c1 and m_h is not None:
            hc = hc + self.mh_embed(to_c(m_h) / self.c1_c_m)
        for layer in self.layers:
            hc = layer(hc, xc, alpha_1, beta_1, Uri=Uri)
        out = to_r(hc)
        if self.norm_placement == "output":
            out = rms_norm(out, variance_epsilon=self.norm_eps)
        return out.to(hidden_states.dtype)
