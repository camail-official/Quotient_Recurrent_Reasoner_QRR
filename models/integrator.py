import math
import torch
import torch.nn as nn
from models.common import trunc_normal_init_


class _GeoIntegrator(nn.Module):
    """Marker base class; QRRModelInner checks it before enabling the c1_reduced_state
    trunk input."""
    pass


# projection_mode "none": F_H = F and the trunk sees m_H = m (no horizontal projection of
# force or momentum). The rotation R_a still runs and `a` is still computed. The fused
# triton step (gauge_fused_step.py) implements this law with PROJ=False.
PROJECTION_MODES = ("none",)   # the release ships the unprojected law only
# projection_mode "none" runs on the FUSED z-step.
_FUSED_PROJOFF = True


def resolve_projection_mode(config=None):
    mode = str(getattr(config, "projection_mode", "none") or "none")
    assert mode in PROJECTION_MODES, f"projection_mode={mode!r} not in {PROJECTION_MODES}"
    return mode


class QuotientIntegrator(_GeoIntegrator):
    """Order-2 quotient integrator ('qso'): free-amplitude quotient dynamics with an
    optional patience-based step-size controller.

    Per token, y in R^H ~ C^{H/2} via ADJACENT real pairs; J = mult by i.  NO sphere:
    no retraction, no unit-normalisation after the initial draw.

        a     = <F, Jy> / max(||y||^2, q_den_eps)      EXACT ratio (clamp_min, NOT +eps)
        F_H   = F - a Jy                               projection_mode 'full'; F_H = F
                                                       under 'none' (release)
        R     = e^{a J}                                the vertical coefficient IS the
                                                       phase increment; PARAMETER-FREE

      order 2 ('qso'); with c1_reduced_state the trunk also sees RAW m_H:
        beta = sigmoid(b_H)                            ONE learned global scalar
        mbar = beta m_H + (1 - beta) F_H               alpha_H == 1 - beta_H, dependent
        y'   = R (y + eta_used mbar)
        m_H' = R (eta_used mbar)                       carried momentum IS eta*mbar

    ETA ORDERING.  The step is built with eta_used = eta_t, then
    eta_{t+1} = decay*eta_t if adapt else eta_t.  q_eta_used holds eta_t (what actually
    multiplied the proposal), q_eta holds eta_{t+1}.  Executing with the freshly-updated eta
    is wrong, and anything rescaling by the step size must divide by q_eta_used.  With the
    default q_decay_train = q_decay_eval = 1.0 the controller is off and eta stays eta0.

    eta multiplies ONLY the quotient proposal, NEVER R.  a is computed before eta, so R has
    no DIRECT eta factor -- but across steps a previous eta changes (y, m) hence F hence a,
    so the fiber TRAJECTORY differs with the controller on vs off.

    R is EXACTLY a gauge transformation (a invariant, trunk equivariant, readout invariant),
    so for an exactly equivariant trunk R == I gives identical logits/loss/halting/gradients:
    R audits trunk equivariance rather than adding a mechanism.  a is detached before
    cos/sin accordingly.

    Controller state (q_eta, q_eta_used, q_residue, q_exec, q_best, q_patience) is kept
    SEPARATE from the legacy residues/stepsize fields, which stay pinned (inf / 1) so the
    legacy halting predicates in QRRModel.forward never fire.
    """

    def __init__(self, config):
        super().__init__()
        self.mode = str(config.integrator)
        assert self.mode == "qso", self.mode      # release: order-2 QRR integrator only
        self.order = 2
        self.n = int(getattr(config, "osc_n", 4))      # ONLY for the _fresh_y draw
        assert config.hidden_size % 2 == 0
        assert config.hidden_size % self.n == 0
        self.init_std = config.init_std
        self.projection_mode = resolve_projection_mode(config)
        self.proj_force = self.projection_mode == "full"
        self.proj_c1 = self.projection_mode in ("full", "c1_only")
        self.decay_patience = int(getattr(config, "q_patience", 10))

        self.den_eps = float(getattr(config, "q_den_eps", 1e-12))
        self.eps_r = float(getattr(config, "q_eps_r", 1e-8))
        self.eta0 = float(getattr(config, "q_eta0", 1.0))
        self.decay_train = float(getattr(config, "q_decay_train", 1.0))
        self.decay_eval = float(getattr(config, "q_decay_eval", 1.0))
        self.r_thresh = float(getattr(config, "q_rthresh", 0.1))
        self.apply_R = bool(getattr(config, "q_apply_R", True))
        self.use_ckpt = bool(getattr(config, "osc_ckpt", False))
        self.c1_c_m = float(getattr(config, "c1_c_m", 4.3455))
        assert 0.0 < self.decay_train <= 1.0 and 0.0 < self.decay_eval <= 1.0, \
            "eta must never increase"

        b0 = float(getattr(config, "q_beta_H_init", 0.5))
        assert 0.0 < b0 < 1.0, b0
        # config field MUST drive the init -- a hardcoded 0.0 would make it decorative
        self.b_H = nn.Parameter(torch.tensor(math.log(b0 / (1.0 - b0)),
                                             dtype=torch.float32))
        # No weight decay: WD would pull b_H -> 0 i.e. beta_H -> 0.5, which IS the
        # default init, and pin it there.
        self.b_H._no_weight_decay = True
        self._last_a = None            # detached diagnostic

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _c(t):
        return torch.view_as_complex(
            t.reshape(*t.shape[:-1], t.shape[-1] // 2, 2).contiguous())

    @staticmethod
    def _r(z):
        return torch.view_as_real(z).reshape(*z.shape[:-1], z.shape[-1] * 2)

    @staticmethod
    def _hi(t):
        """Promote bf16/fp16 to fp32; NEVER demote fp64 (gauge_layers.to_c idiom)."""
        return t if t.dtype in (torch.float32, torch.float64) else t.float()

    def mh_for_trunk(self, m_flat, y_flat):
        """RAW m_H.  The gauge trunk ALREADY divides by cfg.c1_c_m
        (GaugeFixedPointTransformer.forward), so pre-multiply by the SAME value read from
        the SAME key: the net factor is exactly 1 whatever the yaml says."""
        return m_flat * self.c1_c_m

    def _linf_c(self, u):
        """||u||_{inf,C} = max_{s,c} |u_{s,c}| -> [B].  EXACTLY U(1)^S invariant, unlike
        a real-coordinate L-inf.  Per-token max over channels then max over tokens."""
        p = self._hi(u).reshape(*u.shape[:-1], u.shape[-1] // 2, 2)
        return (p[..., 0].square() + p[..., 1].square()).sqrt().amax(-1).amax(-1)

    def _fresh_y(self, batch_size, seq_len, hidden_size, dtype, device):
        """trunc_normal draw with a per-n-cell unit normalisation applied AT INIT ONLY.
        After step 0 the norm is free."""
        y = trunc_normal_init_(
            torch.empty(batch_size, seq_len, hidden_size, dtype=dtype, device=device),
            std=self.init_std)
        c = y.view(batch_size, seq_len, hidden_size // self.n, self.n)
        return (c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)).reshape(
            batch_size, seq_len, hidden_size)

    def detach_state(self, state: dict):
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach()
        return state

    # -------------------------------------------------------------------- reset
    def reset(self, reset_flag: torch.Tensor, shape: tuple, dtype: torch.dtype,
              device: torch.device, state: dict, reset_metadata: bool = False):
        """Build a NEW state dict: slots with reset_flag get a fresh state, the rest carry
        over.  Every key must be rebuilt here: this runs on every outer step, and at eval
        n_steps == 1 (QRRModel.forward), i.e. before every z-step.  q_eta/q_eta_used are
        fp32 regardless of forward_dtype so a decay like 0.997 never quantises.  NO key
        may be [B,S] -- it would broadcast against f3=[B,1,1] into [B,B,S] silently."""
        B, S, H = shape[0], shape[1], shape[2]
        f1 = reset_flag.view(-1)
        f3 = f1.view(-1, 1, 1)

        y = self._fresh_y(B, S, H, dtype, device)
        residues = torch.inf * torch.ones(B).to(device)
        stepsize = torch.ones(B, 1, 1, dtype=dtype, device=device)
        patience = self.decay_patience * torch.ones(B).to(device)
        iter_idx = torch.zeros(B, dtype=torch.int32, device=device)
        best_residues = torch.inf * torch.ones(B).to(device)
        q_eta = self.eta0 * torch.ones(B, 1, 1, dtype=torch.float32, device=device)
        q_eta_used = q_eta.clone()
        q_residue = torch.inf * torch.ones(B, dtype=torch.float32, device=device)
        q_exec = torch.inf * torch.ones(B, dtype=torch.float32, device=device)
        q_best = torch.inf * torch.ones(B, dtype=torch.float32, device=device)
        q_patience = float(self.decay_patience) * torch.ones(
            B, dtype=torch.float32, device=device)

        def _pack(sel):
            d = dict(y=sel("y", y.contiguous(), f3),
                     residues=sel("residues", residues, f1),
                     stepsize=sel("stepsize", stepsize, f3),
                     patience=sel("patience", patience, f1),
                     iter_idx=sel("iter_idx", iter_idx, f1),
                     best_residues=sel("best_residues", best_residues, f1),
                     q_eta=sel("q_eta", q_eta, f3),
                     q_eta_used=sel("q_eta_used", q_eta_used, f3),
                     q_residue=sel("q_residue", q_residue, f1),
                     q_exec=sel("q_exec", q_exec, f1),
                     q_best=sel("q_best", q_best, f1),
                     q_patience=sel("q_patience", q_patience, f1))
            d["m"] = sel("m", torch.zeros_like(y), f3)
            return d

        if state is None:
            return _pack(lambda k, fresh, m: fresh)
        return _pack(lambda k, fresh, m: fresh if (reset_metadata and k in (
            "residues", "stepsize", "patience", "iter_idx", "best_residues"))
            else torch.where(m, fresh, state[k]))

    # --------------------------------------------------------------------- core
    def _core(self, y_flat, m_flat, eta, z_new):
        """PURE (y, m, eta, z_new) -> (y', m', r_prop, r_exec).

        fp32 CUDA inputs route through the fused triton fwd/bwd pair
        (models/gauge_fused_step.py; same math incl. detached theta and
        ratio-of-board-maxes r stats) whenever the projection mode is fusable.

        eta is an explicit INPUT so the checkpointed recompute is exact.  No data-dependent
        Python branch (the controller's torch.where lives in _finish): pretrain.py
        all-reduces every param whose grad is not None, so values may differ across ranks
        but control flow must not."""
        from models.gauge_fused_step import FUSED as _FUSED, fused_core as _fused
        _fusable_proj = self.proj_force or (_FUSED_PROJOFF and not self.proj_c1)
        if (_FUSED and _fusable_proj and m_flat is not None
                and y_flat.is_cuda and y_flat.dtype == torch.float32):
            return _fused(self, y_flat, m_flat, eta, z_new)
        dt = y_flat.dtype
        if z_new.dtype != dt:
            z_new = z_new.to(dt)
        F = z_new - y_flat
        # fp32 ALWAYS for the split: at bf16 the 8-bit mantissa turns the horizontality
        # leak into ~4e-3 instead of ~5e-9 and the momentum EMA carries it.
        with torch.autocast(device_type=y_flat.device.type, enabled=False):
            yc, Fc = self._c(self._hi(y_flat)), self._c(self._hi(F))
            n2 = yc.abs().square().sum(-1).clamp_min(self.den_eps)
            a = (yc.conj() * Fc).sum(-1).imag / n2              # EXACT ratio
            # projection_mode "none" (release): F_H = F.  `a` is still computed
            # because R consumes it as the rotation angle.
            F_H = self._r(Fc - 1j * a.unsqueeze(-1) * yc) if self.proj_force else self._r(Fc)
        F_H = F_H.to(dt)
        a32 = a.unsqueeze(-1)                                   # fp32 [B,S,1]
        self._last_a = a32.detach()

        beta = torch.sigmoid(self.b_H.float()).to(dt)       # fp32 sigmoid
        prop = beta * m_flat + (1.0 - beta) * F_H               # == mbar_{t+1}
        m_out = eta.to(dt) * prop                               # CARRY eta*mbar

        with torch.no_grad():
            pd, yd = prop.detach(), y_flat.detach()
            # r_prop: the UNDAMPED proposal -- what the controller sees.  eta-independent.
            r_prop = self._linf_c(pd) / (self._linf_c(yd + pd) + self.eps_r)
            # r_exec: executed displacement, normalised by the CURRENT state.  Using
            # ||y + eta*prop|| here would be non-monotone in eta and singular (on
            # F_H = -2y it is 2eta/|1-2eta|, which RISES at the first decay and diverges
            # at eta = 0.5).
            ed = (eta.to(dt) * prop).detach()
            r_exec = self._linf_c(ed) / (self._linf_c(yd) + self.eps_r)

        y_n = y_flat + eta.to(dt) * prop
        if not self.apply_R:
            return (y_n, m_out, r_prop, r_exec)

        # fp32 rotation ALWAYS + remainder: at bf16 cos^2+sin^2 = 1 +/- 2^-8, R stops being
        # orthogonal and ||y|| random-walks ~13% over 1000 steps.  a is DETACHED: dL/da == 0
        # in exact arithmetic, so any gradient here is pure roundoff noise.
        th = torch.remainder(a32.detach(), 2.0 * math.pi)
        c_, s_ = th.cos().to(dt), th.sin().to(dt)

        def _rot(u):
            p = u.view(*u.shape[:-1], u.shape[-1] // 2, 2)
            return torch.stack([c_ * p[..., 0] - s_ * p[..., 1],
                                s_ * p[..., 0] + c_ * p[..., 1]], -1).reshape_as(u)

        return (_rot(y_n), _rot(m_out), r_prop, r_exec)

    # --------------------------------------------------------------- step/finish
    @property
    def ckpt_keys(self):
        return ("y", "m", "q_eta")

    def step(self, state: dict, z_new: torch.Tensor):
        eta = state["q_eta"]
        args = (state["y"], state["m"], eta, z_new)
        if self.use_ckpt and self.training and torch.is_grad_enabled():
            outs = torch.utils.checkpoint.checkpoint(self._core, *args, use_reentrant=False)
        else:
            outs = self._core(*args)
        return self._finish(state, *outs)

    def _finish(self, state, y_new, m_out, r_prop, r_exec):
        """Mutates in place.  The patience controller acts on the gauge-invariant
        r_prop -- all torch.where, no Python branch on tensor data."""
        state["m"] = m_out
        state["y"] = y_new
        with torch.no_grad():
            r = r_prop.float()
            # eta_used is the eta that ACTUALLY multiplied this step's proposal; the
            # decay below decides the NEXT one.  Rescale by q_eta_used, not q_eta.
            state["q_eta_used"] = state["q_eta"].clone()
            state["q_residue"] = r
            state["q_exec"] = r_exec.float()
            improved = r < state["q_best"]
            state["q_best"] = torch.where(improved, r, state["q_best"])
            P = torch.full_like(state["q_patience"], float(self.decay_patience))
            state["q_patience"] = torch.where(improved, P, state["q_patience"] - 1.0)
            adapt = (state["q_patience"] <= 0) & (r >= self.r_thresh)
            state["q_patience"] = torch.where(adapt, P, state["q_patience"])
            decay = self.decay_train if self.training else self.decay_eval
            state["q_eta"] = state["q_eta"] * torch.where(
                adapt, torch.full_like(r, decay), torch.ones_like(r)
            ).to(state["q_eta"].dtype).reshape(-1, 1, 1)
            # legacy fields pinned inert: neither legacy halting predicate can fire
            state["residues"] = torch.full_like(state["residues"], torch.inf)
            state["stepsize"] = torch.full_like(state["stepsize"], 1.0)
        state["iter_idx"] = state["iter_idx"] + 1
        return state
