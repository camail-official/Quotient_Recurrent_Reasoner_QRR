from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn

from models.layers import RotaryEmbedding, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.config import QRRConfig
from models.integrator import QuotientIntegrator, _GeoIntegrator

IGNORE_LABEL_ID = -100


@dataclass
class QRRInnerCarry:
    z_L_state: dict
    dropout_mask: torch.Tensor

@dataclass
class QRRCarry:
    inner_carry: QRRInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class QRRModelInner(nn.Module):
    def __init__(self, config: QRRConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        # lm_head / q_head are NOT used in forward (decoding goes through gauge_lm / gauge_q).
        # Kept only because they are in the trained checkpoints' state dict (strict load) and
        # their construction draws from the global init RNG stream.
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            # See config.puzzle_emb_init_std: a zero init is an exact stationary point of
            # the dressed injection, so a nonzero symmetry-breaking init may be needed.
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size,
                                                    init_std=getattr(self.config, "puzzle_emb_init_std", 0.0),
                                                    cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        else:
            pass

        # Reasoning trunk: the gauge-equivariant transformer.
        assert bool(getattr(self.config, "gauge_trunk", False)), \
            "the release ships the gauge trunk only"
        from models.gauge_transformer import GaugeFixedPointTransformer
        self.L_level = GaugeFixedPointTransformer(self.config, self.config.L_layers)

        # Invariant readouts (rich features + feature RMSNorm -- 'rich_rms').
        from models.gauge_layers import GaugeReadout
        hc = self.config.hidden_size // 2
        assert str(getattr(self.config, "gauge_readout", "rich_rms")) == "rich_rms", \
            "the release ships the rich_rms readout only"
        self.gauge_lm = GaugeReadout(hc, self.config.vocab_size,
                                     k_ro=int(getattr(self.config, "gauge_readout_k", 8)),
                                     r=int(getattr(self.config, "gauge_inv_r", 8)),
                                     feat_norm=True, norm_eps=self.config.rms_norm_eps,
                                     hidden=int(getattr(self.config, "gauge_readout_hidden", 256)),
                                     tag="lm")
        self.gauge_q = GaugeReadout(hc, 2, k_ro=4,
                                    r=int(getattr(self.config, "gauge_inv_r", 8)),
                                    final_bias=-5.0, feat_norm=True,
                                    norm_eps=self.config.rms_norm_eps, tag="q")

        # State-evolution law: the order-2 quotient integrator.
        assert getattr(self.config, "integrator", "") == "qso", \
            "the release ships integrator 'qso' only"
        self.L_optimizer = QuotientIntegrator(self.config)

        # Unused q_head (see above): original zero-weight / -5 bias init kept.
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return QRRInnerCarry(
            z_L_state = None,
            dropout_mask = None,
        )

    def reset_carry(self, 
                    reset_flag: torch.Tensor, 
                    batch: torch.Tensor,
                    carry: QRRInnerCarry):        
        shape = (batch.shape[0], batch.shape[1] + self.puzzle_emb_len, self.config.hidden_size)
        device = batch.device
        dtype = self.forward_dtype
        
        if self.training:
            dropout_mask = torch.empty(*shape, device=device, dtype=dtype).bernoulli_(p=1 - self.config.variational_dropout).div_(1 - self.config.variational_dropout)
            if carry.dropout_mask is not None:
                dropout_mask = torch.where(reset_flag.view(-1, 1, 1), dropout_mask, carry.dropout_mask)
        else:
            dropout_mask = torch.ones(*shape, device=device, dtype=dtype)

        z_L_state = self.L_optimizer.reset(reset_flag, shape, dtype, device, carry.z_L_state)
        # re_defect halting state. reset() builds a NEW dict, so these keys are re-attached
        # here; slots being reset start fresh (new puzzle, no crossing yet).
        if self.config.halting_mechanism == "re_defect":
            B = batch.shape[0]
            prev = carry.z_L_state if carry.z_L_state is not None else {}
            z_f = torch.zeros(B, dtype=torch.bool, device=device)
            z_zero = torch.zeros(B, dtype=torch.long, device=device)
            z_neg = torch.full((B,), -1, dtype=torch.long, device=device)
            z_inf = torch.full((B,), float("inf"), device=device)
            z_L_state["re_hit"] = torch.where(reset_flag, z_f, prev.get("re_hit", z_f))
            z_L_state["re_t"] = torch.where(reset_flag, z_zero, prev.get("re_t", z_zero))
            # consecutive sub-threshold z-step counter (re_halt_window latch)
            z_L_state["re_run"] = torch.where(reset_flag, z_zero, prev.get("re_run", z_zero))
            z_L_state["re_fire"] = torch.where(reset_flag, z_neg, prev.get("re_fire", z_neg))
            z_L_state["re_last"] = torch.where(reset_flag, z_inf, prev.get("re_last", z_inf))
        return QRRInnerCarry(
            z_L_state = z_L_state,
            dropout_mask=dropout_mask,
        )
    
    def _c1_on(self) -> bool:
        """Whether the momentum m_H is fed to the trunk as a second input (c1_reduced_state).
        Gauge trunk only: m_H has charge +1, so only the gauge trunk consumes it equivariantly."""
        if not bool(getattr(self.config, "c1_reduced_state", False)):
            return False
        assert getattr(self.config, "gauge_trunk", False), \
            "c1_reduced_state requires gauge_trunk (m_H is charge +1)"
        assert isinstance(self.L_optimizer, _GeoIntegrator)
        return True

    def _c1_mh(self, m, y):
        """The m_H trunk input, in ONE place so _z_step and the checkpointed _pure_step_q
        cannot drift (a mirror that computes this differently gives an identical forward
        and a silently zero gradient).

        The trunk sees RAW m_H: QuotientIntegrator.mh_for_trunk pre-multiplies by c1_c_m
        to cancel the trunk's own division by c1_c_m -- net factor exactly 1."""
        # projection_mode 'none': m_H = m (no horizontal projection).
        scale = getattr(self.L_optimizer, "mh_for_trunk", None)
        return scale(m, y) if scale is not None else m

    def _z_step(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        # c1_reduced_state: momentum as a second charge+1 trunk input.  RE-DERIVED here
        # from (y, m) every step, never carried -- a carried split would silently define a
        # different transport law.  NOT detached: df/dm_H is part of the recurrent
        # Jacobian being learned.
        _c1 = {}
        if self._c1_on():
            _c1 = dict(m_h=self._c1_mh(state["m"], state["y"]))
        # variational_dropout == 0 => the mask is exactly all-ones: skip the dead
        # full-state multiply. The mask is still built/carried (unchanged carry layout).
        _zl = self.L_level(state["y"], input_embeddings, **_c1, **seq_info)
        z_new = _zl if self.config.variational_dropout == 0.0 else dropout_mask * _zl
        return self.L_optimizer.step(state, z_new)

    def decode_logprobs(self, y, input_embeddings, inputs):
        """THE decode entry point: state -> per-cell LOGITS over the vocabulary [B, S_grid, V].
        Prefix slicing happens HERE, once, so the training loss, the accuracy metric and
        the official evaluator all read one object."""
        return self.gauge_lm(y, input_embeddings)[:, self.puzzle_emb_len:]

    def forward(self,
                carry: QRRInnerCarry,
                batch: Dict[str, torch.Tensor],
                force_grad: bool,
                n_steps: int) -> Tuple[
        Tuple[QRRInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Real-RoPE cache sliced to the input length. The gauge trunk ignores cos_sin
        # (its attention applies PairRope); it is passed for interface compatibility.
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=self.puzzle_emb_len)
        # Per-segment precompute for the gauge trunk (dress GEMM on the segment-constant
        # embedding + residual-scaling algebra): shared graph node across the z-steps.
        if hasattr(self.L_level, "make_seg_ctx"):
            seq_info["seg_ctx"] = self.L_level.make_seg_ctx(input_embeddings)
        z_state, dropout_mask = carry.z_L_state, carry.dropout_mask
        use_ckpt = bool(getattr(self.config, "zstep_ckpt", False)) and force_grad
        if use_ckpt:
            # zstep_ckpt checkpoints trunk + integrator-core per z-step. The checkpointed
            # fn must be PURE (checkpoint re-runs it in backward with the saved inputs;
            # _z_step mutates the state dict in place): pure (y, m, eta) -> _core outputs,
            # dict mutation via _finish OUTSIDE the checkpoint.
            assert isinstance(self.L_optimizer, QuotientIntegrator)

            def _pure_step_q(y, m, eta):
                # m_h is derived INSIDE from the CHECKPOINTED (y, m) via the same _c1_mh
                # helper, and the call routes through _core so b_H sits INSIDE the
                # checkpoint -- a mirror that rebuilds the trunk but bypasses _core gives
                # an identical forward and a silently ZERO dL/db_H.
                _c1q = {}
                if self._c1_on():
                    _c1q = dict(m_h=self._c1_mh(m, y))
                z_new = dropout_mask * self.L_level(y, input_embeddings, **_c1q, **seq_info)
                return self.L_optimizer._core(y, m, eta, z_new)

        # PARTIAL CHECKPOINTING (zstep_ckpt_keep, default 0 = every step checkpointed).
        # keep=k leaves the LAST k z-steps of the segment uncheckpointed, so their
        # activations are stored instead of recomputed: memory grows linearly in k, backward
        # recompute falls linearly in k.  The trailing steps are the ones kept because they
        # are the first the backward pass needs, so the peak overlaps least with the
        # recompute of the earlier ones.
        # k >= n_steps is equivalent to zstep_ckpt=False.  Checkpointing is mathematically
        # exact, so every k gives identical gradients.
        _keep = int(getattr(self.config, "zstep_ckpt_keep", 0)) if use_ckpt else 0
        _first_kept = n_steps - _keep
        with torch.set_grad_enabled(force_grad):
            for _i in range(n_steps):
                if self.config.halting_mechanism == "re_defect":
                    # pre-step state for the defect; clone because _finish mutates in place
                    with torch.no_grad():
                        _re_py = z_state["y"].detach().clone()
                        _re_pm = z_state["m"].detach().clone() if "m" in z_state else None
                if use_ckpt and _i >= _first_kept:
                    # kept: run the ordinary path, store activations
                    z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info)
                elif use_ckpt:
                    outs = torch.utils.checkpoint.checkpoint(
                        _pure_step_q, z_state["y"], z_state["m"], z_state["q_eta"],
                        use_reentrant=False)
                    z_state = self.L_optimizer._finish(z_state, *outs)
                else:
                    z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info)
                # re_defect halting: relative defect between consecutive z-steps of the joint
                # (y, m) state after per-token phase alignment,
                #   ph_s = conj(<(y0,m0)_s, (y1,m1)_s>) / |.|,
                #   dre  = ||ph * (y1,m1) - (y0,m0)|| / ||(y0,m0)||   (per board),
                # evaluated at EVERY z-step and LATCHED.  Same quantity at train and eval;
                # training halts on it at the segment boundary, eval only records re_fire
                # (see QRRModel.forward).
                if self.config.halting_mechanism == "re_defect":
                    with torch.no_grad():
                        _y1 = z_state["y"].detach()
                        _m1 = z_state["m"].detach() if "m" in z_state else None
                        _Hh = _y1.shape[-1]
                        def _cplx(v):
                            # .float(): no-op in fp32; view_as_complex rejects bfloat16.
                            # Metric-only (no_grad).
                            return torch.view_as_complex(
                                v.float().reshape(*v.shape[:-1], _Hh // 2, 2).contiguous())
                        _ip = (_cplx(_re_py).conj() * _cplx(_y1)).sum(-1, keepdim=True)
                        if _m1 is not None:
                            _ip = _ip + (_cplx(_re_pm).conj() * _cplx(_m1)).sum(-1, keepdim=True)
                        _ph = _ip.conj() / (_ip.abs() + 1e-30)
                        _by = torch.view_as_real(_cplx(_y1) * _ph).flatten(-2) - _re_py
                        _n2 = _by.flatten(1).norm(dim=1).square()
                        _q2 = _re_py.flatten(1).norm(dim=1).square()
                        if _m1 is not None:
                            _bm = torch.view_as_real(_cplx(_m1) * _ph).flatten(-2) - _re_pm
                            _n2 = _n2 + _bm.flatten(1).norm(dim=1).square()
                            _q2 = _q2 + _re_pm.flatten(1).norm(dim=1).square()
                        _dre = torch.sqrt(_n2) / torch.sqrt(_q2).clamp_min(1e-30)
                        z_state["re_last"] = _dre
                        _rwas = z_state["re_hit"]
                        # re_halt_window latch: fires ON the W-th CONSECUTIVE sub-threshold
                        # z-step.  W=1 is a single-crossing latch.
                        _below = _dre < float(self.config.re_halt_thresh)
                        _W = int(getattr(self.config, "re_halt_window", 1))
                        z_state["re_run"] = torch.where(
                            _below, z_state["re_run"] + 1, torch.zeros_like(z_state["re_run"]))
                        z_state["re_hit"] = _rwas | (z_state["re_run"] >= _W)
                        z_state["re_t"] = z_state["re_t"] + 1
                        z_state["re_fire"] = torch.where(
                            z_state["re_hit"] & ~_rwas, z_state["re_t"], z_state["re_fire"])

        output = self.decode_logprobs(z_state["y"], input_embeddings, batch["inputs"])
        q_logits = self.gauge_q(z_state["y"][:, 0], input_embeddings[:, 0]).to(torch.float32)

        new_carry = QRRInnerCarry(
            z_L_state=self.L_optimizer.detach_state(z_state),
            dropout_mask=carry.dropout_mask)          # New carry no grad
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class QRRModel(nn.Module):
    """Outer wrapper: per-board carry, step counting and halting around QRRModelInner."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = QRRConfig(**config_dict)
        self.inner = QRRModelInner(self.config)
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return QRRCarry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def set_num_iters(self):
        if self.training:
            assert self.config.max_iter_dist == 'det', "release: max_iter_dist 'det' only"
            self.max_iter = max(0, int(self.config.max_iter))
        else:
            self.max_iter = self.config.max_iter_eval if self.config.max_iter_eval is not None else self.config.max_iter

    def forward(
        self,
        carry: QRRCarry,
        batch: Dict[str, torch.Tensor],
    ):
        # Update data, carry (removing halted sequences)
        # (the state of halted slots is re-drawn by the integrator's reset())
        new_inner_carry = self.inner.reset_carry(carry.halted, batch['inputs'], carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward-backward inner model
        n_steps = self.config.n_backwards_L if self.training else 1
        carry_in = new_inner_carry
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            carry_in, new_current_data, force_grad=self.training, n_steps=n_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        if self.config.halting_mechanism == "re_defect":
            outputs["re_hit"] = new_inner_carry.z_L_state["re_hit"]
            outputs["re_defect"] = new_inner_carry.z_L_state["re_last"]
            outputs["re_fire"] = new_inner_carry.z_L_state["re_fire"]

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.max_iter
            halted = is_last_step

            # At eval, halting is batch-wide only (keeps ranks in lockstep).  These legacy
            # predicates never fire: the quotient integrator pins residues=inf / stepsize=1,
            # so eval runs to max_iter and re_fire is only RECORDED, never acted on.
            if not self.training:
                halted = halted | (new_inner_carry.z_L_state['residues'].max() < self.config.fp_thresh) \
                                | (new_inner_carry.z_L_state['stepsize'].max() < 1e-3)

            if self.training and (self.max_iter > 1):
                if self.config.halting_mechanism == 'fixed_point':
                    # legacy predicate, inert here (residues pinned +inf): runs to max_iter
                    halted = halted | (new_inner_carry.z_L_state['residues'] < self.config.fp_thresh) \
                                    | (new_inner_carry.z_L_state['stepsize'].view(-1) < 1e-3)
                elif self.config.halting_mechanism == 're_defect':
                    # per-board training halt on the latched re_defect crossing (training only;
                    # eval runs to max_iter, see above)
                    halted = halted | new_inner_carry.z_L_state['re_hit']
                elif self.config.halting_mechanism == 'fixed_iterations':
                    pass
                else:
                    raise ValueError(
                        "release halting: fixed_point | re_defect | fixed_iterations")

        return QRRCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
