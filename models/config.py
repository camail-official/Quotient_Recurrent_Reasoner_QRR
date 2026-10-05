"""QRR release configuration.

One flat config for the released architecture: the gauge-equivariant trunk
(pure-gauge dressed-SDPA transport), the order-2 quotient integrator ('qso',
projection_mode 'none'), the rich_rms invariant readouts and the re_defect
halting rule.  Field names are FROZEN to the trained checkpoints' yaml keys.

NOTE pydantic silently ignores undeclared keys, so every field the release
code reads MUST be declared here -- an undeclared live field would silently
fall back to its getattr default.
"""
from typing import Optional, Literal
from pydantic import BaseModel


class QRRConfig(BaseModel):
    # ---- data / shapes (filled from the dataset at build time) ----
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # ---- transformer trunk ----
    hidden_size: int
    num_heads: int
    L_layers: int
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    norm_placement: str = "none"          # 'output' | 'none'
    conv_type: str = "none"               # 'conv1d' | 'conv2d' | 'none'
    conv_kernel_size: int = 4
    alpha_1_init: float = 0.5             # pair-tied residual scaling
    alpha_2_init: float = 0.5
    forward_dtype: str = "float32"
    variational_dropout: float = 0.0

    # ---- puzzle-embedding prefix ----
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    # symmetry-breaking init for the sparse puzzle table: the dressed injection
    # D(x,h)*x is quadratic in x at the origin, so a zero init is an EXACT
    # stationary point (zero gradient forever). Mini-ARC sets 0.01.
    puzzle_emb_init_std: float = 0.0

    # ---- gauge trunk (release law) ----
    gauge_trunk: bool = True
    gauge_covstart: bool = False          # covariant power-iteration start (ON: sudoku, miniarc)
    gauge_conn_eps: float = 1e-6
    gauge_dress_eps: float = 1e-3
    gauge_attn_inner_c: int = 512         # complex inner width (= 2 * hidden_c)
    gauge_inter_c: int = 1536             # GaugeSwiGLU complex width
    gauge_xroute_dim: int = 16            # input-routed attention (d_x per head)
    gauge_gate_anchors: int = 4           # signed relative-phase SwiGLU gate (anchors)
    gauge_gate_anchor_eps: float = 1e-3
    gauge_readout: str = "rich_rms"
    gauge_readout_k: int = 8              # k_ro of the lm readout
    gauge_readout_hidden: int = 256
    gauge_inv_r: int = 8

    # ---- quotient integrator (order-2 'qso') ----
    integrator: str = "qso"
    projection_mode: Literal["none"] = "none"         # release law: unprojected force + raw momentum
    init_std: float = 1.0                 # trunc-normal scale of the fresh state draw
    osc_n: int = 4                        # n-cell size of the init retraction only
    osc_ckpt: bool = False                # checkpoint the integrator step fn
    q_den_eps: float = 1e-12
    q_eps_r: float = 1e-8
    q_eta0: float = 1.0
    q_decay_train: float = 1.0            # controller OFF at train (eta bitwise 1.0)
    q_decay_eval: float = 1.0
    q_patience: int = 10
    q_rthresh: float = 0.1
    q_apply_R: bool = True
    q_beta_H_init: float = 0.5            # b_H = logit(0.5) = 0, learned thereafter
    c1_reduced_state: bool = False        # momentum as second charge+1 trunk input
    c1_c_m: float = 4.3455                # m_H scale; cancels exactly (see mh_for_trunk)

    # ---- rollout / halting ----
    max_iter: int = 12                    # max deep-supervision segments per training episode
    max_iter_eval: Optional[int] = None
    max_iter_dist: str = "det"
    n_backwards_L: int = 1                # with-grad z-steps per segment
    zstep_ckpt: bool = False              # checkpoint trunk+core per z-step
    zstep_ckpt_keep: int = 0
    halting_mechanism: Literal["fixed_point", "fixed_iterations",
                               "re_defect"] = "fixed_point"
    # re_defect: joint (y,m) per-token phase-aligned relative-equilibrium defect
    # < re_halt_thresh, latched on the re_halt_window-th consecutive crossing.
    # Train-only per board; at eval it is RECORDED (re_fire), never acted on.
    re_halt_thresh: float = 0.05
    re_halt_window: int = 1
    fp_thresh: float = 0.1                # legacy predicate; pinned inert under qso
