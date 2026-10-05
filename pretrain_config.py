from typing import Optional, List

import pydantic


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    # every key besides name/loss is passed to models.config.QRRConfig
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig

    # data
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []

    # optimization
    global_batch_size: int
    epochs: int
    optimizer: str = "adamw"              # 'adamw' | 'adam_atan2'
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip_norm: Optional[float] = None   # global-norm clip + skip of non-finite steps
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    ema: bool = False
    ema_rate: float = 0.999

    # run bookkeeping
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    resume_from: Optional[str] = None     # resume bundle (file or its directory)
    seed: int = 0
    eval_interval: Optional[int] = None   # epochs per train/eval iteration
    min_eval_interval: Optional[int] = 0  # skip evaluation for the first N iterations
    checkpoint_every_eval: bool = False
    checkpoint_every_n_steps: Optional[int] = None
