import torch
import torch.nn as nn
import torch.distributed as dist

from models import rng_streams
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from pretrain_config import PretrainConfig
from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import load_model_class


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        # Parameters registered for deferred init draw from their own name-derived
        # streams after construction (models/rng_streams.py).
        _deferred = rng_streams.init_deferred(config.seed)
        if _deferred:
            print(f"[rng] deferred init for {len(_deferred)} params: {_deferred}")
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    dense_optimizer = _build_dense_optimizer(model, config)
    if config.arch.puzzle_emb_ndim == 0:  # type: ignore
        return model, [dense_optimizer], [config.lr]

    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0,  # set by the scheduler
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size,
        ),
        dense_optimizer,
    ]
    return model, optimizers, [config.puzzle_emb_lr, config.lr]


def _split_decay_param_groups(model: nn.Module, weight_decay: float):
    """Parameters tagged ``p._no_weight_decay = True`` (residual-scale alphas, the
    integrator's b_H) go to the no-decay group."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_dense_optimizer(model: nn.Module, config: PretrainConfig) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    param_groups = _split_decay_param_groups(model, config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=0,  # set by the scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    if name == "adam_atan2":
        # Mini-ARC arm. adam-atan2-pytorch asserts lr > 0 at construction; the
        # scheduler overwrites it before the first step.
        from adam_atan2_pytorch import AdamAtan2
        return AdamAtan2(
            param_groups,
            lr=1.0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected 'adamw' or 'adam_atan2'")
