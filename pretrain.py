"""QRR training.

    torchrun --standalone --nproc_per_node=4 pretrain.py --config-name sudoku

Configs live in config/ (one per task, the exact settings of the released
checkpoints). Hydra overrides work as usual, e.g. `seed=1 +run_name=my_run`.
In-run evaluation reads out at the evaluation cap with no halting; the reported
numbers come from evaluate.py.
"""
import os
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")   # precision law: fp32 weights, tf32 matmul

from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import copy
import json
import math
import shutil

import yaml
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from utils.resume import restore_train_state, save_resume_bundle
from models.ema import EMAHelper
from pretrain_config import PretrainConfig
from create_model import create_model


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def autocast_ctx(config: PretrainConfig):
    dtype = getattr(torch, (config.arch.__pydantic_extra__ or {}).get("forward_dtype", "float32"))
    return torch.amp.autocast(device_type="cuda", dtype=dtype, cache_enabled=False)


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)
    return TrainState(step=0, total_steps=total_steps, model=model, optimizers=optimizers,
                      optimizer_lrs=optimizer_lrs, carry=None)


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            evaluators.append(load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__))  # type: ignore
    return evaluators


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    with autocast_ctx(config):
        train_state.carry, masked_loss, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
        scaled_loss = (1 / global_batch_size) * masked_loss
    scaled_loss.backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Global-norm clip; the whole optimizer step is skipped when the loss or the
    # gradient norm is non-finite. (Sparse puzzle-embedding grads bypass the clip;
    # SignSGD is magnitude-free.)
    if config.grad_clip_norm is not None:
        gn = torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), config.grad_clip_norm)
        if not (torch.isfinite(scaled_loss.detach()) and torch.isfinite(gn)):
            for optim in train_state.optimizers:
                optim.zero_grad(set_to_none=True)
            if rank == 0:
                print(f"[clip] non-finite at step {train_state.step}; step skipped", flush=True)
            return

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))  # same order on every rank
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader,
             eval_metadata: PuzzleDatasetMetadata, evaluators: List[Any], rank: int, world_size: int,
             cpu_group: Optional[dist.ProcessGroup]):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set()
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        eval_model = train_state.model
        eval_model.set_num_iters()  # eval budget = max_iter_eval

        metric_keys = []
        metric_values = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = eval_model.initial_carry(batch)  # type: ignore

            while True:
                with autocast_ctx(config):
                    carry, _, loss, metrics, preds, all_finish = eval_model(
                        carry=carry, batch=batch, return_keys=return_keys)
                if all_finish:
                    break

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            del carry, loss, preds, batch, all_finish

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {metric_name: reduced_metrics[set_id, metric_id]
                               for metric_id, metric_name in enumerate(metric_keys)}
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        for evaluator in evaluators:
            save_path = None
            if config.checkpoint_path is not None:
                save_path = os.path.join(config.checkpoint_path,
                                         f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}")
                os.makedirs(save_path, exist_ok=True)
            metrics = evaluator.result(save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                reduced_metrics = reduced_metrics or {}
                reduced_metrics.update(metrics)

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    for code_file in (get_model_source_path(config.arch.name), get_model_source_path(config.arch.loss.name)):
        if code_file is not None:
            shutil.copy(code_file, os.path.join(config.checkpoint_path, os.path.basename(code_file)))
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-QRR"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        objects = [config]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


def append_metrics_jsonl(config: PretrainConfig, row: dict):
    """Durable eval record beside the checkpoints, independent of wandb."""
    print("EVAL_METRICS " + json.dumps(row), flush=True)
    if config.checkpoint_path:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        with open(os.path.join(config.checkpoint_path, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")


@hydra.main(config_path="config", config_name="sudoku", version_base=None)
def launch(hydra_config: DictConfig):
    RANK, WORLD_SIZE, CPU_PROCESS_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=int(os.environ.get("NCCL_TIMEOUT_MIN", "60"))))
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    # Each train iteration covers `eval_interval` epochs; the data order is keyed on it.
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    evaluators = create_evaluators(config, eval_metadata)

    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    steps_per_epoch = train_state.total_steps / config.epochs   # used for the logged epoch only

    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    start_iter_id, prev_wandb_id = 0, None
    if config.resume_from is not None:
        start_iter_id, prev_wandb_id = restore_train_state(
            resume_from=config.resume_from, train_state=train_state,
            ema_helper=ema_helper, dataset=train_loader.dataset)

    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        wandb.init(project=config.project_name, name=config.run_name,
                   id=prev_wandb_id, resume="allow" if prev_wandb_id else None,
                   config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        if config.resume_from is None:
            wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    for _iter_id in range(start_iter_id, total_iters):
        ############ Train
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            train_state.model.set_num_iters()  # type: ignore[attr-defined]
            for m in train_state.model.modules():
                if hasattr(m, "reset_mask"):
                    m.reset_mask()
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                metrics["epoch"] = train_state.step / steps_per_epoch
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)
            if (RANK == 0 and config.checkpoint_every_n_steps is not None and train_state.step > 0
                    and train_state.step % config.checkpoint_every_n_steps == 0):
                save_train_state(config, train_state)

        if _iter_id < config.min_eval_interval:
            continue

        ############ Evaluate (EMA weights when enabled)
        if config.ema:
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            train_state_eval = train_state
        train_state_eval.model.eval()
        metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators,
                           rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)

        if RANK == 0 and metrics is not None:
            set_keys = [k for k, v in metrics.items() if isinstance(v, dict)]
            log = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    prefix = "eval/" if len(set_keys) == 1 else f"eval/{k}/"
                    log.update({f"{prefix}{mk}": mv for mk, mv in v.items()})
                else:
                    log[k] = v
            log["epoch"] = train_state.step / steps_per_epoch
            wandb.log(log, step=train_state.step)
            append_metrics_jsonl(config, {**{k: float(v) for k, v in log.items()}, "step": int(train_state.step)})

        ############ Checkpoint (at eval, step_N holds the EMA weights and the resume bundle the raw ones;
        # the extra checkpoint_every_n_steps saves above write the raw weights)
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state_eval)
            save_resume_bundle(
                checkpoint_path=config.checkpoint_path,
                train_state=train_state,
                ema_helper=ema_helper,
                dataset_iters=(_iter_id + 1) * len(train_metadata.sets),
                iter_id=_iter_id,
                wandb_run_id=wandb.run.id if wandb.run is not None else None,
            )
        if config.ema:
            del train_state_eval

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
