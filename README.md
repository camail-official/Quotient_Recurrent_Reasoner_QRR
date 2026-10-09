# QRR — Quotient Recurrent Reasoner

The official implementation of the paper "Looped Reasoning Finishes Earlier Than You Think".

Model, training and evaluation code for the four QRR tasks: Sudoku-Extreme, Maze-Hard,
Maze-Unique and Mini-ARC.

```
evaluate.py         evaluate a checkpoint on its task (all four protocols)
pretrain.py         training (torchrun, DDP)
config/             one training config per task (exact settings of the released runs)
models/             QRR: gauge-equivariant trunk, quotient integrator, readouts, loss head
dataset/            dataset builders
evaluators/         Mini-ARC / ARC voting evaluator (pass@K)
opt/                optional inference kernels for the Sudoku protocol (--opt)
create_model.py, pretrain_config.py, puzzle_dataset.py, utils/
```

## Setup

Python 3.12 and a CUDA build of PyTorch with triton. All runs use fp32 weights with tf32
matmuls; the scripts set `NVIDIA_TF32_OVERRIDE=1` themselves.

```
pip install -r requirements.txt
```

## Data

Run from the repository root.

```
# Sudoku-Extreme: 1k training puzzles x 1000 augmentations; evaluation on a fixed 10% of the test split (42,279 boards)
python -m dataset.build_sudoku_dataset --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python -m dataset.make_test_subset --src data/sudoku-extreme-1k-aug-1000 --dst data/sudoku-extreme-1k-aug-1000-test10

# Maze-Hard: 1k train / 1k test, no augmentation
python -m dataset.build_maze_dataset --output-dir data/maze-30x30-hard-1k-noaug

# Maze-Unique: 1k train / 1k test, unique shortest path
python -m dataset.build_maze_unique_dataset --output-dir data/maze-30x30-unique-1k \
    --grid-size 30 --train-samples 1000 --test-samples 1000 --maze-mode perfect \
    --length-distribution uniform --min-path-length 100 --max-path-length 140 --require-unique --dedupe

# Mini-ARC: clone https://github.com/KSB21ST/MINI-ARC into raw_data/MINI-ARC, then
python -m dataset.miniarc_to_kaggle --src raw_data/MINI-ARC/data/MiniARC --out raw_data/miniarc
python -m dataset.build_miniarc_dataset --num-aug 100
```

The Sudoku builder draws its 1k training subsample with numpy's global RNG (unseeded, as in
TRM), so a rebuilt training set differs from ours; the test set is deterministic.

## Evaluation

```
python evaluate.py --task sudoku      --ckpt checkpoints/sudoku/step_*
python evaluate.py --task maze_hard   --ckpt checkpoints/maze_hard/step_*
python evaluate.py --task maze_unique --ckpt checkpoints/maze_unique/step_*
python evaluate.py --task miniarc     --ckpt checkpoints/miniarc/step_*
```

Each run prints a summary and writes `eval_out/<task>/summary.json` (per-board results in
`eval_out/<task>/*.npz`). One GPU per run.

```
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python evaluate.py --task sudoku --ckpt ... --start $((i*10570)) --n 10570 &
done; wait
```

**`--opt` (Sudoku only).** This flag enables inference-only kernels: fused conv2d, fused
re_defect, cached attention extras, lazy decoding, and bf16 SDPA inputs.

## Training

```
torchrun --standalone --nproc_per_node=<GPUs> pretrain.py --config-name <task> +run_name=<name>
```

| task | config | GPUs used | lr | global batch | released / final step |
|---|---|---|---|---|---|
| Sudoku-Extreme | `sudoku` | 4 | 3e-4 | 768 | 52,080 / 78,120 |
| Maze-Hard | `maze_hard` | 4 | 1e-4 | 768 | 32,550 / 78,120 |
| Maze-Unique | `maze_unique` | 4 | 2e-4 | 384 | 15,624 / 15,624 |
| Mini-ARC | `miniarc` | 1 | 6e-4 (AdamAtan2) | 768 | 68,620 / 68,620 |
