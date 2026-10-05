"""Evaluate a released QRR checkpoint on its task.

    python evaluate.py --task sudoku      --ckpt checkpoints/sudoku/step_52080
    python evaluate.py --task maze_hard   --ckpt checkpoints/maze_hard/step_32550
    python evaluate.py --task maze_unique --ckpt checkpoints/maze_unique/step_15624
    python evaluate.py --task miniarc     --ckpt checkpoints/miniarc/step_68620

Protocols (defaults reproduce the reported numbers; Sudoku does not reseed its initial states, so it
reproduces them statistically, within ~0.1-0.3 pp, not board by board):

  task         test set                                  cap     halt rule          reported metric
  sudoku       sudoku-extreme-1k-aug-1000-test10 (42,279) 35,000 re_defect 0.03/W1  exact at halt
  maze_hard    maze-30x30-hard-1k-noaug (1,000)           3,000  re_defect 0.075/W1 exact at halt
  maze_unique  maze-30x30-unique-1k (1,000)               3,000  re_defect 0.15/W1  exact at halt
  miniarc      miniarc-aug-100 (149 tasks, 14,447 rows)   100    re_defect 0.08/W2  task pass@1 at halt

Evaluation runs one z-step per forward up to the cap. Halting is read out, not acted on:
re_defect is the joint (y, m) per-token phase-aligned defect between consecutive z-steps,
and a board halts at the first z-step that completes W consecutive values below theta. Its
answer is decoded at that step. Boards that never halt keep the cap answer. tau = z-steps
to halt.

Sudoku runs at cap 35,000 over 42,279 boards, which takes several GPU-hours. Shard it with
--start/--n over several processes sharing one --out directory. Finished chunks are saved
and skipped on rerun, and the summary covers every chunk in --out.
"""
import os
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")   # precision law: fp32 weights, tf32 matmul

import argparse
import glob
import json
import time

import numpy as np
import torch
import yaml

from models import rng_streams
from models.qrr_model import QRRModel

ROOT = os.path.dirname(os.path.abspath(__file__))

PROTOCOLS = {
    # seed: None = the protocol does not reseed (the default torch RNG state is used)
    "sudoku":      dict(data="data/sudoku-extreme-1k-aug-1000-test10/test", cap=35000,
                        theta=0.03, W=1, chunk=512, seed=None),
    "maze_hard":   dict(data="data/maze-30x30-hard-1k-noaug/test", cap=3000,
                        theta=0.075, W=1, chunk=250, seed=0),
    "maze_unique": dict(data="data/maze-30x30-unique-1k/test", cap=3000,
                        theta=0.15, W=1, chunk=250, seed=0),
    "miniarc":     dict(data="data/miniarc-aug-100", cap=100,
                        theta=0.08, W=2, chunk=256, seed=25),
}


def load_arch(task):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", f"{task}.yaml")))
    return {k: v for k, v in cfg["arch"].items() if k not in ("name", "loss")}, cfg


def build_model(arch, ckpt, batch_size, meta, cap):
    # An eval build never runs init_deferred (weights come from the checkpoint), so drop
    # registrations left by a previous build in this process.
    rng_streams.reset_deferred()
    cfg = dict(arch, batch_size=batch_size, seq_len=meta["seq_len"], vocab_size=meta["vocab_size"],
               num_puzzle_identifiers=meta["num_puzzle_identifiers"], max_iter_eval=cap)
    with torch.device("cuda"):
        net = QRRModel(cfg)
    sd = torch.load(ckpt, map_location="cuda", weights_only=True)
    sd = {k.removeprefix("_orig_mod.").removeprefix("model."): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    net.set_num_iters()
    return net


def decode(inner, y, emb):
    return inner.gauge_lm(y, emb)[:, inner.puzzle_emb_len:].argmax(-1)


def score(dec, lab):
    mk = lab != 0                                    # label 0 = padding
    exact = ((dec == lab) | ~mk).all(-1)
    cell = ((dec == lab) & mk).sum(-1) / mk.sum(-1)
    return exact.cpu().numpy(), cell.float().cpu().numpy()


# ------------------------------------------------------------------ board tasks (sudoku / maze_hard / maze_unique)
@torch.no_grad()
def rollout_halt(net, inp, lab, cap):
    """Rollout to `cap`; decode each board at its halt step (model's re_defect latch) and at cap."""
    inner = net.inner
    B = inp.shape[0]
    bt = {"inputs": inp, "labels": lab, "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device="cuda")}
    with torch.device("cuda"):
        carry = net.initial_carry(bt)
    carry.steps, carry.halted = carry.steps.cuda(), carry.halted.cuda()
    emb = inner._input_embeddings(inp.to(torch.int32), bt["puzzle_identifiers"])
    dec_halt = torch.zeros(B, lab.shape[1], dtype=torch.int16, device="cuda")
    seen = torch.zeros(B, dtype=torch.bool, device="cuda")
    for _ in range(cap):
        carry, _ = net(carry, bt)
        newly = carry.inner_carry.z_L_state["re_hit"] & ~seen
        if bool(newly.any()):
            dec_halt[newly] = decode(inner, carry.inner_carry.z_L_state["y"], emb)[newly].to(torch.int16)
            seen |= newly
        if bool(carry.halted.all()):
            break
    dec_cap = decode(inner, carry.inner_carry.z_L_state["y"], emb)
    fired = seen.cpu().numpy()
    ex_cap, cell_cap = score(dec_cap, lab)
    ex_halt, cell_halt = score(dec_halt.long(), lab)
    ex_halt[~fired], cell_halt[~fired] = ex_cap[~fired], cell_cap[~fired]   # never halted -> cap
    return dict(exact_halt=ex_halt, exact_cap=ex_cap, cell_halt=cell_halt, cell_cap=cell_cap,
                fired=fired, tau=carry.inner_carry.z_L_state["re_fire"].cpu().numpy())


def eval_boards(a, arch):
    meta = json.load(open(os.path.join(a.data, "dataset.json")))
    inp_all = np.load(os.path.join(a.data, "all__inputs.npy"), mmap_mode="r")
    lab_all = np.load(os.path.join(a.data, "all__labels.npy"), mmap_mode="r")
    assert 0 <= a.start < inp_all.shape[0]
    N = inp_all.shape[0] - a.start if a.n == 0 else min(a.n, inp_all.shape[0] - a.start)
    arch = dict(arch, halting_mechanism="re_defect", re_halt_thresh=a.theta, re_halt_window=a.W)
    os.makedirs(a.out, exist_ok=True)
    # Finished chunks are reused on rerun, so an --out directory is tied to one protocol.
    proto = dict(task=a.task, ckpt=os.path.abspath(a.ckpt), data=os.path.abspath(a.data), cap=a.cap,
                 theta=a.theta, W=a.W, seed=a.seed, chunk=a.chunk)
    pf = os.path.join(a.out, "protocol.json")
    if os.path.exists(pf):
        old = json.load(open(pf))
        assert old == proto, f"{a.out} holds results of a different protocol {old}; use a fresh --out"
    else:
        json.dump(proto, open(pf, "w"), indent=1)
    for lo in range(a.start, a.start + N, a.chunk):
        hi = min(lo + a.chunk, a.start + N)
        part = os.path.join(a.out, f"{a.task}_b{lo}-{hi}.npz")
        if os.path.exists(part):
            continue
        t0 = time.time()
        inp = torch.as_tensor(np.array(inp_all[lo:hi])).cuda().long()
        lab = torch.as_tensor(np.array(lab_all[lo:hi])).cuda().long()
        if a.task.startswith("maze"):
            assert bool((lab != 0).all()), "maze labels carry no padding; every cell is scored"
        if a.seed is not None:                       # before the build and the initial-state draw
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        net = build_model(arch, a.ckpt, hi - lo, meta, a.cap)
        r = rollout_halt(net, inp, lab, a.cap)
        del net; torch.cuda.empty_cache()
        tmp = os.path.join(a.out, f".tmp_{a.task}_b{lo}-{hi}.npz")   # not matched by the summary glob
        np.savez_compressed(tmp, boards=np.arange(lo, hi), **r)
        os.replace(tmp, part)
        print(f"  boards {lo}-{hi}: exact@cap {r['exact_cap'].mean():.4f}"
              f"  exact@halt {r['exact_halt'].mean():.4f}  fired {r['fired'].mean():.3f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    return summarize_boards(a)


def summarize_boards(a):
    parts = sorted(glob.glob(os.path.join(a.out, f"{a.task}_b*.npz")))
    Z = [np.load(p) for p in parts]
    R = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    assert len(np.unique(R["boards"])) == len(R["boards"]), "overlapping shards in --out"
    s = dict(task=a.task, ckpt=a.ckpt, cap=a.cap, n=int(len(R["boards"])),
             exact_cap=float(R["exact_cap"].mean()), cell_cap=float(R["cell_cap"].mean()))
    f, t = R["fired"], R["tau"]
    s.update(theta=a.theta, W=a.W, seed=a.seed,
             exact_halt=float(R["exact_halt"].mean()), cell_halt=float(R["cell_halt"].mean()),
             coverage=float(f.mean()), false_halt=float((f & ~R["exact_halt"]).mean()),
             tau_median=float(np.median(t[f])) if f.any() else None,
             tau_iqr=[float(x) for x in np.percentile(t[f], [25, 75])] if f.any() else None,
             mean_nfe=float(np.where(f, t, a.cap).mean()))
    return s


# ------------------------------------------------------------------ miniarc
def eval_miniarc(a, arch):
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from evaluators.miniarc import MiniARC
    _, cfg = load_arch(a.task)
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=cfg["seed"], dataset_paths=[a.data], global_batch_size=a.chunk,
                                           test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1),
                       split="test")
    meta = ds.metadata.model_dump()
    arch = dict(arch, halting_mechanism="re_defect", re_halt_thresh=a.theta, re_halt_window=a.W)
    net = build_model(arch, a.ckpt, a.chunk, meta, a.cap)
    if a.seed is not None:                           # after the build, before any initial-state draw
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    ev_halt, ev_cap = (MiniARC(data_path=a.data, eval_metadata=ds.metadata) for _ in range(2))
    taus, fired_all, real_all = [], [], []
    with torch.inference_mode():
        _miniarc_loop(a, net, ds, ev_halt, ev_cap, taus, fired_all, real_all)
    rh, rc = ev_halt.result(None, rank=0, world_size=1), ev_cap.result(None, rank=0, world_size=1)
    real = np.concatenate(real_all); f = np.concatenate(fired_all)[real]; t = np.concatenate(taus)[real]
    return dict(task=a.task, ckpt=a.ckpt, cap=a.cap, theta=a.theta, W=a.W, seed=a.seed, rows=int(real.sum()),
                pass1_halt=rh["eval/MiniARC/pass@1"], pass2_halt=rh["eval/MiniARC/pass@2"],
                pass1_cap=rc["eval/MiniARC/pass@1"], pass2_cap=rc["eval/MiniARC/pass@2"],
                coverage=float(f.mean()), tau_median=float(np.median(t[f])) if f.any() else None,
                tau_iqr=[float(x) for x in np.percentile(t[f], [25, 75])] if f.any() else None)


def _miniarc_loop(a, net, ds, ev_halt, ev_cap, taus, fired_all, real_all):
    for _, batch, _ in ds:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device("cuda"):
            carry = net.initial_carry(batch)
        B = batch["inputs"].shape[0]
        pred_halt = torch.zeros_like(batch["inputs"], dtype=torch.long)
        seen = torch.zeros(B, dtype=torch.bool, device="cuda")
        for _ in range(a.cap):
            carry, out = net(carry=carry, batch=batch)
            pred = out["logits"].argmax(-1)
            newly = carry.inner_carry.z_L_state["re_hit"] & ~seen
            pred_halt[newly] = pred[newly]
            seen |= newly
        pred_halt[~seen] = pred[~seen]               # never halted -> cap
        qh = out["q_halt_logits"]
        ev_halt.update_batch(batch, {"preds": pred_halt, "q_halt_logits": qh})
        ev_cap.update_batch(batch, {"preds": pred, "q_halt_logits": qh})
        real = (batch["puzzle_identifiers"] != ds.metadata.blank_identifier_id).cpu().numpy()
        taus.append(carry.inner_carry.z_L_state["re_fire"].cpu().numpy())
        fired_all.append(seen.cpu().numpy())
        real_all.append(real)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(PROTOCOLS))
    ap.add_argument("--ckpt", required=True, help="checkpoint file step_N (EMA weights)")
    ap.add_argument("--data", default=None,
                    help="test split directory; for miniarc the dataset root (default: see table)")
    ap.add_argument("--out", default=None, help="output directory (default: eval_out/<task>)")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="rollout seed (default: protocol's)")
    ap.add_argument("--chunk", type=int, default=None, help="boards per rollout batch")
    ap.add_argument("--start", type=int, default=0, help="first board (sharding; board tasks only)")
    ap.add_argument("--n", type=int, default=0, help="boards in this shard (0 = to the end; clamped at the end)")
    ap.add_argument("--opt", action="store_true",
                    help="sudoku only: inference kernel stack (opt/); ~1.4x faster, statistically "
                         "equivalent (bf16 SDPA inputs move single boards within the seed noise)")
    a = ap.parse_args()
    P = PROTOCOLS[a.task]
    for k in ("data", "cap", "theta", "W", "chunk"):
        if getattr(a, k) is None:
            setattr(a, k, P[k])
    if a.seed is None:
        a.seed = P["seed"]
    a.out = a.out or os.path.join("eval_out", a.task)
    if a.opt:
        assert a.task == "sudoku", "the opt stack is validated on the sudoku protocol only"
        import opt
        opt.install()
    arch, _ = load_arch(a.task)
    s = eval_miniarc(a, arch) if a.task == "miniarc" else eval_boards(a, arch)
    print(json.dumps(s, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
