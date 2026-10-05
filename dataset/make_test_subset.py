"""Fixed random subset of a puzzle dataset's TEST split (default 10%, seed 0).

Builds the Sudoku-Extreme evaluation set: 42,279 of the 422,786 test boards
(data/sudoku-extreme-1k-aug-1000-test10). The draw is sorted indices from
numpy default_rng(seed); `subset_indices.npy` records them.

    python -m dataset.make_test_subset --src data/sudoku-extreme-1k-aug-1000 \
        --dst data/sudoku-extreme-1k-aug-1000-test10
"""
import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sudoku-extreme-1k-aug-1000")
    ap.add_argument("--dst", default="data/sudoku-extreme-1k-aug-1000-test10")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src_t = os.path.join(a.src, "test")
    dst_t = os.path.join(a.dst, "test")
    meta = json.load(open(os.path.join(src_t, "dataset.json")))
    os.makedirs(dst_t, exist_ok=True)
    n_full = meta["total_groups"]
    assert meta["mean_puzzle_examples"] == 1.0, "1 example per puzzle assumed"

    gi = np.load(os.path.join(src_t, "all__group_indices.npy"))
    pi = np.load(os.path.join(src_t, "all__puzzle_indices.npy"))
    assert np.array_equal(gi, np.arange(len(gi))), "group_indices must be identity"
    assert np.array_equal(pi, np.arange(len(pi))), "puzzle_indices must be identity"

    n_sub = int(round(n_full * a.frac))
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n_full, size=n_sub, replace=False))

    for field in ("inputs", "labels", "puzzle_identifiers", "group_difficulties"):
        p = os.path.join(src_t, f"all__{field}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p, mmap_mode="r")
        np.save(os.path.join(dst_t, f"all__{field}.npy"), np.asarray(arr[idx]))

    # 1 example per puzzle, 1 puzzle per group -> both index arrays are identity
    np.save(os.path.join(dst_t, "all__group_indices.npy"),
            np.arange(n_sub + 1, dtype=gi.dtype))
    np.save(os.path.join(dst_t, "all__puzzle_indices.npy"),
            np.arange(n_sub + 1, dtype=pi.dtype))

    meta_out = dict(meta)
    meta_out["total_groups"] = n_sub
    meta_out["total_puzzles"] = n_sub
    json.dump(meta_out, open(os.path.join(dst_t, "dataset.json"), "w"))

    src_ids = os.path.join(a.src, "identifiers.json")
    if os.path.exists(src_ids):
        shutil.copy(src_ids, os.path.join(a.dst, "identifiers.json"))

    np.save(os.path.join(a.dst, "subset_indices.npy"), idx)  # provenance
    print(f"wrote {a.dst}: {n_sub}/{n_full} groups ({n_sub/n_full:.3%}), seed {a.seed}")


if __name__ == "__main__":
    main()
