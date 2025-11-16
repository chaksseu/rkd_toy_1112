#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniformly sample n points from an existing 2D dataset folder,
then save the subset (npy/csv) and visualizations (scatter/heatmap).

- Input:  src_dir with train.npy (and optionally val.npy, *labels.npy)
- Select: split = train | val | all (all = concat of available splits)
- Sample: uniform random, without replacement by default (use --with_replacement to allow replacement)
- Output: out_dir with sampled train.npy (+ labels if present), optional CSV,
          scatter/heatmap PNGs, and metadata.json with index mapping.
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def maybe_load(path: Path):
    return np.load(path) if path.exists() else None

def save_split(xy, labels, out_dir: Path, prefix: str, csv: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}.npy", xy.astype(np.float32))
    if labels is not None:
        np.save(out_dir / f"{prefix}_labels.npy", labels.astype(np.int32))
    if csv:
        (out_dir / f"{prefix}.csv").write_text("x,y\n" if labels is None else "x,y,label\n", encoding="utf-8")
        with open(out_dir / f"{prefix}.csv", "ab") as f:
            if labels is None:
                np.savetxt(f, xy, fmt="%.6f", delimiter=b",")
            else:
                np.savetxt(f, np.c_[xy, labels], fmt=[b"%.6f", b"%.6f", b"%d"], delimiter=b",")

def save_scatter(xy: np.ndarray, path: Path, title: str = "", s: int = 6, labels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    if labels is None:
        plt.scatter(xy[:, 0], xy[:, 1], s=s, edgecolors="none")
    else:
        c = (labels.astype(int) % 10)  # simple color mapping
        plt.scatter(xy[:, 0], xy[:, 1], s=s, c=c, edgecolors="none")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def save_heatmap(xy: np.ndarray, path: Path, bins: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    pad = 0.05 * float(max(dx, dy, 1e-6))
    H, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=bins,
        range=[[xmin - pad, xmax + pad], [ymin - pad, ymax + pad]]
    )
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(H.T, origin="lower", aspect="equal",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

NUM=32
def main():
    ap = argparse.ArgumentParser(description="Uniformly sample n points from an existing dataset folder")
    ap.add_argument("--src_dir", type=str, default="smile_data_n8192_scale10_rot0_trans_0_0",
                    help="input folder containing train.npy (and optional val.npy, *_labels.npy)")
    ap.add_argument("--out_dir", type=str, default=f"smile_data_n8192_scale10_rot0_trans_0_0_n{NUM}",
                    help="output folder to save the sampled subset")
    ap.add_argument("--n", type=int, default=NUM, help="number of points to sample")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                    help="which split to sample from (all = concat of available splits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_replacement", action="store_true", help="sample with replacement (default: without)")
    ap.add_argument("--csv", action="store_true", help="also save CSVs")
    ap.add_argument("--scatter_dot", type=int, default=6, help="scatter dot size")
    ap.add_argument("--heatmap_bins", type=int, default=128, help="heatmap bins")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load available splits
    train = maybe_load(src / "train.npy")
    val   = maybe_load(src / "val.npy")
    tr_y  = maybe_load(src / "train_labels.npy")
    va_y  = maybe_load(src / "val_labels.npy")

    if args.split == "train":
        if train is None:
            raise FileNotFoundError(f"train.npy not found in {src}")
        X = train
        Y = tr_y
        src_name = "train"
        src_lengths = [len(train)]
        src_offsets = [0]
    elif args.split == "val":
        if val is None:
            raise FileNotFoundError(f"val.npy not found in {src}")
        X = val
        Y = va_y
        src_name = "val"
        src_lengths = [len(val)]
        src_offsets = [0]
    else:  # all
        arrays = []
        labels = []
        offsets = []
        lengths = []
        offset_acc = 0

        if train is not None:
            arrays.append(train)
            lengths.append(len(train))
            offsets.append(offset_acc)
            offset_acc += len(train)
            labels.append(tr_y if tr_y is not None else None)

        if val is not None:
            arrays.append(val)
            lengths.append(len(val))
            offsets.append(offset_acc)
            offset_acc += len(val)
            labels.append(va_y if va_y is not None else None)

        if not arrays:
            raise FileNotFoundError(f"No train.npy or val.npy found in {src}")

        X = np.vstack(arrays)
        # If any label is None, drop labels entirely to avoid mismatch
        if all(lbl is not None for lbl in labels):
            Y = np.concatenate(labels)
        else:
            Y = None

        src_name = "all"
        src_lengths = lengths
        src_offsets = offsets

    N = len(X)
    n = args.n

    if not args.with_replacement and n > N:
        raise ValueError(f"Requested n={n} > available N={N} without replacement. Use --with_replacement or decrease n.")

    if args.with_replacement:
        idx = rng.integers(low=0, high=N, size=n, dtype=np.int64)
    else:
        idx = rng.choice(N, size=n, replace=False)

    Xs = X[idx]
    Ys = (Y[idx] if Y is not None else None)

    # Save subset
    prefix = "train"  # keep a simple, single 'train' file in the sampled folder
    save_split(Xs, Ys, out, prefix, csv=args.csv)

    # Save visualizations
    save_scatter(Xs, out / "fig_scatter.png",
                 title=f"Sampled {src_name}  n={n}", s=args.scatter_dot, labels=Ys)
    save_heatmap(Xs, out / "fig_heatmap.png", bins=args.heatmap_bins)

    # Build metadata with index mapping
    meta = {
        "source_dir": str(src.resolve()),
        "source_split": args.split,
        "num_in": int(N),
        "num_out": int(n),
        "with_replacement": bool(args.with_replacement),
        "seed": int(args.seed),
        "has_labels": bool(Y is not None),
        "index_mapping": {
            "selected_indices_in_concat_space": idx.tolist(),
            "concat_offsets": src_offsets,   # only meaningful when split=='all'
            "concat_lengths": src_lengths    # only meaningful when split=='all'
        },
        "figures": {
            "scatter": str((out / "fig_scatter.png").resolve()),
            "heatmap": str((out / "fig_heatmap.png").resolve())
        }
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[OK] Saved sampled subset to:", out.resolve())
    print(f" - {prefix}.npy", "(+ labels)" if Y is not None else "")
    if args.csv:
        print(f" - {prefix}.csv")
    print(" - fig_scatter.png / fig_heatmap.png")
    print(" - metadata.json")
    print(f"Source: {src_name} | N={N}  ->  Sampled n={n}")

if __name__ == "__main__":
    main()
