#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample N points from a 2D dataset folder and apply a 3x3 homography
to obtain paired (source, transformed) coordinates.

- Input:
    src_dir with train.npy (and optionally val.npy, *_labels.npy)
- Steps:
    1) Select a split: train | val | all
    2) Uniformly sample n points (with/without replacement)
    3) Build homography H (direct 9 params or parametric transform)
    4) Apply H only to the sampled points
- Output (out_dir):
    - src.npy:          [n, 2] original points
    - tgt.npy:          [n, 2] transformed points
    - pairs.npy:        [n, 4] array (x_src, y_src, x_tgt, y_tgt)
    - pairs_labels.npy: [n] labels for each pair (if labels exist)
    - metadata.json:    sampling indices + H info
    - scatter/heatmap PNGs for src / tgt / both
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def maybe_load(path: Path):
    return np.load(path) if path.exists() else None


def save_scatter(xy: np.ndarray, path: Path, title: str = "", s: int = 6, labels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    if labels is None:
        plt.scatter(xy[:, 0], xy[:, 1], s=s, edgecolors="none")
    else:
        c = (labels.astype(int) % 10)
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
        range=[[xmin - pad, xmax + pad], [ymin - pad, ymax + pad]],
    )
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="equal",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_homography_from_params(
    scale: float = 1.0,
    rotate_deg: float = 0.0,
    translate=(0.0, 0.0),
    anisotropic_scale=None,
    perspective=(0.0, 0.0),
) -> np.ndarray:
    """
    Build H = P @ T @ R @ S_aniso @ S_iso  (column-vector convention)
    - P (perspective): [[1,0,0],[0,1,0],[p,q,1]]
    - T (translate):   [[1,0,tx],[0,1,ty],[0,0,1]]
    - R (rotate ccw):  [[c,-s,0],[s,c,0],[0,0,1]]
    - S_aniso:         [[sx,0,0],[0,sy,0],[0,0,1]]
    - S_iso:           [[s,0,0],[0,s,0],[0,0,1]]
    """
    s = float(scale)
    tx, ty = translate
    p, q = perspective

    # isotropic scale
    S_iso = np.array(
        [
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # anisotropic scale (optional)
    if anisotropic_scale is not None:
        sx, sy = anisotropic_scale
        S_aniso = np.array(
            [
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
    else:
        S_aniso = np.eye(3, dtype=np.float64)

    # rotation
    th = math.radians(rotate_deg)
    c, s_ = math.cos(th), math.sin(th)
    R = np.array(
        [
            [c, -s_, 0],
            [s_, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # translation
    T = np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # perspective skew
    P = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [p, q, 1],
        ],
        dtype=np.float64,
    )

    H = P @ T @ R @ S_aniso @ S_iso
    return H


def parse_H_from_flat_list(flat9):
    a, b, c, d, e, f, g, h, i = [float(x) for x in flat9]
    H = np.array([[a, b, c], [d, e, f], [g, h, i]], dtype=np.float64)
    return H


def apply_homography(
    xy: np.ndarray,
    H: np.ndarray,
    w_eps: float = 1e-9,
    drop_invalid: bool = False,
):
    """
    xy: [N,2], H: [3,3]
    Returns (out_xy, n_valid_w)
      - out_xy: [N,2] if drop_invalid=False
      - out_xy: [M,2] (M<=N) if drop_invalid=True
    """
    N = xy.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    homo = np.hstack([xy.astype(np.float64), ones])  # [N,3]
    out = (H @ homo.T).T  # [N,3]
    w = out[:, 2:3]
    mask = np.abs(w) > w_eps
    if drop_invalid:
        valid = mask[:, 0]
        out = out[valid]
        w = w[valid]
    out_xy = out[:, :2] / w
    return out_xy.astype(np.float32), int(mask.sum())


def main():
    ap = argparse.ArgumentParser(
        description="Sample N points from a 2D dataset and apply a homography to get (src, tgt) pairs."
    )
    ap.add_argument(
        "--src_dir",
        type=str,
        default="smile_data_n8192_scale10_rot0_trans_0_0",
        help="input folder containing train.npy (and optional val.npy, *_labels.npy)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="smile_data_n8192_scale10_rot0_trans_0_0_H_{n}",
        help="output folder (default: <src_dir>_pairs_n{n}_H)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=32,
        help="number of points to sample (default: 32)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="which split to sample from (all = concat of available splits)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for sampling",
    )
    ap.add_argument(
        "--with_replacement",
        action="store_true",
        help="sample with replacement (default: without)",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="also save CSV with (x,y,x_H,y_H[,label])",
    )
    ap.add_argument(
        "--scatter_dot",
        type=int,
        default=6,
        help="scatter dot size",
    )
    ap.add_argument(
        "--heatmap_bins",
        type=int,
        default=128,
        help="heatmap bins",
    )

    # Option A: direct H
    ap.add_argument("--H", type=float, nargs=9, default=[32, -13, 100, 55, 8, 200, 0.05, 0.005, 1.2], help="a b c d e f g h i (row-major)")

    # Option B: parametric H (used if --H is None)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--rotate_deg", type=float, default=60.0)
    ap.add_argument(
        "--translate",
        type=float,
        nargs=2,
        default=[100.0, 200.0],
    )
    ap.add_argument(
        "--anisotropic_scale",
        type=float,
        nargs=2,
        default=[3.2, 0.8],
    )
    ap.add_argument(
        "--perspective",
        type=float,
        nargs=2,
        default=[0.0001, 0.001],
        help="(p, q) for bottom row [p q 1]",
    )

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src_dir)
    if not src.exists():
        raise FileNotFoundError(f"src_dir not found: {src}")

    if args.out_dir is None:
        out = Path(str(src) + f"_pairs_n{args.n}_H")
    else:
        out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------
    # 1) Load available splits
    # ----------------------
    train = maybe_load(src / "train.npy")
    val = maybe_load(src / "val.npy")
    tr_y = maybe_load(src / "train_labels.npy")
    va_y = maybe_load(src / "val_labels.npy")

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
        if all(lbl is not None for lbl in labels):
            Y = np.concatenate(labels)
        else:
            Y = None

        src_name = "all"
        src_lengths = lengths
        src_offsets = offsets

    N_total = len(X)
    n = args.n
    if not args.with_replacement and n > N_total:
        raise ValueError(
            f"Requested n={n} > available N={N_total} without replacement. "
            f"Use --with_replacement or decrease n."
        )

    # ----------------------
    # 2) Sample indices
    # ----------------------
    if args.with_replacement:
        idx = rng.integers(low=0, high=N_total, size=n, dtype=np.int64)
    else:
        idx = rng.choice(N_total, size=n, replace=False)

    X_src = X[idx].astype(np.float32)  # [n,2]
    Y_src = (Y[idx] if Y is not None else None)

    # ----------------------
    # 3) Build homography H
    # ----------------------
    if args.H is not None:
        H = parse_H_from_flat_list(args.H)
        H_mode = "direct"
        H_params = None
    else:
        H = build_homography_from_params(
            scale=args.scale,
            rotate_deg=args.rotate_deg,
            translate=tuple(args.translate),
            anisotropic_scale=(
                tuple(args.anisotropic_scale) if args.anisotropic_scale is not None else None
            ),
            perspective=tuple(args.perspective),
        )
        H_mode = "parametric"
        H_params = {
            "scale": float(args.scale),
            "rotate_deg": float(args.rotate_deg),
            "translate": list(map(float, args.translate)),
            "anisotropic_scale": (
                list(map(float, args.anisotropic_scale))
                if args.anisotropic_scale is not None
                else None
            ),
            "perspective": list(map(float, args.perspective)),
            "composition": "H = P @ T @ R @ S_aniso @ S_iso  (column vectors)",
        }

    # ----------------------
    # 4) Apply H only to sampled points
    # ----------------------
    X_tgt, n_valid_w = apply_homography(X_src, H, drop_invalid=False)  # keep 1:1 pairs

    # form pair array [n,4] = (x_src, y_src, x_tgt, y_tgt)
    pairs = np.concatenate([X_src, X_tgt], axis=1).astype(np.float32)  # [n,4]

    # ----------------------
    # 5) Save arrays
    # ----------------------
    np.save(out / "src.npy", X_src)
    np.save(out / "tgt.npy", X_tgt)
    np.save(out / "pairs.npy", pairs)
    if Y_src is not None:
        np.save(out / "pairs_labels.npy", Y_src.astype(np.int32))

    # Optional CSV
    if args.csv:
        csv_path = out / "pairs.csv"
        header = "x_src,y_src,x_tgt,y_tgt\n" if Y_src is None else "x_src,y_src,x_tgt,y_tgt,label\n"
        csv_path.write_text(header, encoding="utf-8")
        with open(csv_path, "ab") as f:
            if Y_src is None:
                np.savetxt(f, pairs, fmt="%.6f", delimiter=b",")
            else:
                arr = np.c_[pairs, Y_src]
                np.savetxt(f, arr, fmt=[b"%.6f", b"%.6f", b"%.6f", b"%.6f", b"%d"], delimiter=b",")

    # ----------------------
    # 6) Visualizations
    # ----------------------
    save_scatter(
        X_src,
        out / "fig_scatter_src.png",
        title=f"Sampled {src_name} (src)  n={n}",
        s=args.scatter_dot,
        labels=Y_src,
    )
    save_heatmap(X_src, out / "fig_heatmap_src.png", bins=args.heatmap_bins)

    save_scatter(
        X_tgt,
        out / "fig_scatter_tgt.png",
        title=f"Transformed (tgt)  n={n}",
        s=args.scatter_dot,
        labels=Y_src,
    )
    save_heatmap(X_tgt, out / "fig_heatmap_tgt.png", bins=args.heatmap_bins)

    # all-in-one: overlay src/tgt in different colors (no labels)
    out_all = out / "fig_scatter_src_tgt.png"
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(X_src[:, 0], X_src[:, 1], s=args.scatter_dot, edgecolors="none", label="src")
    plt.scatter(X_tgt[:, 0], X_tgt[:, 1], s=args.scatter_dot, edgecolors="none", marker="x", label="tgt")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"src vs tgt  n={n}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_all, dpi=150, bbox_inches="tight")
    plt.close()

    # ----------------------
    # 7) Metadata
    # ----------------------
    meta = {
        "source_dir": str(src.resolve()),
        "source_split": args.split,
        "num_in_total": int(N_total),
        "num_sampled": int(n),
        "with_replacement": bool(args.with_replacement),
        "seed": int(args.seed),
        "has_labels": bool(Y is not None),
        "indices_in_concat_space": idx.tolist(),
        "concat_offsets": src_offsets,
        "concat_lengths": src_lengths,
        "H_mode": H_mode,
        "H_matrix": H.tolist(),
        "H_params": H_params,
        "n_valid_w": int(n_valid_w),
        "files": {
            "src": str((out / "src.npy").resolve()),
            "tgt": str((out / "tgt.npy").resolve()),
            "pairs": str((out / "pairs.npy").resolve()),
            "pairs_labels": str((out / "pairs_labels.npy").resolve())
            if Y_src is not None
            else None,
            "scatter_src": str((out / "fig_scatter_src.png").resolve()),
            "scatter_tgt": str((out / "fig_scatter_tgt.png").resolve()),
            "scatter_src_tgt": str(out_all.resolve()),
            "heatmap_src": str((out / "fig_heatmap_src.png").resolve()),
            "heatmap_tgt": str((out / "fig_heatmap_tgt.png").resolve()),
        },
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ----------------------
    # 8) Logs
    # ----------------------
    print("[OK] Saved paired (src, tgt) data to:", out.resolve())
    print(" - src.npy (original sampled points)")
    print(" - tgt.npy (transformed points)")
    print(" - pairs.npy ((x_src,y_src,x_tgt,y_tgt) pairs)", "+ pairs_labels.npy" if Y_src is not None else "")
    if args.csv:
        print(" - pairs.csv")
    print(" - fig_scatter_*.png / fig_heatmap_*.png / fig_scatter_src_tgt.png")
    print(" - metadata.json")
    print(" - H =\n", H)
    if H_params is not None:
        print(" - H composition:", H_params["composition"])


if __name__ == "__main__":
    main()
