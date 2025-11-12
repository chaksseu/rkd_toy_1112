#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a 3x3 projective transform (homography) to 2D datasets.

- Input: a folder containing train.npy (and optionally val.npy, and *_labels.npy)
         Each .npy is shaped [N, 2] (x, y) and labels are [N] (optional).
- Transform: H @ [x, y, 1]^T -> [x', y', w']^T, then (x'/w', y'/w')
- Output: transformed train_H.npy / val_H.npy (+ CSV opt), labeled figures, metadata.
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_scatter(xy: np.ndarray, path: Path, title: str = "", s: int = 6, c=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    if c is None:
        plt.scatter(xy[:, 0], xy[:, 1], s=s, edgecolors="none")
    else:
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
    S_iso = np.array([[s, 0, 0],
                      [0, s, 0],
                      [0, 0, 1]], dtype=np.float64)

    # anisotropic scale (optional)
    if anisotropic_scale is not None:
        sx, sy = anisotropic_scale
        S_aniso = np.array([[sx, 0, 0],
                            [0, sy, 0],
                            [0, 0, 1]], dtype=np.float64)
    else:
        S_aniso = np.eye(3, dtype=np.float64)

    # rotation
    th = math.radians(rotate_deg)
    c, s_ = math.cos(th), math.sin(th)
    R = np.array([[c, -s_, 0],
                  [s_,  c, 0],
                  [0,   0, 1]], dtype=np.float64)

    # translation
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)

    # perspective skew
    P = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [p, q, 1]], dtype=np.float64)

    H = P @ T @ R @ S_aniso @ S_iso
    return H


def parse_H_from_flat_list(flat9):
    a,b,c,d,e,f,g,h,i = [float(x) for x in flat9]
    H = np.array([[a,b,c],[d,e,f],[g,h,i]], dtype=np.float64)
    return H


def apply_homography(xy: np.ndarray, H: np.ndarray, w_eps: float = 1e-9, drop_invalid: bool = True):
    """
    xy: [N,2], H: [3,3]
    Returns transformed points [M,2] (M==N unless dropping near-singular w).
    """
    N = xy.shape[0]
    ones = np.ones((N,1), dtype=np.float64)
    homo = np.hstack([xy.astype(np.float64), ones])  # [N,3]
    out = (H @ homo.T).T                              # [N,3]
    w = out[:,2:3]
    mask = np.abs(w) > w_eps
    if drop_invalid:
        valid = mask[:,0]
        out = out[valid]
        w = w[valid]
    # normalize
    out_xy = out[:, :2] / w
    return out_xy.astype(np.float32), mask.sum() if drop_invalid else N


def maybe_load(path: Path):
    return np.load(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Apply 3x3 homography to 2D dataset(s)")
    ap.add_argument("--src_dir", type=str, default="smile_data_n8192_scale10_rot0_trans_0_0", help="input folder containing train.npy (and optional val.npy, *_labels.npy)")
    ap.add_argument("--out_dir", type=str, default="smile_data_n8192_scale10_rot0_trans_0_0_H_32_-13_100_55_8_200_0.05_0.005_1.2", help="output folder (default: <src_dir>_H)")
    ap.add_argument("--csv", action="store_true", help="also save CSVs")
    ap.add_argument("--scatter_dot", type=int, default=6, help="scatter dot size")
    ap.add_argument("--heatmap_bins", type=int, default=128)
    ap.add_argument("--H", type=float, nargs=9, default=[32, -13, 100, 55, 8, 200, 0.05, 0.005, 1.2], help="a b c d e f g h i (row-major)")

    # Option A: direct H

    # Option B: parametric H (used if --H is None)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--rotate_deg", type=float, default=60.0)
    ap.add_argument("--translate", type=float, nargs=2, default=[100.0, 200.0])
    ap.add_argument("--anisotropic_scale", type=float, nargs=2, default=[3.2, 0.8])
    ap.add_argument("--perspective", type=float, nargs=2, default=[0.0001, 0.001], help="(p, q) for bottom row [p q 1]")

    args = ap.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir) if args.out_dir is not None else Path(str(src) + "_H")
    out.mkdir(parents=True, exist_ok=True)

    # Load inputs
    train = np.load(src / "train.npy").astype(np.float32)
    val = maybe_load(src / "val.npy")
    train_labels = maybe_load(src / "train_labels.npy")
    val_labels = maybe_load(src / "val_labels.npy")

    # Build H
    if args.H is not None:
        H = parse_H_from_flat_list(args.H)
        H_mode = "direct"
        H_params = None
    else:
        H = build_homography_from_params(
            scale=args.scale,
            rotate_deg=args.rotate_deg,
            translate=tuple(args.translate),
            anisotropic_scale=(tuple(args.anisotropic_scale) if args.anisotropic_scale is not None else None),
            perspective=tuple(args.perspective),
        )
        H_mode = "parametric"
        H_params = {
            "scale": float(args.scale),
            "rotate_deg": float(args.rotate_deg),
            "translate": list(map(float, args.translate)),
            "anisotropic_scale": (list(map(float, args.anisotropic_scale)) if args.anisotropic_scale is not None else None),
            "perspective": list(map(float, args.perspective)),
            "composition": "H = P @ T @ R @ S_aniso @ S_iso  (column vectors)"
        }

    # Apply H
    train_H, n_train_valid = apply_homography(train, H, drop_invalid=True)
    val_H, n_val_valid = (None, 0)
    if val is not None:
        val_H, n_val_valid = apply_homography(val, H, drop_invalid=True)

    # Save arrays
    np.save(out / "train.npy", train_H)
    if train_labels is not None:
        np.save(out / "train_labels.npy", train_labels)

    if val_H is not None:
        np.save(out / "val.npy", val_H)
        if val_labels is not None:
            np.save(out / "val_labels.npy", val_labels)

    # Optional CSVs
    if args.csv:
        with open(out / "train.csv", "w", encoding="utf-8") as f:
            if train_labels is None:
                f.write("x,y\n")
                np.savetxt(f, train_H, fmt="%.6f", delimiter=",")
            else:
                f.write("x,y,label\n")
                np.savetxt(f, np.c_[train_H, train_labels], fmt=["%.6f","%.6f","%d"], delimiter=",")
        if val_H is not None:
            with open(out / "val.csv", "w", encoding="utf-8") as f:
                if val_labels is None:
                    f.write("x,y\n")
                    np.savetxt(f, val_H, fmt="%.6f", delimiter=",")
                else:
                    f.write("x,y,label\n")
                    np.savetxt(f, np.c_[val_H, val_labels], fmt=["%.6f","%.6f","%d"], delimiter=",")

    # Metadata
    meta = {
        "source_dir": str(src.resolve()),
        "num_train_in": int(train.shape[0]),
        "num_val_in": (int(val.shape[0]) if val is not None else 0),
        "num_train_out": int(train_H.shape[0]),
        "num_val_out": (int(val_H.shape[0]) if val_H is not None else 0),
        "n_train_valid_w": int(n_train_valid),
        "n_val_valid_w": int(n_val_valid),
        "H_mode": H_mode,
        "H_matrix": H.tolist(),
        "H_params": H_params,
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Visualizations (equal aspect)
    # color by label if available
    c_train = None if train_labels is None else (train_labels.astype(int) % 10)
    save_scatter(train_H, out / "fig_scatter_train.png",
                 title=f"train (H)  N={train_H.shape[0]}", s=args.scatter_dot, c=c_train)
    save_heatmap(train_H, out / "fig_heatmap_train.png", bins=args.heatmap_bins)

    if val_H is not None and len(val_H) > 0:
        c_val = None if val_labels is None else (val_labels.astype(int) % 10)
        save_scatter(val_H, out / "fig_scatter_val.png",
                     title=f"val (H)  N={val_H.shape[0]}", s=args.scatter_dot, c=c_val)
        save_heatmap(val_H, out / "fig_heatmap_val.png", bins=args.heatmap_bins)

    # All-in-one (if both exist, concat for overview)
    if val_H is not None and len(val_H) > 0:
        all_H = np.vstack([train_H, val_H])
        save_scatter(all_H, out / "fig_scatter_all.png",
                     title=f"all (H)  N={all_H.shape[0]}", s=args.scatter_dot)
        save_heatmap(all_H, out / "fig_heatmap_all.png", bins=args.heatmap_bins)
    else:
        save_scatter(train_H, out / "fig_scatter_all.png",
                     title=f"all (H)  N={train_H.shape[0]}", s=args.scatter_dot)
        save_heatmap(train_H, out / "fig_heatmap_all.png", bins=args.heatmap_bins)

    print("[OK] Saved to:", out.resolve())
    print(" - train.npy", "(+ train_labels.npy)" if train_labels is not None else "")
    if val_H is not None:
        print(" - val.npy", "(+ val_labels.npy)" if val_labels is not None else "")
    print(" - fig_scatter_*.png / fig_heatmap_*.png")
    print(" - metadata.json")
    print(" - H =\n", H)
    if H_params is not None:
        print(" - H composition:", H_params["composition"])


if __name__ == "__main__":
    main()
