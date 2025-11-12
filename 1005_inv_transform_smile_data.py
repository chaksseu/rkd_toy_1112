#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Invert a previously applied 3x3 homography on a 2D dataset:
Load transformed (x', y'), apply H^{-1} to recover (x, y),
then save arrays, CSV (optional), and visualizations.

- Input:  src_dir with train.npy (and optionally val.npy, *_labels.npy)
          Optionally src_dir/metadata.json containing "H_matrix"
- H source:
    (A) --H a b c d e f g h i   (row-major), or
    (B) auto-read from metadata.json ("H_matrix"), or
    (C) if neither, error
- Output: out_dir with recovered train.npy (/val.npy), labels passthrough,
          optional CSVs, scatter/heatmap PNGs, and metadata.json.
"""

import argparse, json, math
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
        range=[[xmin - pad, xmax + pad], [ymin - pad, ymax + pad]]
    )
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(H.T, origin="lower", aspect="equal",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def parse_H_from_flat_list(flat9):
    a,b,c,d,e,f,g,h,i = [float(x) for x in flat9]
    H = np.array([[a,b,c],[d,e,f],[g,h,i]], dtype=np.float64)
    return H

def load_H_from_metadata(src_dir: Path):
    meta_path = src_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        Hm = meta.get("H_matrix", None)
        if Hm is None:
            return None
        H = np.array(Hm, dtype=np.float64)
        if H.shape != (3,3):
            return None
        return H
    except Exception:
        return None

def apply_homography(xy: np.ndarray, H: np.ndarray, w_eps: float = 1e-9, drop_invalid: bool = True):
    """
    xy: [N,2], H: [3,3]
    Returns (transformed_xy, n_valid_w)
    """
    N = xy.shape[0]
    homo = np.hstack([xy.astype(np.float64), np.ones((N,1), dtype=np.float64)])  # [N,3]
    out = (H @ homo.T).T                                                          # [N,3]
    w = out[:, 2:3]
    mask = np.abs(w) > w_eps
    if drop_invalid:
        valid = mask[:, 0]
        out = out[valid]
        w = w[valid]
    out_xy = out[:, :2] / w
    return out_xy.astype(np.float32), int(mask.sum()) if drop_invalid else N

def main():
    ap = argparse.ArgumentParser(description="Apply inverse homography to revert transformed dataset")
    # 기본값은 예시: 너가 만든 H 폴더
    ap.add_argument("--src_dir", type=str,
                    default="smile_data_n8192_scale10_rot0_trans_0_0_H_32_-13_100_55_8_200_0.05_0.005_1.2_n16",
                    help="folder containing the transformed dataset (train.npy/val.npy)")
    ap.add_argument("--out_dir", type=str,
                    default=f"recovered_from_H",
                    help="output folder to save the recovered dataset")
    ap.add_argument("--csv", action="store_true", help="also save CSVs")
    ap.add_argument("--scatter_dot", type=int, default=6, help="scatter dot size")
    ap.add_argument("--heatmap_bins", type=int, default=128, help="heatmap bins")
    ap.add_argument("--w_eps", type=float, default=1e-9, help="|w| threshold; below is considered invalid")
    ap.add_argument("--keep_all", action="store_true",
                    help="do NOT drop invalid points (may produce NaN/Inf). Default: drop invalid.")
    # H 입력 옵션
    ap.add_argument("--H", type=float, nargs=9, default=None,
                    help="Forward homography H (row-major): a b c d e f g h i. If omitted, try reading metadata.json.")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드
    Xtr = maybe_load(src / "train.npy")
    Xva = maybe_load(src / "val.npy")
    Ytr = maybe_load(src / "train_labels.npy")
    Yva = maybe_load(src / "val_labels.npy")
    if Xtr is None and Xva is None:
        raise FileNotFoundError(f"No train.npy or val.npy in {src}")

    # 2) 정방향 H 로드
    if args.H is not None:
        H = parse_H_from_flat_list(args.H)
        H_src = "cli"
    else:
        H = load_H_from_metadata(src)
        H_src = "metadata.json" if H is not None else None

    if H is None:
        raise ValueError("H not provided. Pass --H a b c d e f g h i or ensure metadata.json has 'H_matrix'.")

    # 3) 역행렬 H_inv (프로젝티브 스케일 정규화)
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]

    # 4) 적용 (student→teacher 복원)
    drop_invalid = not args.keep_all

    stats = {}
    if Xtr is not None:
        Xtr_rec, n_valid_tr = apply_homography(Xtr, H_inv, w_eps=args.w_eps, drop_invalid=drop_invalid)
        stats["train_in"] = int(len(Xtr))
        stats["train_valid_w"] = int(n_valid_tr)
        stats["train_out"] = int(len(Xtr_rec))
        save_split(Xtr_rec, Ytr if (Ytr is not None and (not drop_invalid or len(Ytr)==len(Xtr_rec))) else Ytr, out, "train", csv=args.csv)
        # 시각화
        save_scatter(Xtr_rec, out / "fig_scatter_train.png",
                     title=f"Recovered train  N={len(Xtr_rec)}", s=args.scatter_dot, labels=Ytr if (Ytr is not None and len(Ytr)==len(Xtr_rec)) else None)
        save_heatmap(Xtr_rec, out / "fig_heatmap_train.png", bins=args.heatmap_bins)

    if Xva is not None:
        Xva_rec, n_valid_va = apply_homography(Xva, H_inv, w_eps=args.w_eps, drop_invalid=drop_invalid)
        stats["val_in"] = int(len(Xva))
        stats["val_valid_w"] = int(n_valid_va)
        stats["val_out"] = int(len(Xva_rec))
        save_split(Xva_rec, Yva if (Yva is not None and (not drop_invalid or len(Yva)==len(Xva_rec))) else Yva, out, "val", csv=args.csv)
        save_scatter(Xva_rec, out / "fig_scatter_val.png",
                     title=f"Recovered val  N={len(Xva_rec)}", s=args.scatter_dot, labels=Yva if (Yva is not None and len(Yva)==len(Xva_rec)) else None)
        save_heatmap(Xva_rec, out / "fig_heatmap_val.png", bins=args.heatmap_bins)

    # all 시각화
    if Xtr is not None and (src / "val.npy").exists():
        all_rec = np.vstack([np.load(out / "train.npy"), np.load(out / "val.npy")])
        save_scatter(all_rec, out / "fig_scatter_all.png",
                     title=f"Recovered all  N={len(all_rec)}", s=args.scatter_dot)
        save_heatmap(all_rec, out / "fig_heatmap_all.png", bins=args.heatmap_bins)
    elif Xtr is not None:
        save_scatter(np.load(out / "train.npy"), out / "fig_scatter_all.png",
                     title=f"Recovered all  N={len(np.load(out / 'train.npy'))}", s=args.scatter_dot)
        save_heatmap(np.load(out / "train.npy"), out / "fig_heatmap_all.png", bins=args.heatmap_bins)
    else:
        save_scatter(np.load(out / "val.npy"), out / "fig_scatter_all.png",
                     title=f"Recovered all  N={len(np.load(out / 'val.npy'))}", s=args.scatter_dot)
        save_heatmap(np.load(out / "val.npy"), out / "fig_heatmap_all.png", bins=args.heatmap_bins)

    # 5) 메타 저장
    meta = {
        "source_dir": str(src.resolve()),
        "H_source": H_src,
        "H_forward": H.tolist(),
        "H_inverse": H_inv.tolist(),
        "w_eps": float(args.w_eps),
        "drop_invalid": bool(drop_invalid),
        "scatter_dot": int(args.scatter_dot),
        "heatmap_bins": int(args.heatmap_bins),
        "stats": stats
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 콘솔 로그
    print("[OK] Saved recovered dataset to:", out.resolve())
    for k, v in stats.items():
        print(f" - {k}: {v}")
    print(" - figures: fig_scatter_* / fig_heatmap_*")
    print(" - metadata.json")
    print(" - H (forward):\n", H)
    print(" - H_inv (normalized):\n", H_inv)

if __name__ == "__main__":
    main()
