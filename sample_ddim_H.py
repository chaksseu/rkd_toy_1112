#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a trained 2D toy diffusion model (MLPDenoiser), run DDIM sampling, and save:
  1) x0 scatter (no H)
  2) x0 scatter with homography H applied
  3) side-by-side comparison (axes synced)

- H can be provided via --H (comma-separated 9 numbers) or --H-file (.npy/.json/.txt).
- H application space selectable: model (default, pre-denorm) or plot (post-denorm).

Example:
python ddim_sample_and_plot.py \
  --ckpt runs/1002_only_diff_loss_teacher8192/ckpt_student_step500000.pt \
  --norm-stats smile_data_n8192_scale10_rot0_trans_0_0/teacher_normalization_stats.json \
  --T 50 --ddim-steps 25 --num-samples 8192 --seed 42 --out runs/1002_only_diff_loss_teacher8192/figs \
  --eta 0.0 --device cuda:1 \
  --H "1,0,0,0,1,0,0,0,1" --H-space model
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler

# -------------------- Model -------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class MLPDenoiser(nn.Module):
    """ε-predictor for 2D toy; same as training."""
    def __init__(self, in_dim=2, time_dim=64, hidden=256, depth=8, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t)
        h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))

# -------------------- Schedulers -------------------- #
def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"
    return train_sched, sample_sched

# -------------------- Utils -------------------- #
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_norm_stats(json_path: str):
    if not json_path:
        return None, None
    with open(json_path, "r") as f:
        d = json.load(f)
    mu = np.array(d["mean"], dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma

def denormalize_np(arr: np.ndarray, mu: np.ndarray | None, sigma: np.ndarray | None):
    if mu is None or sigma is None:
        return arr
    return arr * sigma + mu

def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    data = np.asarray(data)
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    base = max(dx, dy, 1e-3)
    pad = pad_ratio * base
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0
    return (xmid - half, xmid + half), (ymid - half, ymid + half)

# -------------------- Homography helpers -------------------- #
def parse_H_from_string(s: str) -> np.ndarray:
    vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    if len(vals) != 9:
        raise ValueError("H string must contain 9 numbers (row-major).")
    H = np.array(vals, dtype=np.float32).reshape(3, 3)
    return H

def load_H_from_file(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        H = np.load(p)
        if H.shape != (3, 3):
            raise ValueError("H .npy must be shape (3,3).")
        return H.astype(np.float32)
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            vals = json.load(f)
        H = np.array(vals, dtype=np.float32).reshape(3, 3)
        return H
    else:
        # txt or others: read whitespace/comma separated 9 numbers
        txt = p.read_text()
        return parse_H_from_string(txt)

def apply_homography_rowvec(xy: np.ndarray, H: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Row-vector convention: [x, y, 1] @ H^T = [X, Y, W], then (x',y')=(X/W, Y/W).
    xy: (N,2), H: (3,3)
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2)")
    N = xy.shape[0]
    ones = np.ones((N, 1), dtype=xy.dtype)
    homo = np.concatenate([xy, ones], axis=1)             # (N,3)
    out = homo @ H.T                                      # (N,3)
    W = out[:, 2:3]
    # keep sign, avoid zero
    W_safe = np.sign(W) * np.maximum(np.abs(W), eps)
    xy_t = out[:, :2] / W_safe
    return xy_t

# -------------------- Sampling -------------------- #
@torch.no_grad()
def sample_x0_ddim(model: nn.Module, scheduler: DDIMScheduler, num_samples: int, device: torch.device,
                   sample_steps: int, dim: int = 2, eta: float = 0.0):
    """DDIM sampling producing x0 from N(0, I)."""
    scheduler.set_timesteps(sample_steps, device=device)
    x = torch.randn(num_samples, dim, device=device)
    for t in scheduler.timesteps:  # [T-1, ..., 0]
        t_b = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
        x_in = scheduler.scale_model_input(x, t)
        eps = model(x_in, t_b)
        x = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
    return x

# -------------------- Plotting -------------------- #
def save_scatter(path: Path, data: np.ndarray, title: str, dpi: int = 150, pad_ratio: float = 0.05):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.scatter(data[:, 0], data[:, 1], s=6, edgecolors="none")
    ax.set_aspect("equal", adjustable="box")
    xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_compare(path: Path, A: np.ndarray, B: np.ndarray, title_left: str, title_right: str,
                 dpi: int = 150, pad_ratio: float = 0.05):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharex=True, sharey=True)
    both = np.vstack([A, B])
    xlim, ylim = _square_limits_from(both, pad_ratio=pad_ratio)
    for ax, data, ttl in ((axes[0], A, title_left), (axes[1], B, title_right)):
        ax.scatter(data[:, 0], data[:, 1], s=6, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(ttl)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# -------------------- Main -------------------- #
DDIM_STEP = 50
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/1010_rkd_xt_b512_ddim_full_init_student_H_with_diffW10_randN/ckpt_student_step560000.pt", type=str, help="Path to model checkpoint (.pt)")
    p.add_argument("--norm-stats", default="smile_data_n8192_scale10_rot0_trans_0_0/teacher_normalization_stats.json", type=str, help="Path to teacher_normalization_stats.json")
    p.add_argument("--out", default="vis_ddim", type=str, help="Output directory")
    p.add_argument("--T", default=50, type=int, help="Training total diffusion steps (0..T-1)")
    p.add_argument("--ddim-steps", default=DDIM_STEP, type=int, help="Number of DDIM inference steps")
    p.add_argument("--eta", default=0.0, type=float, help="DDIM eta (0 = deterministic)")
    p.add_argument("--num-samples", default=8192, type=int, help="Number of samples")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--dim", default=2, type=int)
    p.add_argument("--time-dim", default=64, type=int)
    p.add_argument("--hidden", default=256, type=int)
    p.add_argument("--depth", default=8, type=int)
    p.add_argument("--dpi", default=150, type=int)
    p.add_argument("--title", default="DDIM Samples (x0)", type=str)
    p.add_argument("--filename", default=f"ddim_samples{DDIM_STEP}.png", type=str)

    # --- NEW: Homography options ---
    p.add_argument("--H", default="0.28118, -0.39711, -0.58264,-0.18684, 1.58305, -0.36159, -0.03023, 0.01408, 0.96309", type=str, help="Comma-separated 9 numbers (row-major) for 3x3 H")
    p.add_argument("--H-file", default="", type=str, help="Path to H file (.npy/.json/.txt)")
    p.add_argument("--H-space", default="model", choices=["model", "plot"],
                   help="Apply H in 'model' space (pre-denorm) or 'plot' space (post-denorm)")
    p.add_argument("--H-eps", default=1e-6, type=float, help="Epsilon to stabilize W division")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # schedulers
    train_sched, sample_sched = build_schedulers(args.T)

    # model
    model = MLPDenoiser(in_dim=args.dim, time_dim=args.time_dim, hidden=args.hidden, depth=args.depth, out_dim=args.dim)
    ckpt_path = Path(args.ckpt)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    # sampling
    x0 = sample_x0_ddim(
        model=model,
        scheduler=sample_sched,
        num_samples=args.num_samples,
        device=device,
        sample_steps=args.ddim_steps,
        dim=args.dim,
        eta=args.eta,
    )
    x0_np = x0.detach().cpu().numpy()

    # normalization
    mu, sigma = load_norm_stats(args.norm_stats)

    # --- Prepare H (optional) ---
    H = None
    if args.H_file:
        H = load_H_from_file(args.H_file)
    elif args.H:
        H = parse_H_from_string(args.H)
    if H is not None:
        H = np.asarray(H, dtype=np.float32).reshape(3, 3)

    # --- Branches: no-H vs H-applied ---
    # base (no H)
    x0_plot = denormalize_np(x0_np, mu, sigma)

    if H is None:
        # Just save the base plot
        out_path = out_dir / args.filename
        save_scatter(out_path, x0_plot,
                     f"{args.title}\nsteps={args.ddim_steps}, eta={args.eta}, N={args.num_samples}",
                     dpi=args.dpi)
        print(f"[OK] Saved: {out_path.resolve()}")
        return

    # With H applied
    if args.H_space == "model":
        # Apply in model space (pre-denorm), then denorm for plotting
        x0_H_model = apply_homography_rowvec(x0_np, H, eps=float(args.H_eps))
        x0_H_plot = denormalize_np(x0_H_model, mu, sigma)
    else:
        # Apply in plot space (post-denorm)
        x0_H_plot = apply_homography_rowvec(x0_plot, H, eps=float(args.H_eps))

    # --- Save all three: base, H, compare ---
    base_path = out_dir / args.filename
    save_scatter(base_path, x0_plot,
                 f"{args.title} (no H)\nsteps={args.ddim_steps}, eta={args.eta}, N={args.num_samples}",
                 dpi=args.dpi)

    H_path = out_dir / (Path(args.filename).stem + "_H.png")
    save_scatter(H_path, x0_H_plot,
                 f"{args.title} (with H)\nsteps={args.ddim_steps}, eta={args.eta}, N={args.num_samples}",
                 dpi=args.dpi)

    cmp_path = out_dir / (Path(args.filename).stem + "_compare.png")
    save_compare(cmp_path, x0_plot, x0_H_plot, "No H", "With H", dpi=args.dpi)

    print(f"[OK] Saved: {base_path.resolve()}")
    print(f"[OK] Saved: {H_path.resolve()}")
    print(f"[OK] Saved: {cmp_path.resolve()}")

if __name__ == "__main__":
    main()
