#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student DDIM Trajectory Visualizer (INVERTED NOISE from STUDENT DATA)
- STUDENT(MLP epsilon-predictor) 로드
- STUDENT DATA에서 x0 배치를 뽑아 정규화 -> DDIM Inversion으로 x_T (inverted noise) 생성
- 그 x_T로부터 DDIM trajectory 수집/저장 (norm/denorm 모두: NPY/overlay/frames/GIF)
- 모델 score(vector field) 시각화 (norm/denorm)
"""

import os, re, math, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler

# ===================== CONFIG ===================== #
CUDA_NUM = 3
BATCH_SIZE = 1024
WANDB_NAME = f"1107_lr1e4_n32_b{BATCH_SIZE}_ddim_50_150_steps"

CONFIG = {
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/{WANDB_NAME}",
    "T": 1000,
    "seed": 42,
    "dim": 2,
    "student_hidden": 256,
    "student_depth": 8,
    "student_time_dim": 64,
    "auto_find_ckpt_pattern": r"ckpt_student_step(\d+)\.pt",
}

# ===================== UTILS ===================== #
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def find_latest_student_ckpt(out_dir: Path, pattern: str):
    regex = re.compile(pattern)
    best_num, best_path = -1, None
    if not out_dir.exists(): return None
    for f in out_dir.glob("ckpt_student_step*.pt"):
        m = regex.fullmatch(f.name)
        if m:
            step = int(m.group(1))
            if step > best_num:
                best_num, best_path = step, f
    return str(best_path) if best_path else None

def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    data = np.asarray(data)
    xmin, xmax = float(data[:,0].min()), float(data[:,0].max())
    ymin, ymax = float(data[:,1].min()), float(data[:,1].max())
    dx, dy = xmax-xmin, ymax-ymin
    base = max(dx, dy, 1e-3)
    pad = pad_ratio * base
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    xmid = (xmin+xmax)/2.0; ymid = (ymin+ymax)/2.0
    span = max(xmax-xmin, ymax-ymin); half = span/2.0
    return (xmid-half, xmid+half), (ymid-half, ymid+half)

def colors_from_noise_ids(ids: np.ndarray, alpha: float = 0.9):
    ids = np.asarray(ids, dtype=np.int64)
    phi = 0.6180339887498949
    hues = (ids * phi) % 1.0
    cmap = plt.get_cmap("hsv")
    cols = cmap(hues); cols[:,3] = alpha
    return cols

def load_norm_stats(json_path: str):
    p = Path(json_path)
    with p.open("r") as f:
        d = json.load(f)
    mu = np.array(d["mean"], dtype=np.float32)
    sg = np.array(d["std"],  dtype=np.float32)
    return mu, sg

def load_student_x0(path: str, fmt: str = "npy", dim: int = 2) -> np.ndarray:
    p = Path(path)
    if fmt == "npy":
        X = np.load(p)
    elif fmt == "csv":
        X = np.loadtxt(p, delimiter=",")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    assert X.ndim == 2 and X.shape[1] >= dim, f"Expect (N,{dim}+) array."
    return X[:, :dim].astype(np.float32)

# ===================== MODEL ===================== #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0,1))
        return emb

class MLPDenoiser(nn.Module):
    def __init__(self, in_dim=2, time_dim=64, hidden=256, depth=8, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i==0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t); h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x,t)

# ===================== SCHEDULERS ===================== #
def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
    train_sched.config.prediction_type = "epsilon"
    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"
    return train_sched, sample_sched

# ===================== SAMPLING (TRAJ) ===================== #
@torch.no_grad()
def collect_student_xt_seq_ddim(student: nn.Module,
                                sample_sched: DDIMScheduler,
                                z: torch.Tensor,
                                device: torch.device,
                                sample_steps: int = 100,
                                eta: float = 0.0):
    sched = DDIMScheduler.from_config(sample_sched.config)
    sched.set_timesteps(sample_steps, device=device)
    x = z.to(device).detach().clone()
    B = x.shape[0]; xs, ts = [], []
    was_train = student.training; student.eval()
    try:
        for t in sched.timesteps:
            t_int = int(t)
            xs.append(x.detach().cpu().numpy()); ts.append(t_int)
            tb = torch.full((B,), t_int, device=device, dtype=torch.long)
            xin = sched.scale_model_input(x, t)
            eps = student(xin, tb)
            x = sched.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
    finally:
        if was_train: student.train()
    return np.stack(xs, 0), np.asarray(ts, dtype=int)

@torch.no_grad()
def ddim_inverse_student_last(student: nn.Module,
                              sample_sched: DDIMScheduler,
                              x0_norm: torch.Tensor,
                              device: torch.device,
                              sample_steps: int = 100,
                              eta: float = 0.0) -> torch.Tensor:
    """
    DDIM Inversion (eta=0 권장): x0_norm -> ... -> x_T (마지막만 반환)
    x0_norm: (N, D) normalized to student training stats
    """
    inv = DDIMInverseScheduler.from_config(sample_sched.config)
    inv.set_timesteps(sample_steps, device=device)
    x = x0_norm.to(device).detach().clone()
    student.eval()
    for t in inv.timesteps:
        t_b = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        xin = inv.scale_model_input(x, t)
        eps = student(xin, t_b)
        x = inv.step(eps, t, x).prev_sample
    return x  # x_T

# ===================== SCORE FIELD ===================== #
def _alpha_bar_at(sched: DDIMScheduler, t_int: int) -> float:
    return float(sched.alphas_cumprod[int(t_int)].item())

@torch.no_grad()
def compute_score_grid_norm(student: nn.Module,
                            sample_sched: DDIMScheduler,
                            t_int: int,
                            xlim, ylim,
                            grid: int,
                            device: torch.device):
    xs = np.linspace(xlim[0], xlim[1], grid, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)

    x = torch.from_numpy(pts).to(device)
    t_b = torch.full((x.shape[0],), int(t_int), device=device, dtype=torch.long)
    t_tensor = torch.tensor(int(t_int), device=device, dtype=torch.long)

    xin = sample_sched.scale_model_input(x, t_tensor)
    eps = student(xin, t_b)  # ε(x,t)
    sigma_t = math.sqrt(1.0 - _alpha_bar_at(sample_sched, t_int)) + 1e-12
    s = -eps / sigma_t  # score ≈ -ε/σ_t

    Sx = s[:,0].reshape(Xg.shape).detach().cpu().numpy()
    Sy = s[:,1].reshape(Yg.shape).detach().cpu().numpy()
    Sm = np.sqrt(Sx*Sx + Sy*Sy)
    return Xg, Yg, Sx, Sy, Sm

def render_score_field_both(seq_norm: np.ndarray, ts: np.ndarray,
                            student: nn.Module, sample_sched: DDIMScheduler,
                            out_dir: Path, grid: int = 60, stream_density: float = 1.2,
                            mu: np.ndarray = None, sg: np.ndarray = None,
                            t_list: list = None, device: torch.device = torch.device("cpu")):
    K, B, _ = seq_norm.shape
    XY_all_norm = seq_norm.reshape(-1, 2)
    xlim_n, ylim_n = _square_limits_from(XY_all_norm, pad_ratio=0.06)

    have_den = (mu is not None) and (sg is not None)
    if have_den:
        seq_den = seq_norm * sg[None,None,:] + mu[None,None,:]
        XY_all_den = seq_den.reshape(-1, 2)
        xlim_d, ylim_d = _square_limits_from(XY_all_den, pad_ratio=0.06)

    out_norm = ensure_dir(out_dir / "score_field_norm")
    out_den  = ensure_dir(out_dir / "score_field_denorm") if have_den else None

    if not t_list:
        t_list = np.unique(np.linspace(0, len(ts)-1, num=5, dtype=int)).tolist()
        t_list = [int(ts[i]) for i in t_list]

    for t_int in t_list:
        idx = np.where(ts == t_int)[0]
        k = int(idx[0]) if len(idx) > 0 else int(np.argmin(np.abs(ts - t_int)))
        t_used = int(ts[k])

        Xg, Yg, Sx, Sy, Sm = compute_score_grid_norm(student, sample_sched, t_used,
                                                     xlim_n, ylim_n, grid, device)

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        cs = ax.contour(Xg, Yg, Sm, levels=16, linewidths=1.2, cmap="magma", alpha=0.95)
        ax.streamplot(Xg, Yg, Sx, Sy, color=Sm, cmap="magma",
                      density=stream_density, linewidth=1.0, arrowsize=1.2)
        step = max(1, grid // 18)
        ax.quiver(Xg[::step,::step], Yg[::step,::step],
                  Sx[::step,::step], Sy[::step,::step],
                  angles='xy', scale_units='xy', scale=1.0, width=0.003, color='k', alpha=0.9)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim_n); ax.set_ylim(*ylim_n)
        ax.set_title(f"Score field (norm) @ t={t_used}")
        fig.colorbar(cs, ax=ax, shrink=0.82, label="||score||")
        fig.tight_layout()
        fig.savefig(out_norm / f"score_field_t{t_used:04d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        if have_den:
            Xg_d = Xg * sg[0] + mu[0]
            Yg_d = Yg * sg[1] + mu[1]
            Sx_d = Sx / max(sg[0], 1e-12)
            Sy_d = Sy / max(sg[1], 1e-12)
            Sm_d = np.sqrt(Sx_d*Sx_d + Sy_d*Sy_d)

            fig, ax = plt.subplots(figsize=(6.2, 6.2))
            cs = ax.contour(Xg_d, Yg_d, Sm_d, levels=16, linewidths=1.2, cmap="magma", alpha=0.95)
            ax.streamplot(Xg_d, Yg_d, Sx_d, Sy_d, color=Sm_d, cmap="magma",
                          density=stream_density, linewidth=1.0, arrowsize=1.2)
            step = max(1, grid // 18)
            ax.quiver(Xg_d[::step,::step], Yg_d[::step,::step],
                      Sx_d[::step,::step], Sy_d[::step,::step],
                      angles='xy', scale_units='xy', scale=1.0, width=0.003, color='k', alpha=0.9)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xlim_d); ax.set_ylim(*ylim_d)
            ax.set_title(f"Score field (denorm) @ t={t_used}")
            fig.colorbar(cs, ax=ax, shrink=0.82, label="||score||")
            fig.tight_layout()
            fig.savefig(out_den / f"score_field_t{t_used:04d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

# ===================== PLOTTING (TRAJ, FRAMES, GIF) ===================== #
def plot_trajectories_overlay(seq_xy: np.ndarray, out_path: Path,
                              max_lines: int = 512, line_alpha: float = 0.85,
                              dot_size: int = 14, pad_ratio: float = 0.06,
                              title: str = None):
    K, B, _ = seq_xy.shape
    cols = colors_from_noise_ids(np.arange(B), alpha=line_alpha)
    XY_all = seq_xy.reshape(-1, 2)
    xlim, ylim = _square_limits_from(XY_all, pad_ratio=pad_ratio)
    pick = np.linspace(0, B-1, num=min(B, max_lines), dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for b in pick:
        xy = seq_xy[:, b, :]
        ax.plot(xy[:,0], xy[:,1], lw=1.2, alpha=line_alpha, c=cols[b])
        ax.scatter(xy[0,0],  xy[0,1],  s=dot_size, c=[cols[b]], marker='o',
                   edgecolors='k', linewidths=0.4, zorder=3)
        ax.scatter(xy[-1,0], xy[-1,1], s=dot_size, c=[cols[b]], marker='X',
                   edgecolors='k', linewidths=0.4, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title: ax.set_title(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_per_timestep_frames(seq_xy: np.ndarray, ts: np.ndarray, out_dir: Path, dot_size: int = 10):
    K, B, _ = seq_xy.shape
    cols = colors_from_noise_ids(np.arange(B), alpha=0.9)
    XY_all = seq_xy.reshape(-1, 2)
    xlim, ylim = _square_limits_from(XY_all, pad_ratio=0.06)
    frames_dir = ensure_dir(out_dir)
    for k in range(K):
        xy = seq_xy[k]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(xy[:,0], xy[:,1], s=dot_size, c=cols, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"Student $x_t$ (t={int(ts[k])})")
        fig.tight_layout()
        fig.savefig(frames_dir / f"t{int(ts[k]):04d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return frames_dir

def maybe_make_gif(frames_dir: Path, out_path: Path, fps: int = 10):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[GIF] imageio not installed; skip GIF. `pip install imageio` to enable.")
        return
    files = list(frames_dir.glob("t*.png"))
    if not files:
        print("[GIF] no frames found."); return
    def _t_of(p: Path):
        st = p.stem
        return int(st[1:]) if st.startswith("t") and st[1:].isdigit() else -1
    files_sorted = sorted(files, key=_t_of, reverse=True)  # 큰 t -> 0
    imgs = [imageio.imread(str(p)) for p in files_sorted]
    imageio.mimsave(str(out_path), imgs, duration=1.0/max(fps,1))
    print(f"[GIF] saved (large t -> 0) -> {out_path}")

# ===================== MAIN LOGIC ===================== #
@torch.no_grad()
def visualize_student_ddim_trajectories(cfg: dict,
                                        student_ckpt_path: str,
                                        n_noises: int = 128,
                                        ddim_steps: int = 100,
                                        eta: float = 0.0,
                                        save_frames: bool = True,
                                        make_gif_flag: bool = True,
                                        out_dir_override: str = None,
                                        student_stats_path: str = None,
                                        score_ts: str = "",
                                        score_grid: int = 60,
                                        score_density: float = 1.2,
                                        student_data_path: str = None,
                                        data_format: str = "npy"):
    """
    - student_data_path & student_stats_path가 주어지면:
        STUDENT DATA x0 -> normalize -> DDIM Inversion -> x_T (inverted noise) 로 시작
    - 아니면 기존처럼 랜덤 노이즈로 시작 (fallback)
    """
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_root = Path(out_dir_override) if out_dir_override else Path(cfg["out_dir"])
    out_traj = ensure_dir(out_root / "traj")

    # schedulers + student
    _, sample_sched = build_schedulers(cfg["T"])
    student = MLPDenoiser(in_dim=cfg["dim"], time_dim=cfg["student_time_dim"],
                          hidden=cfg["student_hidden"], depth=cfg["student_depth"],
                          out_dim=cfg["dim"]).to(device)
    student.load_state_dict(torch.load(student_ckpt_path, map_location=device), strict=True)
    student.eval()
    print(f"[LOAD] Student checkpoint -> {student_ckpt_path}")

    # ---------- START NOISE: inverted from student DATA ----------
    use_inversion = (student_data_path is not None) and (student_stats_path is not None) and Path(student_stats_path).exists()
    mu, sg = (None, None)
    if use_inversion:
        # 1) load stats (for normalization & later denorm)
        mu, sg = load_norm_stats(student_stats_path)
        assert mu.shape[0] >= cfg["dim"] and sg.shape[0] >= cfg["dim"], "stats dim mismatch."

        # 2) load student x0 data (raw)
        X0 = load_student_x0(student_data_path, fmt=data_format, dim=cfg["dim"])   # (N_all, D)
        N_all = X0.shape[0]
        if N_all == 0:
            raise ValueError("Loaded student data is empty.")

        # 3) sample n_noises rows (with replacement if needed)
        idx = np.random.choice(N_all, size=int(n_noises), replace=(N_all < n_noises))
        x0_sel_den = X0[idx]                         # denormalized x0 (raw domain)
        x0_sel_norm = (x0_sel_den - mu) / sg         # normalize to student domain
        x0_sel_norm_t = torch.from_numpy(x0_sel_norm).to(device)

        # 4) DDIM inversion: x0_norm -> x_T
        z = ddim_inverse_student_last(student, sample_sched, x0_sel_norm_t,
                                      device=device, sample_steps=int(ddim_steps), eta=float(eta))

        # 저장: 선택된 x0 및 inverted noise
        np.save(out_traj / "inversion_x0_denorm.npy", x0_sel_den.astype(np.float32))
        np.save(out_traj / "inversion_x0_norm.npy", x0_sel_norm.astype(np.float32))
        np.save(out_traj / "inverted_noise_z.npy", z.detach().cpu().numpy().astype(np.float32))
        print(f"[INVERT] generated inverted noise from STUDENT DATA: shape={tuple(z.shape)}")
    else:
        # fallback: pure random noise (권장 경로는 아님)
        print("[WARN] --student-data or --student-stats not provided; using pure random noise as fallback.")
        z = torch.randn(int(n_noises), cfg["dim"], device=device)

    # ---------- collect DDIM trajectories from z ----------
    seq_norm, ts = collect_student_xt_seq_ddim(student, sample_sched, z, device,
                                               sample_steps=int(ddim_steps), eta=float(eta))
    K, B, _ = seq_norm.shape

    # Save & overlay (norm)
    np.save(out_traj / "student_traj_xy_norm.npy", seq_norm)
    np.save(out_traj / "student_traj_ts.npy", ts)
    title_sfx = "inverted" if use_inversion else "random"
    plot_trajectories_overlay(seq_norm, out_traj / f"student_traj_norm_{title_sfx}_steps{K}_N{B}.png",
                              max_lines=512, title=f"Student DDIM (norm/{title_sfx}) steps={K} N={B} eta={eta}")
    if save_frames:
        f_norm = ensure_dir(out_traj / f"frames_norm_{title_sfx}")
        save_per_timestep_frames(seq_norm, ts, f_norm)
        if make_gif_flag:
            maybe_make_gif(f_norm, out_traj / f"student_traj_norm_{title_sfx}_steps{K}_N{B}.gif", fps=10)

    # Denorm copies (optional, also needed for score field in denorm)
    if mu is not None and sg is not None:
        seq_den = seq_norm * sg[None,None,:] + mu[None,None,:]
        np.save(out_traj / "student_traj_xy_denorm.npy", seq_den)
        plot_trajectories_overlay(seq_den, out_traj / f"student_traj_denorm_{title_sfx}_steps{K}_N{B}.png",
                                  max_lines=512, title=f"Student DDIM (denorm/{title_sfx}) steps={K} N={B} eta={eta}")
        if save_frames:
            f_den = ensure_dir(out_traj / f"frames_denorm_{title_sfx}")
            save_per_timestep_frames(seq_den, ts, f_den)
            if make_gif_flag:
                maybe_make_gif(f_den, out_traj / f"student_traj_denorm_{title_sfx}_steps{K}_N{B}.gif", fps=10)
    else:
        print("[DENORM] --student-stats not provided or file missing -> skip denorm trajectory outputs.")

    # -------- Pure score field (norm & denorm) --------
    if score_ts:
        t_list = [int(s) for s in score_ts.split(",") if s.strip().isdigit()]
    else:
        t_list = None  # auto-pick (5개)
    render_score_field_both(seq_norm=seq_norm, ts=ts,
                            student=student, sample_sched=sample_sched,
                            out_dir=out_traj, grid=int(score_grid), stream_density=float(score_density),
                            mu=mu, sg=sg, t_list=t_list, device=device)

    print(f"[DONE] out dir: {out_traj.resolve()}")


# RKD
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW1.0_diff0.0/ckpt_student_step080000.pt
# RKD + DIFF
# runs/1027_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_diff1.0/ckpt_student_step070000.pt
# RKD + INV + INVINV
# ckpt_student_step075000_rkd_inv_invinv.pt
# RKD + INV + INVINV + FD
# runs/1107_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.0001/ckpt_student_step110000.pt


# ===================== ARGS / ENTRY ===================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Student DDIM trajectories from INVERTED NOISE (student data) + pure score field.")
    parser.add_argument("--ckpt", type=str,
        default="runs/1107_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.1_invW0.1_invinvW1.0_fidW0.0001/ckpt_student_step110000.pt",
        help="Path to student checkpoint .pt.")
    parser.add_argument("--n", type=int, default=32, help="Number of samples to invert (and thus number of trajectories).")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps for both inversion and forward sampling.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0.0 for deterministic).")
    parser.add_argument("--frames", type=bool, default=True, help="Save per-timestep frames (norm/denorm).")
    parser.add_argument("--gif", type=bool, default=True, help="Make GIFs (norm/denorm).")
    parser.add_argument("--out", type=str, default="vis_traj_inverted_fd", help="Output directory root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # required for INVERSION
    parser.add_argument("--student-data", type=str, default="smile_data_n32_scale2_rot60_trans_50_-20/train.npy", help="Path to student-domain x0 dataset (npy or csv).")
    parser.add_argument("--data-format", type=str, default="npy", choices=["npy","csv"], help="Dataset file format.")
    parser.add_argument("--student-stats", type=str,
                        default="smile_data_n32_scale2_rot60_trans_50_-20/normalization_stats.json",
                        help="JSON with {'mean':[...],'std':[...]} for normalization/denorm.")

    # Pure score field 옵션
    parser.add_argument("--score-ts", type=str, default="", help="Comma-separated raw t's for score field (e.g., '999,750,500,250,0'). Empty -> auto-pick 5.")
    parser.add_argument("--score-grid", type=int, default=80, help="Grid resolution for score field.")
    parser.add_argument("--score-density", type=float, default=1.2, help="Streamplot density.")
    return parser.parse_args()

def main():
    cfg = CONFIG
    args = parse_args()
    set_seed(args.seed if args.seed is not None else cfg["seed"])

    ckpt_path = args.ckpt
    if ckpt_path in (None, "", "auto"):
        auto = find_latest_student_ckpt(Path(cfg["out_dir"]), cfg["auto_find_ckpt_pattern"])
        if auto is None:
            raise FileNotFoundError("No --ckpt provided and no checkpoint found in out_dir.")
        ckpt_path = auto
        print(f"[AUTO] Using latest checkpoint: {ckpt_path}")

    visualize_student_ddim_trajectories(
        cfg=cfg,
        student_ckpt_path=ckpt_path,
        n_noises=int(args.n),
        ddim_steps=int(args.steps),
        eta=float(args.eta),
        save_frames=bool(args.frames),
        make_gif_flag=bool(args.gif),
        out_dir_override=args.out,
        student_stats_path=args.student_stats,
        score_ts=args.score_ts,
        score_grid=int(args.score_grid),
        score_density=float(args.score_density),
        student_data_path=args.student_data,
        data_format=args.data_format,
    )

if __name__ == "__main__":
    main()
