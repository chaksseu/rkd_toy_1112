#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import json, math, random
from pathlib import Path
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Dataset, DataLoader
import itertools
# ===================== CONFIG ===================== #


WANDB_NAME=f"1014_rkd_W0.1_xt_b512_ddim_0.5steps_adft_full_init_H_with_diff_W1.0"


CONFIG = {
    # I/O
    "device": f"cuda:0",
    "out_dir": f"runs/{WANDB_NAME}",
    # teacher / student
    "teacher_ckpt": "ckpt_teahcer8192_step310000.pt",  # REQUIRED
    # "student_init_ckpt": "",                     
    "student_init_ckpt": "ckpt_student_H_1e5_1010_step090000.pt",                     
    # "student_init_ckpt": "ckpt_student8_step310000.pt",                     
    "resume_student_ckpt": f"",        
    "teacher_data_stats": "smile_data_n8192_scale10_rot0_trans_0_0/teacher_normalization_stats.json",

    # diffusion loss 가중치
    "W_DIFF": 1.0,                               # ε-pred MSE 가중치
    "W_RKD": 0.1,
    "rkd_ddim_steps_to_t": 50,   # t_sel까지 최대 몇 번의 DDIM 전이만 사용할지

    "batch_size": 512,
    "num_noises": 8192, 
    "epochs_total": 5000000,          # 총 스텝 수 (기존 epochs_per_stage 대신 사용)

    "noise_pool_file": None,        # None이면 out_dir/data/noises_pool.npy 로 저장
    "regen_noise_pool": False,      # True면 항상 새로 만듦
    
    # schedule / time
    "T": 50,                 # total diffusion steps (timesteps = 0..T-1)

    "seed": 42,
    # RKD weights
    "W_NORM": 0.0,
    "use_mu_normalization": True,
    # noises / data dims
    "dim": 2,                 # 2D toy

    # model sizes
    "teacher_hidden": 256, "teacher_depth": 8, "teacher_time_dim": 64,
    "student_hidden": 256, "student_depth": 8, "student_time_dim": 64,
    # optim
    "lr": 1e-5, "weight_decay": 0.0, "max_grad_norm": 1.0,
    # sampling viz
    "vis_interval_epochs": 10000,
    "n_vis": 8192,       # 경로를 수집/표시할 noise 개수
    "ddim_steps": 50,
    "ddim_eta": 0.0,
    # wandb
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-AICA-1014",
    "wandb_run_name": WANDB_NAME,

    "use_learnable_H": True,
    "H_init": [1,0,0,  0,1,0,  0,0,1],     # 초기값(I)
    "H_eps": 1e-6,                        # w 분모 안정화
    "resume_H_ckpt": "",                  # (옵션) H만 재개 로드 경로
}

CONFIG.update({
    # student 데이터 경로/형식
    "student_data_path": "smile_data_n8192_scale10_rot0_trans_0_0_H_32_-13_100_55_8_200_0.05_0.005_1.2_n16/train.npy",   # 혹은 .csv
    # "student_data_path": "smile_data_n8192_scale10_rot0_trans_0_0/train.npy",   # 혹은 .csv
    "student_data_format": "npy",                # "npy" | "csv"
    "student_dataset_batch_size": 16,          # 없으면 batch_size 사용
})

# ===================== UTILS ===================== #
import re

from pathlib import Path

def ensure_noise_pool(cfg) -> Path:
    """노이즈 풀(.npy) 보장 생성 후 경로 반환"""
    out_dir = Path(cfg["out_dir"])
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(cfg["noise_pool_file"] or (data_dir / "noises_pool.npy"))
    if cfg.get("regen_noise_pool", False) or (not path.exists()):
        N, D = cfg["num_noises"], cfg["dim"]
        z = np.random.randn(N, D).astype("float32")
        np.save(path, z)
        print(f"[NOISE] generated pool: {path}  shape={(N,D)}")
    else:
        print(f"[NOISE] using existing pool: {path}")
    return path

def sample_noise_batch(noise_pool: np.ndarray, B: int, device: torch.device) -> torch.Tensor:
    """노이즈 풀에서 인덱스 랜덤 샘플로 배치 구성"""
    N = noise_pool.shape[0]
    idx = np.random.randint(0, N, size=B)
    z = torch.from_numpy(noise_pool[idx]).to(device=device, dtype=torch.float32)
    return z


def make_ddim_like(train_sched, device, T: int):
    """
    train_sched.config(베타/타임스텝 등)를 복사해 DDIM 스케줄러 생성.
    timesteps = [T-1, ..., 0] 로 설정.
    """
    ddim = DDIMScheduler.from_config(train_sched.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = "epsilon"
    ddim.set_timesteps(T, device=device)  # [T-1, ..., 0]
    return ddim




def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_norm_stats(json_path: str):
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)           # {"mean": [...], "std": [...]}
    mu = np.array(d["mean"],  dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma

def save_and_log_xt_pairs_for_all_t(
    left_seq: np.ndarray,   # [K,B,2] Teacher
    right_seq: np.ndarray,  # [K,B,2] Student(or Student+H)
    ts: np.ndarray,         # [K]
    noise_ids: np.ndarray,  # [B]
    out_dir: Path,
    step_i: int,
    sync_axes: bool = False,
    dot_size: int = 6,
    use_wandb: bool = False,
    subdir_name: str = "xt_pairs",      # ← 추가: 저장 폴더/로깅 태그 분리용
):
    base = out_dir / "figs" / f"{subdir_name}_step{step_i:06d}"
    base.mkdir(parents=True, exist_ok=True)

    logged = {}
    for k, t in enumerate(ts):
        left_xy  = left_seq[k]   # (B,2)
        right_xy = right_seq[k]  # (B,2)
        path = base / f"t{int(t):03d}.png"

        plot_pair_scatter_colored(
            left_xy=left_xy,
            right_xy=right_xy,
            noise_ids=noise_ids,
            N_total=len(noise_ids),
            left_title = fr"Teacher $x_t$ (t={int(t)})",
            right_title= fr"Student $x_t$ (t={int(t)})",
            out_path=path,
            dot_size=dot_size,
            sync_axes=sync_axes,
        )

        if use_wandb:
            # ← 태그도 분리해서 W&B에서 raw/H가 따로 보이게
            logged[f"img/{subdir_name}/t{int(t):03d}"] = wandb.Image(str(path))

    if use_wandb and logged:
        wandb.log(logged, step=step_i)

    return base



def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    """
    데이터의 x/y 범위를 보고, 긴 변 기준(span)으로 여백을 준 뒤
    중앙 정렬된 정사각형 xlim/ylim을 반환.
    """
    data = np.asarray(data)
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin

    # 긴 변 + 최소 폭 보호
    base = max(dx, dy, 1e-3)
    pad  = pad_ratio * base

    # 여백 적용
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    # 중앙 정렬된 정사각형으로 확장
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0

    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def colors_from_noise_ids(ids: np.ndarray, N_total: int, alpha: float = 0.85):
    """
    같은 'global noise id'에는 항상 같은 색이 나오도록 결정적 매핑.
    - hue = (id * φ) mod 1.0  (φ ≈ 0.618…)
    - colormap: hsv (연속)
    """
    ids = np.asarray(ids, dtype=np.int64)
    phi = 0.6180339887498949
    hues = (ids * phi) % 1.0
    cmap = plt.get_cmap("hsv")
    cols = cmap(hues)
    cols[:, 3] = alpha  # alpha
    return cols

def plot_pair_scatter_colored(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    noise_ids: np.ndarray,
    N_total: int,
    left_title: str,
    right_title: str,
    out_path: Path,
    dot_size: int = 8,
    sync_axes: bool = False,
    pad_ratio: float = 0.05,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(left_xy); B = np.asarray(right_xy)
    assert A.shape == B.shape and A.shape[1] == 2, "Expect (B,2) arrays for both panels."

    colors = colors_from_noise_ids(np.asarray(noise_ids), N_total)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             sharex=sync_axes, sharey=sync_axes)

    # 공통 정사각형 축 범위 (sync_axes=True)
    if sync_axes:
        both = np.vstack([A, B])
        xlim_all, ylim_all = _square_limits_from(both, pad_ratio=pad_ratio)

    for ax, data, title in ((axes[0], A, left_title), (axes[1], B, right_title)):
        ax.scatter(data[:, 0], data[:, 1], s=dot_size, c=colors, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

        if sync_axes:
            ax.set_xlim(*xlim_all); ax.set_ylim(*ylim_all)
        else:
            xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_scatter_colored(
    left_xy: np.ndarray,      # Teacher (B,2)
    mid_xy: np.ndarray,       # Student raw (B,2)
    right_xy: np.ndarray,     # Student + H(t) (B,2)
    noise_ids: np.ndarray,
    N_total: int,
    out_path: Path,
    titles: Tuple[str, str, str] = (r"Teacher $x_t$", r"Student $x_t$ (raw)", r"Student $x_t$ + H(t)"),
    dot_size: int = 6,
    sync_axes: bool = True,
    pad_ratio: float = 0.05,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(left_xy); B = np.asarray(mid_xy); C = np.asarray(right_xy)
    assert A.shape == B.shape == C.shape and A.shape[1] == 2, "Expect three (B,2) arrays."

    colors = colors_from_noise_ids(np.asarray(noise_ids), N_total)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=sync_axes, sharey=sync_axes)

    if sync_axes:
        all_pts = np.vstack([A, B, C])
        xlim_all, ylim_all = _square_limits_from(all_pts, pad_ratio=pad_ratio)

    for ax, data, title in zip(axes, (A, B, C), titles):
        ax.scatter(data[:, 0], data[:, 1], s=dot_size, c=colors, edgecolors="none")
        ax.set_aspect("equal", adjustable="box"); ax.set_title(title)
        if sync_axes:
            ax.set_xlim(*xlim_all); ax.set_ylim(*ylim_all)
        else:
            xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_and_log_xt_triplets_for_all_t(
    teacher_seq: np.ndarray,   # [K,B,2]
    student_seq: np.ndarray,   # [K,B,2]
    studentH_seq: np.ndarray,  # [K,B,2]
    ts: np.ndarray,            # [K]
    noise_ids: np.ndarray,     # [B]
    out_dir: Path,
    step_i: int,
    use_wandb: bool = False,
    subdir_name: str = "xt_triplets",
    dot_size: int = 6,
    sync_axes: bool = True,
):
    base = out_dir / "figs" / f"{subdir_name}_step{step_i:06d}"
    base.mkdir(parents=True, exist_ok=True)

    logged = {}
    for k, t in enumerate(ts):
        path = base / f"t{int(t):03d}.png"
        plot_triplet_scatter_colored(
            left_xy  = teacher_seq[k],
            mid_xy   = student_seq[k],
            right_xy = studentH_seq[k],
            noise_ids=noise_ids,
            N_total=len(noise_ids),
            out_path=path,
            titles=(fr"Teacher $x_t$ (t={int(t)})", fr"Student $x_t$ (raw)", fr"Student $x_t$ + H(t)"),
            dot_size=dot_size,
            sync_axes=sync_axes,
        )
        if use_wandb:
            logged[f"img/{subdir_name}/t{int(t):03d}"] = wandb.Image(str(path))

    if use_wandb and logged:
        wandb.log(logged, step=step_i)

    return base



@torch.no_grad()
def collect_xt_seq_ddim(
    model: nn.Module,
    ddim: DDIMScheduler,
    z: torch.Tensor,
    t_stop: int = 0,
    return_ts: bool = True,
):
    """
    같은 z(x_{T-1})에서 시작해 DDIM(eta=0)으로 t_stop까지 한 스텝씩 내려가며
    각 시점의 x_t를 모두 수집.
    return:
      - seq: [K, B, 2] (K = 수집된 timestep 수, T-1→...→t_stop 순서)
      - ts : [K]       (정수 timesteps)
    """
    x = z.clone()
    B = x.size(0)
    xs, ts = [], []
    for t_tensor in ddim.timesteps:      # [T-1, ..., 0]
        t_int = int(t_tensor)
        xs.append(x.detach().cpu().numpy())  # 현재 x가 곧 x_t
        ts.append(t_int)
        if t_int == t_stop:
            break
        t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
        x_in = ddim.scale_model_input(x, t_tensor)
        eps  = model(x_in, t_b)
        x    = ddim.step(model_output=eps, timestep=t_tensor, sample=x, eta=0.0).prev_sample

    seq = np.stack(xs, axis=0)  # [K, B, 2]
    return (seq, np.asarray(ts, dtype=int)) if return_ts else seq

@torch.no_grad()
def apply_H_to_seq_per_t(seq_S: np.ndarray, ts: np.ndarray, H_module: nn.Module, device: torch.device) -> np.ndarray:
    """
    seq_S: [K,B,2] (numpy, normalized)
    ts   : [K]     (int timesteps)
    return: seq_S_H [K,B,2] with H(t) applied per timestep
    """
    K, B, _ = seq_S.shape
    outs = []
    for k in range(K):
        t_k = int(ts[k])
        xk  = torch.from_numpy(seq_S[k]).to(device=device, dtype=torch.float32)   # (B,2)
        xk_H, _ = H_module(xk, t_k)                                              # (B,2)
        outs.append(xk_H.detach().cpu().numpy())
    return np.stack(outs, axis=0)




def make_grad_hook(coef):
    return lambda x: coef * x

def sample_ddim_student(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    num_train_timesteps = int(local.config.num_train_timesteps)

    sample_steps = int(sample_steps or num_train_timesteps)

    local.set_timesteps(sample_steps, device=device)

    max_t = int(local.timesteps[0])

    grad_coefs = []
    for i, t in enumerate(local.timesteps):
        grad_coefs.append(local.alphas_cumprod[t].sqrt().item() * (1-local.alphas_cumprod[t]).sqrt().item() / (1-local.alphas[t].item()))
    grad_coefs = np.array(grad_coefs)
    grad_coefs /= (math.prod(grad_coefs)**(1/len(grad_coefs)))

    model.train()
    for i, t in enumerate(local.timesteps):
        t_int = int(t)

        if (t_int != max_t) and ((t_int % 2) != (t_sel % 2)):
            continue

        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)

        x_in = local.scale_model_input(x.detach(), t)
        # x_in = local.scale_model_input(x, t)

        eps = model(x_in, t_b)
        
        hook_fn = make_grad_hook(grad_coefs[i])
        eps.register_hook(hook_fn)
       
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        if t_int <= t_sel:
            break
    return x


@torch.no_grad()
def sample_ddim_teacher(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    num_train_timesteps = int(local.config.num_train_timesteps)

    sample_steps = int(sample_steps or num_train_timesteps)

    local.set_timesteps(sample_steps, device=device)

    model.eval()

    max_t = int(local.timesteps[0])

    for t in local.timesteps:  # [T-1, ..., 0] 순서
        t_int = int(t)

        if (t_int != max_t) and ((t_int % 2) != (t_sel % 2)):
            continue
        
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        if t_int <= t_sel:
            break
    return x.detach()




# ===================== MODEL ===================== #

class LearnableHomography(nn.Module):
    """
    Row-vector convention: [x, y, 1] @ H^T -> [X, Y, W], (x',y')=(X/W, Y/W)
    H는 각 timestep t마다 다른 3x3 행렬을 학습합니다: shape [T, 3, 3]
    """
    def __init__(self, init_9=None, eps: float = 1e-6, T: int = 50, fix_last_row: bool = False):
        super().__init__()
        self.T = int(T)
        self.eps = float(eps)
        self.fix_last_row = bool(fix_last_row)  # True면 마지막 행을 [0,0,1]로 고정(affine)

        if init_9 is None:
            I = torch.eye(3, dtype=torch.float32)         # (3,3)
        else:
            I = torch.tensor(init_9, dtype=torch.float32).view(3,3)

        # [T,3,3]로 초기화 (모든 t에서 I로 시작)
        H0 = I.unsqueeze(0).repeat(self.T, 1, 1)          # (T,3,3)
        self.H = nn.Parameter(H0)                          # learnable

    def _get_Ht(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape () | (B,) long
        return: H_t of shape (B,3,3) if t is vector, else (1,3,3)
        """
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=self.H.device)
        elif torch.is_tensor(t) and t.ndim == 0:
            t = t.view(1)
        # index
        Ht = self.H.index_select(0, t.clamp(min=0, max=self.T-1))  # (B,3,3)
        if self.fix_last_row:
            # 마지막 행을 [0,0,1]로 고정 (affine 제약)
            Ht = Ht.clone()
            Ht[..., 2, :2] = 0.0
            Ht[..., 2, 2]  = 1.0
        return Ht

    def forward(self, xy: torch.Tensor, t) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xy: (B,2), t: int | () | (B,) long
        returns: (xy_trans: (B,2), w: (B,1))
        """
        B = xy.shape[0]
        device = xy.device
        ones = torch.ones(B, 1, device=device, dtype=xy.dtype)
        homo = torch.cat([xy, ones], dim=-1)              # (B,3)

        if not torch.is_tensor(t):                        # python int
            t = torch.full((B,), int(t), device=device, dtype=torch.long)
        elif t.ndim == 0:                                 # scalar tensor
            t = t.expand(B)

        Ht = self._get_Ht(t)                               # (B,3,3)
        out = torch.bmm(homo.unsqueeze(1), Ht.transpose(1,2)).squeeze(1)  # (B,3)
        w   = out[:, 2:3]
        den = w.sign() * torch.clamp(w.abs(), min=self.eps)
        xy_t = out[:, :2] / den
        return xy_t, w





class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb

class MLPDenoiser(nn.Module):
    """ ε-predictor for 2D toy; used for both Teacher and Student. """
    def __init__(self, in_dim=2, time_dim=64, hidden=128, depth=3, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t); h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x, t

# ===================== SCHEDULERS ===================== #

def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"  # DDIM도 동일하게

    return train_sched, sample_sched


# ===================== PREPARE (SAVE TEACHER ε & x) ===================== #

def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return arr * sigma + mu

def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (arr - mu) / sigma

# ===================== Dataset ===================== #

class StudentX0Dataset(Dataset):
    def __init__(self, path: str, fmt: str, mu: np.ndarray, sigma: np.ndarray):
        self.X = self._load(path, fmt)  # (N,2) or (N,D)
        assert self.X.ndim == 2 and self.X.shape[1] >= 2, "Expect (N,2) or (N,D)"
        self.X = self.X[:, :2].astype(np.float32)
        self.X = normalize_np(self.X, mu, sigma).astype(np.float32)  # 모델 입력 스케일로
    def _load(self, path, fmt):
        p = Path(path)
        if fmt == "npy":
            return np.load(p)
        elif fmt == "csv":
            return np.loadtxt(p, delimiter=",")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i]  # x0 (normalized)

def build_student_dataloader(cfg, mu, sigma):
    bs = int(cfg.get("student_dataset_batch_size", cfg["batch_size"]))
    ds = StudentX0Dataset(cfg["student_data_path"], cfg["student_data_format"], mu, sigma)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

# ===================== RKD on epsilon ===================== #

def loss_rkd_xt(
    xt_s: torch.Tensor,      # [B, D]
    xt_s_2: torch.Tensor,      # [B, D]
    xt_t: torch.Tensor,      # [B, D]
    xt_t_2: torch.Tensor,      # [B, D]
    use_mu_norm: bool = True,
    w_rkd: float = 1.0,
    w_norm: float = 0.0,
    eps: float = 1e-12,
):
    B = xt_s.size(0)
    if B < 2:
        z = xt_s.new_zeros(())
        return {"total": z, "rkd": z, "norm": z, "t_norm": z, "s_norm": z}

    # pairwise distances on xt vectors
    t_full = torch.cdist(xt_t.detach(), xt_t_2.detach(), p=2)
    s_full = torch.cdist(xt_s,           xt_s_2,           p=2)
    iu = torch.triu_indices(B, B, offset=1, device=xt_s.device)
    t_d = t_full[iu].clamp_min(eps); s_d = s_full[iu].clamp_min(eps)

    if use_mu_norm:
        t_d = t_d / t_d.mean().clamp_min(eps)
        s_d = s_d / s_d.mean().clamp_min(eps)
    
    loss_rkd = w_rkd * F.mse_loss(s_d, t_d, reduction="mean")

    total = loss_rkd
    return {"total": total, "rkd": loss_rkd}


# ===================== Training ===================== #

def train_student_uniform_xt(cfg: Dict):
    """
    - 노이즈 풀에서 배치 샘플
    - t ~ Uniform{0..T-1}, 배치 내 t 동일
    - 같은 z로 Teacher/Student를 각각 DDIM 역진행
      * Teacher: eval + no_grad → x_t^T
      * Student: train + grad ON (전체 경로) → x_t^S
    - RKD(x_t^S, x_t^T)로 학습
    """
    out_dir = Path(cfg["out_dir"]); (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 스케줄러/모델
    train_sched, sample_sched = build_schedulers(cfg["T"])
    ddim = make_ddim_like(train_sched, device, cfg["T"])

    teacher = MLPDenoiser(2, cfg["teacher_time_dim"], cfg["teacher_hidden"], cfg["teacher_depth"], 2).to(device)
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], map_location=device), strict=True)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    student = MLPDenoiser(2, cfg["student_time_dim"], cfg["student_hidden"], cfg["student_depth"], 2).to(device)
    # init / resume
    if cfg.get("resume_student_ckpt"):
        p = Path(cfg["resume_student_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[RESUME] Loaded student:", p)
    elif cfg.get("student_init_ckpt"):
        p = Path(cfg["student_init_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[INIT] Loaded student init:", p)
        else:
            print("[INIT] Student from scratch")


    # === Learnable H ===
    H_module = LearnableHomography(init_9=cfg["H_init"], eps=cfg["H_eps"], T=cfg["T"]).to(device)

    if cfg.get("resume_H_ckpt"):
        pH = Path(cfg["resume_H_ckpt"])
        if pH.exists():
            H_module.load_state_dict(torch.load(pH, map_location=device), strict=True)
            print("[RESUME] Loaded H:", pH)

    if not cfg.get("use_learnable_H", True):
        for p in H_module.parameters():
            p.requires_grad = False

    # opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # === Optimizer ===
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(H_module.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # W&B
    if cfg["use_wandb"]:
        wandb.login()
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)
        wandb.define_metric("step")
        wandb.define_metric("loss/*", step_metric="step")
        wandb.define_metric("loss_by_t/*", step_metric="step")
        # wandb.define_metric("loss_by_tp1_t/*", step_metric="step")

    # 노이즈 풀
    noise_path = ensure_noise_pool(cfg)
    noise_pool = np.load(noise_path)  # (N,D), 작으면 통째 로드 OK

    mu_teacher, sigma_teacher = load_norm_stats(cfg["teacher_data_stats"])


    # --- Student 도메인 dataloader (diffusion loss용) ---
    student_loader = build_student_dataloader(cfg, mu_teacher, sigma_teacher)
    student_iter = iter(student_loader)

    def next_student_batch():
        nonlocal student_iter
        try:
            x0 = next(student_iter)  # (B_s, 2) torch.Tensor로 들어옴
        except StopIteration:
            student_iter = iter(student_loader)
            x0 = next(student_iter)
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0, dtype=torch.float32)
        return x0.to(device, non_blocking=True)




    T = int(cfg["T"])
    total_steps = int(cfg.get("epochs_total", 50000))
    B = int(cfg["batch_size"])

    for step_i in range(1, total_steps + 1):
        # 1) t ~ Uniform{0..T-1} (배치 내 동일)
        t_sel = int(np.random.randint(0, T-1))
        # t_sel = torch.full((B,), t_sel, device=device, dtype=torch.long)      
        # t_sel = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
        
        # z 샘플
        # 동일 노이즈 안에서만 사용하므로 오버피팅 역할
        z = sample_noise_batch(noise_pool, B, device)

        # random noise
        # z = torch.randn_like(z)


        ddim_steps = int(cfg.get("rkd_ddim_steps_to_t", 5))
        # ddim_steps = int(np.random.randint(18, 23))
        # ddim_steps = max(round(t_sel * 2 / 5), 1)


        with torch.no_grad():
            xt_T = sample_ddim_teacher(
                model=teacher, sample_scheduler=ddim, z=z, 
                device=device, sample_steps=ddim_steps,
                eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
            )

        student.train()
        xt_S = sample_ddim_student(
            model=student, sample_scheduler=ddim, z=z, 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
        )

        
        xt_S, w_S = H_module(xt_S, t_sel)   

        parts_same_t = loss_rkd_xt(
            xt_S, xt_S, xt_T, xt_T,
            use_mu_norm=cfg["use_mu_normalization"],
            w_rkd=cfg["W_RKD"], w_norm=cfg["W_NORM"]
        )
        same_t_loss = parts_same_t["total"]


        # ===================== NEW: diffusion ε-MSE loss =====================
        # 학생 도메인 x0 배치 가져오기 (정규화 완료 상태)
        x0_batch = next_student_batch()  # shape (B_s, 2)

        # RKD에서 사용한 것과 동일한 t를 공유(원하면 독립적으로 뽑아도 OK)
        t_b_s = torch.full((x0_batch.shape[0],), t_sel, device=device, dtype=torch.long)

        # 표준 훈련 루틴: ε 샘플 → x_t 생성 → ε̂ 예측 → MSE
        # eps = torch.randn_like(x0_batch)
        eps = sample_noise_batch(noise_pool, x0_batch.shape[0], device)

        x_t_for_diff = train_sched.add_noise(x0_batch, eps, t_b_s)  # q(x_t|x0, ε, t)

        eps_pred = student(x_t_for_diff, t_b_s)  # prediction_type='epsilon'
        diff_loss = cfg["W_DIFF"] * F.mse_loss(eps_pred, eps, reduction="mean")
        # # ===============================
        
        # diff_loss = torch.tensor(0.0)

        loss = same_t_loss + diff_loss

        opt.zero_grad()
        loss.backward()
        if cfg.get("max_grad_norm", 0) > 0:
            nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(H_module.parameters()),
                cfg["max_grad_norm"]
            )
        opt.step()

        if (step_i % max(1, total_steps // 20) == 0) or (step_i == 1):
            print(f"[step {step_i:06d}] rkd={same_t_loss.item():.6f}  diff={diff_loss.item():.6f}  total={loss.item():.6f}")



        if cfg["use_wandb"]:
            logdict = {
                "step": step_i,
                "loss/total": float(loss),
                "loss/rkd": float(same_t_loss),
                "loss/diff": float(diff_loss),
                f"loss_by_t/rkd/t{t_sel:02d}": float(same_t_loss),
                "lr": opt.param_groups[0]["lr"],
            }

            wandb.log(logdict, step=step_i)




        # 7) (옵션) 시각화: 그대로 유지 (원 코드와 동일)
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            with torch.no_grad():
                B_plot = min(int(cfg.get("n_vis", 1024)), B)
                z_vis  = sample_noise_batch(noise_pool, B_plot, device)

                seq_T, ts = collect_xt_seq_ddim(teacher, ddim, z_vis, t_stop=0, return_ts=True)
                seq_S, _  = collect_xt_seq_ddim(student.eval(), ddim, z_vis, t_stop=0, return_ts=True)
                seq_S_H   = apply_H_to_seq_per_t(seq_S, ts, H_module, device)


                # stride
                stride = max(1, int(cfg.get("vis_xt_stride", 1)))
                idxs = np.arange(0, len(ts), stride)
                seq_T_s   = seq_T[idxs]
                seq_S_s   = seq_S[idxs]
                seq_S_H_s = seq_S_H[idxs]
                ts_s      = ts[idxs]

                # denorm
                seq_T_s_plot   = denormalize_np(seq_T_s,   mu_teacher, sigma_teacher)
                seq_S_s_plot   = denormalize_np(seq_S_s,   mu_teacher, sigma_teacher)
                seq_S_H_s_plot = denormalize_np(seq_S_H_s, mu_teacher, sigma_teacher)
                
                _ = save_and_log_xt_triplets_for_all_t(
                    teacher_seq = seq_T_s_plot,
                    student_seq = seq_S_s_plot,
                    studentH_seq= seq_S_H_s_plot,
                    ts          = ts_s,
                    noise_ids   = np.arange(B_plot),
                    out_dir     = out_dir,
                    step_i      = step_i,
                    use_wandb   = bool(cfg["use_wandb"]),
                    subdir_name = "xt_triplets",        # 디렉토리/로그 키 접두사
                    dot_size    = 6,
                    sync_axes   = bool(cfg.get("vis_xt_sync_axes", False)),
                )


        # 7) (옵션) 시각화: 랜덤 노이즈에서 학생 모델로 x0 샘플링하여 저장
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            @torch.no_grad()
            def sample_x0_ddim(model, sample_scheduler, num_samples, device, sample_steps, dim=2, eta=0.0):
                local = DDIMScheduler.from_config(sample_scheduler.config)
                local.set_timesteps(int(sample_steps), device=device)
                x = torch.randn(num_samples, dim, device=device)
                for t in local.timesteps:  # [T-1 ... 0]
                    t_b  = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
                    x_in = local.scale_model_input(x, t)
                    eps  = model(x_in, t_b)
                    x    = local.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
                return x

            student.eval()

            B_plot = 8192
            x0_s = sample_x0_ddim(
                model=student,
                sample_scheduler=ddim,
                num_samples=B_plot,
                device=device,
                sample_steps=int(cfg["T"]),
                dim=int(cfg.get("dim", 2)),
                eta=float(cfg.get("ddim_eta", 0.0)),
            )

            figs_dir = Path(cfg["out_dir"]) / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)

            # (1) 원본 x0 (raw)
            x0_s_plot = denormalize_np(x0_s.detach().cpu().numpy(), mu_teacher, sigma_teacher)
            png_path  = figs_dir / f"samples_step{step_i:06d}.png"
            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_plot[:, 0], x0_s_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca(); ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) RAW @ step {step_i}")
            plt.tight_layout(); plt.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img/student_samples/raw": wandb.Image(str(png_path))}, step=step_i)

            # (2) H(t=0) 적용본
            with torch.no_grad():
                x0_s_H, _ = H_module(x0_s, t=torch.zeros(x0_s.shape[0], device=x0_s.device, dtype=torch.long))
            x0_s_H_plot = denormalize_np(x0_s_H.detach().cpu().numpy(), mu_teacher, sigma_teacher)
            png_path_H   = figs_dir / f"samples_H_t0_step{step_i:06d}.png"

            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_H_plot[:, 0], x0_s_H_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca(); ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_H_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) with H(t=0) @ step {step_i}")
            plt.tight_layout(); plt.savefig(png_path_H, dpi=150, bbox_inches="tight"); plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img/student_samples/raw+H": wandb.Image(str(png_path_H))}, step=step_i)




                # --- H matrices full HTML viz over ALL timesteps (paged) ---
        if cfg.get("use_wandb", False) and ((step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps)):
            import html
            with torch.no_grad():
                eps = 1e-12
                rows_per_panel = int(cfg.get("H_vis_rows_per_panel", 20))  # 페이지당 행 수

                # [T,3,3] -> CPU float
                H_all: torch.Tensor = H_module.H.detach().float().cpu()
                Ttot = int(H_all.shape[0])

                def mat_to_pre(M: torch.Tensor) -> str:
                    arr = M.numpy()
                    s = np.array2string(
                        arr,
                        formatter={'float_kind': lambda x: f"{x: .5f}"},
                        max_line_width=200
                    )
                    return f"<pre style='margin:0'>{html.escape(s)}</pre>"

                # t=0..T-1 전부 행 생성
                all_rows = []
                for t in range(Ttot):
                    Ht = H_all[t]                                  # (3,3)
                    Ht_norm = Ht / max(float(Ht[2,2].abs()), eps)   # proj scale 제거

                    # 역행렬(가능하면)
                    try:
                        Ht_inv = torch.linalg.inv(Ht)
                        Ht_inv_norm = Ht_inv / max(float(Ht_inv[2,2].abs()), eps)
                        inv_html  = mat_to_pre(Ht_inv)
                        invn_html = mat_to_pre(Ht_inv_norm)
                    except Exception:
                        inv_html  = "<pre style='margin:0'>(singular)</pre>"
                        invn_html = "<pre style='margin:0'>(singular)</pre>"

                    row = f"""
                    <tr>
                      <td style="text-align:center;">t = {t}</td>
                      <td>{mat_to_pre(Ht)}</td>
                      <td>{mat_to_pre(Ht_norm)}</td>
                      <td>{inv_html}</td>
                      <td>{invn_html}</td>
                    </tr>
                    """
                    all_rows.append(row)

                # 페이지로 나눠서 여러 패널로 로깅
                num_pages = (len(all_rows) + rows_per_panel - 1) // rows_per_panel
                for p in range(num_pages):
                    chunk = all_rows[p*rows_per_panel:(p+1)*rows_per_panel]
                    html_block = f"""
                    <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px;">
                      <div style="margin-bottom:6px;">H(t) page {p+1}/{num_pages} (rows {p*rows_per_panel}..{min((p+1)*rows_per_panel, len(all_rows))-1})</div>
                      <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;">
                        <thead>
                          <tr>
                            <th>timestep</th>
                            <th>H(t)</th>
                            <th>H(t) / H(t)[2,2]</th>
                            <th>H(t)<sup>-1</sup></th>
                            <th>H(t)<sup>-1</sup> / [2,2]</th>
                          </tr>
                        </thead>
                        <tbody>
                          {''.join(chunk)}
                        </tbody>
                      </table>
                    </div>
                    """.strip()
                    wandb.log({f"H_vis/table_p{p:02d}": wandb.Html(html_block)}, step=step_i)

                # (선택) 간단 통계도 함께
                dets, conds = [], []
                for t in range(Ttot):
                    Ht = H_all[t]
                    dets.append(torch.det(Ht).item())
                    try:
                        conds.append(torch.linalg.cond(Ht).item())
                    except Exception:
                        conds.append(float('nan'))
                wandb.log({
                    "H_stats/det_mean": float(np.nanmean(dets)),
                    "H_stats/cond_mean": float(np.nanmean(conds)),
                }, step=step_i)





        # 8) (옵션) 주기적 체크포인트
        if (step_i % (cfg["vis_interval_epochs"]) == 0) or (step_i == total_steps):
            ckpt_path = out_dir / f"ckpt_student_step{step_i:06d}.pt"
            torch.save(student.state_dict(), ckpt_path)
            torch.save(H_module.state_dict(), out_dir / f"ckpt_H_step{step_i:06d}.pt")   # ← 추가
            print("[CKPT]", ckpt_path)

    print("\n[DONE] Out dir:", out_dir.resolve())
    if cfg["use_wandb"]:
        wandb.finish()


# ===================== MAIN ===================== #

def main(cfg: Dict):
    set_seed(cfg["seed"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    train_student_uniform_xt(cfg)

if __name__ == "__main__":
    main(CONFIG)
