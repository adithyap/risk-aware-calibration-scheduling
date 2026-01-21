"""
Intelligent calibration scheduling on CMAPSS turbofan data, adapted for Colab.

This single file:
- Downloads and caches CMAPSS.
- Adapts the data using the calibration recipe (drift sensors, virtual thresholds,
  simulated calibration resets via splicing/stitching, and time-to-drift labels).
- Trains baseline ML regressors and a sequence LSTM, with dynamic batch sizing.
- Produces plots, tables, and a textual summary suitable for paper-ready artifacts.
- Builds a simple scheduling recommendation (ranking by predicted urgency).
- Bundles outputs for easy download in Colab.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import shutil
import tarfile
import zipfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    base_dir: Path = Path("calibration_outputs")  # works locally and in Colab
    invalidate_cache: bool = False
    dataset_urls: List[str] = field(
        default_factory=lambda: [
            # Primary (commonly used mirror)
            "https://s3.amazonaws.com/nasa-cmapss/CMAPSSData.zip",
            # Fallback NASA data portal export
            "https://data.nasa.gov/download/6vrx-r4m3/application%2Fzip",
        ]
    )
    kaggle_slug: str = "palbha/cmapss-jet-engine-simulated-data"
    prefer_kaggle: bool = True
    dataset_name: str = "FD001"
    dataset_names: List[str] = field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"])
    use_small_run: bool = False  # default to full run; override to True for quick sanity checks
    engine_limit: int = 15  # limit number of engines when small run is enabled
    window_size: int = 40
    window_stride: int = 1
    threshold_ratio_min: float = 0.55  # per-sensor threshold range
    threshold_ratio_max: float = 0.8
    baseline_window: int = 20
    post_drift_scale: float = 1.1  # allow re-crossing thresholds
    reset_noise: float = 0.03
    calibration_repeats: int = 3  # number of synthetic calibration cycles per engine
    max_resets_per_run: int = 3  # number of calibration events per synthetic run
    reset_mode: str = "splice"  # "splice" or "stitch"
    stitch_length: int = 40  # cycles used from donor for stitching
    train_fraction: float = 0.75
    seeds: List[int] = field(default_factory=lambda: [42])
    seed: int = 42
    safety_margin: int = 5  # predicted TTD margin for scheduling
    fixed_interval: int = 30  # cycles for fixed policy
    calib_cost: float = 1.0
    violation_cost: float = 5.0
    batch_candidates: List[int] = field(
        default_factory=lambda: [2048, 1024, 512, 256, 192, 128, 96, 64, 48, 32]
    )
    epochs: int = 30
    early_stop_patience: int = 3
    lr: float = 1e-3
    transformer_lr: float = 3e-4
    transformer_epochs: int = 40
    transformer_weight_decay: float = 1e-4
    transformer_warmup_steps: int = 100
    transformer_patience: int = 6
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    use_quantile_lstm: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    rf_estimators: int = 80
    rf_max_samples: float = 0.75  # subsample rows to speed up
    gb_estimators: int = 100
    lgb_estimators: int = 200
    xgb_estimators: int = 150
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_dim_feedforward: int = 128
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    cache_processed: bool = True
    invalidate_processed_cache: bool = False
    results_dirname: str = "outputs"
    plots_dpi: int = 140
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# =============================================================================
# Utilities
# =============================================================================


def ensure_dirs() -> Dict[str, Path]:
    base = cfg.base_dir
    dirs = {
        "data": base / "data",
        "raw": base / "data" / "raw",
        "extracted": base / "data" / "extracted",
        "cache": base / "cache",
        "results": base / cfg.results_dirname,
        "plots": base / cfg.results_dirname / "plots",
        "tables": base / cfg.results_dirname / "tables",
        "models": base / "models",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    if cfg.invalidate_cache:
        print("Invalidating caches...")
        shutil.rmtree(dirs["cache"], ignore_errors=True)
        dirs["cache"].mkdir(parents=True, exist_ok=True)
    if cfg.invalidate_processed_cache:
        proc_cache = dirs["cache"] / "processed.pkl"
        if proc_cache.exists():
            proc_cache.unlink()
    return dirs



def download_file(urls: List[str], target: Path) -> None:
    import urllib.request

    if target.exists():
        print(f"Using cached dataset at {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        print(f"Downloading dataset from {url} ...")
        try:
            with urllib.request.urlopen(url) as resp, open(target, "wb") as f:
                total = int(resp.headers.get("content-length", 0))
                chunk = 1024 * 1024
                with tqdm(
                    total=total, unit="B", unit_scale=True, desc="download"
                ) as pbar:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        pbar.update(len(buf))
            print("Download complete.")
            return
        except Exception as e:  # pragma: no cover - best-effort for colab
            print(f"Download failed from {url}: {e}")
            last_err = e
    raise RuntimeError(f"All dataset download attempts failed: {last_err}")


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"Extraction skipped; found existing files in {dest_dir}")
        return
    print(f"Extracting {archive_path} ...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in {".tar", ".gz", ".tgz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    print("Extraction complete.")


def describe_device() -> str:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / (1024**3)
        return f"cuda:{torch.cuda.current_device()} ({props.name}, {mem_gb:.1f} GB)"
    return "cpu"


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


def fetch_via_kagglehub(slug: str, dest: Path) -> Optional[Path]:
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        print(f"kagglehub not available ({e}); attempting pip install...")
        try:
            import subprocess, sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
            import kagglehub  # type: ignore
        except Exception as e2:  # pragma: no cover - best-effort install
            print(f"kagglehub install failed ({e2}); skipping Kaggle fetch.")
            return None

    try:
        print(f"Downloading CMAPSS via kagglehub slug '{slug}' ...")
        path_str = kagglehub.dataset_download(slug)
        src_path = Path(path_str)
        if not src_path.exists():
            print("kagglehub returned missing path; skipping.")
            return None
        dest.mkdir(parents=True, exist_ok=True)
        # Copy tree to dest for consistency with existing loaders
        for item in src_path.rglob("*"):
            rel = item.relative_to(src_path)
            target = dest / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        print(f"kagglehub download staged at {dest}")
        return dest
    except Exception as e:
        print(f"kagglehub download failed: {e}")
        return None


def generate_synthetic_cmapss(n_engines: int = 20, cycles: int = 150) -> pd.DataFrame:
    rows = []
    for eid in range(1, n_engines + 1):
        drift = np.random.uniform(0.001, 0.01, size=21)
        noise_scale = np.random.uniform(0.01, 0.05, size=21)
        for c in range(1, cycles + 1):
            settings = np.random.normal(0, 1, size=3)
            sensors = []
            for i in range(21):
                base = 1.0 + 0.1 * i
                val = base + drift[i] * c + np.random.normal(0, noise_scale[i])
                sensors.append(val)
            row = [eid, c, *settings, *sensors]
            rows.append(row)
    df = pd.DataFrame(rows, columns=CMAPSS_COLUMN_NAMES)
    return df


# =============================================================================
# Data loading and adaptation
# =============================================================================


CMAPSS_COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_split(dataset_dir: Path, split: str, dataset_name: str) -> pd.DataFrame:
    split_prefix = "train" if split == "train" else "test"
    candidates = [
        dataset_dir / f"{split_prefix}_{dataset_name}.txt",  # common pattern
        dataset_dir / f"{dataset_name}_{split_prefix}.txt",
        dataset_dir / f"{dataset_name}_{split_prefix}_{dataset_name}.txt",
    ]
    files = [p for p in candidates if p.exists()]
    if not files:
        files = list(dataset_dir.glob(f"*{dataset_name}*{split_prefix}*.txt"))
    if not files:
        files = list(dataset_dir.rglob(f"*{dataset_name}*{split_prefix}*.txt"))
    if not files:
        raise FileNotFoundError(f"Unable to locate CMAPSS file for {dataset_name} {split}")
    file_path = files[0]
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(CMAPSS_COLUMN_NAMES)]
    df.columns = CMAPSS_COLUMN_NAMES
    return df


def select_drift_sensors(df: pd.DataFrame, top_k: int = 3) -> Tuple[List[str], pd.DataFrame]:
    corrs = []
    for sensor in [c for c in df.columns if c.startswith("sensor_")]:
        series = df[sensor]
        corr = series.corr(df["cycle"])
        # Spearman captures monotonicity; fallback if NaN
        try:
            from scipy.stats import spearmanr

            sp_corr, _ = spearmanr(df["cycle"], series)
        except Exception:
            sp_corr = corr
        corrs.append({"sensor": sensor, "pearson": corr, "spearman": sp_corr})
    corr_df = pd.DataFrame(corrs)
    corr_df["abs_spearman"] = corr_df["spearman"].abs()
    corr_df = corr_df.sort_values("abs_spearman", ascending=False)
    top_sensors = corr_df.head(top_k)["sensor"].tolist()
    return top_sensors, corr_df


def compute_thresholds(
    g: pd.DataFrame,
    drift_sensors: List[str],
    ratio_range: Tuple[float, float],
    baseline_window: int,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    baseline = g[drift_sensors].head(baseline_window).median()
    tail = g[drift_sensors].tail(baseline_window).median()
    direction = np.sign(tail - baseline).replace(0, 1)
    thresholds = {}
    for s in drift_sensors:
        ratio = random.uniform(*ratio_range)
        thresholds[s] = baseline[s] + (tail[s] - baseline[s]) * ratio
    return baseline, direction, thresholds


def simulate_resets(
    g: pd.DataFrame,
    drift_sensors: List[str],
    baseline: pd.Series,
    direction: pd.Series,
    thresholds: Dict[str, float],
    mode: str,
    donor_df: Optional[pd.DataFrame],
    stitch_len: int,
    reset_noise: float,
    post_drift_scale: float,
    max_resets: int,
) -> pd.DataFrame:
    g = g.reset_index(drop=True).copy()
    g["calibration_event"] = False
    resets = 0
    start_idx = 0
    while resets < max_resets:
        over_thresh = np.zeros(len(g), dtype=bool)
        for s in drift_sensors:
            if direction[s] >= 0:
                over_thresh |= g[s].values >= thresholds[s]
            else:
                over_thresh |= g[s].values <= thresholds[s]
        if not over_thresh[start_idx:].any():
            break
        first_cross = int(np.argmax(over_thresh[start_idx:])) + start_idx
        post = g.iloc[first_cross:].copy()
        n_post = len(post)
        # reset to baseline + noise and allow re-drift
        for s in drift_sensors:
            drift_vector = np.linspace(
                0, (thresholds[s] - baseline[s]) * post_drift_scale, n_post
            )
            noise = np.random.normal(0, reset_noise, size=n_post)
            post[s] = baseline[s] + drift_vector + noise

        if mode == "stitch" and donor_df is not None:
            donor = donor_df.head(stitch_len).copy()
            donor = donor.reset_index(drop=True)
            donor_cycle_start = int(g["cycle"].iloc[first_cross]) + 1
            donor["cycle"] = np.arange(donor_cycle_start, donor_cycle_start + len(donor))
            for s in drift_sensors:
                donor[s] = donor[s].values + np.random.normal(
                    0, reset_noise, size=len(donor)
                )
            donor = donor.iloc[:n_post] if len(donor) >= n_post else donor
            post.update(donor[post.columns])

        g = pd.concat([g.iloc[: first_cross + 1], post], ignore_index=True)
        g.loc[first_cross, "calibration_event"] = True
        start_idx = first_cross + 1
        resets += 1
    return g


def time_to_next_threshold(
    g: pd.DataFrame, drift_sensors: List[str], thresholds: Dict[str, float], direction: pd.Series
) -> pd.Series:
    over_thresh = np.zeros(len(g), dtype=bool)
    for s in drift_sensors:
        if direction[s] >= 0:
            over_thresh |= g[s].values >= thresholds[s]
        else:
            over_thresh |= g[s].values <= thresholds[s]

    ttd = np.zeros(len(g), dtype=np.int32)
    next_hit = None
    for idx in range(len(g) - 1, -1, -1):
        if over_thresh[idx]:
            next_hit = idx
        if next_hit is None:
            # No future threshold; set to distance to end
            ttd[idx] = len(g) - idx - 1
        else:
            ttd[idx] = max(0, next_hit - idx)
    return pd.Series(ttd, name="time_to_drift")


def adapt_dataset(df: pd.DataFrame, drift_sensors: List[str]) -> pd.DataFrame:
    adapted_segments = []
    donor_pool = df.copy()
    engine_ids = sorted(df["engine_id"].unique())
    if cfg.use_small_run:
        engine_ids = engine_ids[: cfg.engine_limit]
    for idx, eid in enumerate(engine_ids):
        g = df[df["engine_id"] == eid].sort_values("cycle")
        baseline, direction, thresholds = compute_thresholds(
            g,
            drift_sensors,
            (cfg.threshold_ratio_min, cfg.threshold_ratio_max),
            cfg.baseline_window,
        )
        for rep in range(cfg.calibration_repeats):
            donor_df = None
            if cfg.reset_mode == "stitch":
                donor_eid = random.choice(engine_ids)
                donor_df = donor_pool[donor_pool["engine_id"] == donor_eid].sort_values(
                    "cycle"
                )
            g_reset = simulate_resets(
                g,
                drift_sensors=drift_sensors,
                baseline=baseline,
                direction=direction,
                thresholds=thresholds,
                mode=cfg.reset_mode,
                donor_df=donor_df,
                stitch_len=cfg.stitch_length,
                reset_noise=cfg.reset_noise,
                post_drift_scale=cfg.post_drift_scale,
                max_resets=cfg.max_resets_per_run,
            )
            ttd = time_to_next_threshold(g_reset, drift_sensors, thresholds, direction)
            g_reset = g_reset.assign(time_to_drift=ttd.values, synthetic_run=rep)
            adapted_segments.append(g_reset)
    adapted = pd.concat(adapted_segments, ignore_index=True)
    return adapted


# =============================================================================
# Sequence construction
# =============================================================================


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    ids: List[Tuple[int, int]] = []
    grouped = df.groupby(["engine_id", "synthetic_run"])
    for (_, _), g in grouped:
        g = g.sort_values("cycle")
        values = g[feature_cols].values
        labels = g[label_col].values
        for start in range(0, len(g) - window + 1, stride):
            end = start + window
            xs.append(values[start:end])
            ys.append(labels[end - 1])  # predict last-step time-to-drift
            ids.append((int(g["engine_id"].iloc[0]), int(g["synthetic_run"].iloc[0])))
    return np.stack(xs), np.array(ys), ids


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def standardize_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats = {}
    train_df = train_df.copy()
    val_df = val_df.copy()
    for col in feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-6
        train_df[col] = (train_df[col] - mean) / std
        val_df[col] = (val_df[col] - mean) / std
        stats[col] = (mean, std)
    return train_df, val_df, stats


# =============================================================================
# Models
# =============================================================================


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class QuantileLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float, quantiles: List[float]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class CNNRegressor(nn.Module):
    def __init__(self, input_size: int, window: int, dropout: float):
        super().__init__()
        # Conv1d expects (batch, channels, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.conv(x)
        return self.head(out)


class TCNRegressor(nn.Module):
    def __init__(self, input_size: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1)
        self.net = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding * 2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out = self.net(x)
        return self.head(out)


class QuantileTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.input_proj = nn.Linear(input_size, d_model)
        self.register_buffer("pos_cache", torch.zeros(1), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        output_dim = len(quantiles) if quantiles else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def positional_encoding(self, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        h = self.input_proj(x)
        pe = self.positional_encoding(h.size(1), h.size(2), h.device)
        h = h + pe
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


# =============================================================================
# Training utilities
# =============================================================================


def auto_batch_size(
    model: nn.Module, sample: torch.Tensor, device: str, candidates: List[int]
) -> int:
    model = model.to(device)
    for bs in candidates:
        try:
            dummy = sample.unsqueeze(0).repeat(bs, 1, 1).to(device)
            with torch.no_grad():
                _ = model(dummy)
            torch.cuda.empty_cache()
            print(f"Selected batch size {bs}")
            return bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise
    print("Falling back to batch size 16")
    return 16


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i : i + 1]
        losses.append(torch.maximum((q - 1) * errors, q * errors))
    return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


def train_lstm(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_size: int,
    device: str,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    if cfg.use_quantile_lstm:
        model: nn.Module = QuantileLSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            quantiles=cfg.quantiles,
        )
    else:
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    model.apply(init_weights)
    model = model.to(device)

    sample_x, _ = train_ds[0]
    batch_size = auto_batch_size(model, sample_x, device, cfg.batch_candidates)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0
    total_steps = cfg.epochs * len(train_loader)
    train_pbar = tqdm(total=total_steps, desc="[LSTM] training", leave=False)

    print("Training LSTM...")
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if cfg.use_quantile_lstm:
                loss = pinball_loss(preds, yb, cfg.quantiles)
            else:
                loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pbar.update(1)
            if len(train_losses) % 10 == 0:
                train_pbar.set_postfix(epoch=epoch + 1, loss=f"{np.mean(train_losses[-10:]):.4f}")
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                if cfg.use_quantile_lstm:
                    val_losses.append(pinball_loss(preds, yb, cfg.quantiles).item())
                else:
                    val_losses.append(criterion(preds, yb).item())
        val_mean = float(np.mean(val_losses))
        history["val_loss"].append(val_mean)
        if val_mean + 1e-6 < best_val:
            best_val = val_mean
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break

    train_pbar.close()
    if "best_state" in locals():
        model.load_state_dict(best_state)
    return model, history


def train_cnn(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_size: int,
    device: str,
) -> Tuple[CNNRegressor, Dict[str, List[float]]]:
    model = CNNRegressor(input_size=input_size, window=cfg.window_size, dropout=cfg.dropout)
    model.apply(init_weights)
    model = model.to(device)
    sample_x, _ = train_ds[0]
    batch_size = auto_batch_size(model, sample_x, device, cfg.batch_candidates)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0
    total_steps = cfg.epochs * len(train_loader)
    train_pbar = tqdm(total=total_steps, desc="[CNN] training", leave=False)

    print("Training CNN...")
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pbar.update(1)
            if len(train_losses) % 10 == 0:
                train_pbar.set_postfix(epoch=epoch + 1, loss=f"{np.mean(train_losses[-10:]):.4f}")
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())
        val_mean = float(np.mean(val_losses))
        history["val_loss"].append(val_mean)
        if val_mean + 1e-6 < best_val:
            best_val = val_mean
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("[CNN] Early stopping triggered.")
                break
    train_pbar.close()
    if "best_state" in locals():
        model.load_state_dict(best_state)
    return model, history


def train_transformer(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_size: int,
    device: str,
    quantiles: Optional[List[float]] = None,
) -> Tuple[QuantileTransformer, Dict[str, List[float]]]:
    model = QuantileTransformer(
        input_size=input_size,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        dim_feedforward=cfg.transformer_dim_feedforward,
        num_layers=cfg.transformer_layers,
        dropout=cfg.transformer_dropout,
        quantiles=quantiles,
    )
    model.apply(init_weights)
    model = model.to(device)
    sample_x, _ = train_ds[0]
    batch_size = auto_batch_size(model, sample_x, device, cfg.batch_candidates)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    use_quantiles = bool(quantiles)
    criterion = nn.SmoothL1Loss() if not use_quantiles else None
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.transformer_lr,
        weight_decay=cfg.transformer_weight_decay,
    )
    total_steps = max(1, cfg.transformer_epochs * len(train_loader))
    warmup_steps = min(cfg.transformer_warmup_steps, total_steps // 2)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0
    train_pbar = tqdm(total=total_steps, desc="[Transformer] training", leave=False)

    print("Training Transformer...")
    global_step = 0
    for epoch in range(cfg.transformer_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if use_quantiles:
                loss = pinball_loss(preds, yb, quantiles if quantiles else [])
            else:
                loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses.append(loss.item())
            train_pbar.update(1)
            if len(train_losses) % 10 == 0:
                train_pbar.set_postfix(epoch=epoch + 1, loss=f"{np.mean(train_losses[-10:]):.4f}")
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                if use_quantiles:
                    val_losses.append(pinball_loss(preds, yb, quantiles if quantiles else []).item())
                else:
                    val_losses.append(criterion(preds, yb).item())
        val_mean = float(np.mean(val_losses))
        history["val_loss"].append(val_mean)
        if val_mean + 1e-6 < best_val:
            best_val = val_mean
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.transformer_patience:
                print("[Transformer] Early stopping triggered.")
                break
    train_pbar.close()
    if "best_state" in locals():
        model.load_state_dict(best_state)
    return model, history


def train_tcn(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_size: int,
    device: str,
) -> Tuple[TCNRegressor, Dict[str, List[float]]]:
    model = TCNRegressor(
        input_size=input_size,
        channels=cfg.tcn_channels,
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.tcn_dropout,
    )
    model.apply(init_weights)
    model = model.to(device)
    sample_x, _ = train_ds[0]
    batch_size = auto_batch_size(model, sample_x, device, cfg.batch_candidates)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0
    total_steps = cfg.epochs * len(train_loader)
    train_pbar = tqdm(total=total_steps, desc="[TCN] training", leave=False)

    print("Training TCN...")
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pbar.update(1)
            if len(train_losses) % 10 == 0:
                train_pbar.set_postfix(epoch=epoch + 1, loss=f"{np.mean(train_losses[-10:]):.4f}")
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())
        val_mean = float(np.mean(val_losses))
        history["val_loss"].append(val_mean)
        if val_mean + 1e-6 < best_val:
            best_val = val_mean
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("[TCN] Early stopping triggered.")
                break
    train_pbar.close()
    if "best_state" in locals():
        model.load_state_dict(best_state)
    return model, history


def evaluate_torch_model(
    model: nn.Module,
    ds: SequenceDataset,
    device: str,
    quantiles: Optional[List[float]] = None,
    clip_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    preds: List[float] = []
    truths: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            if quantiles:
                # take median quantile for point estimate
                q_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
                out = out[:, q_idx : q_idx + 1]
            out_np = out.squeeze(-1).cpu().numpy()
            if clip_min is not None:
                out_np = np.maximum(out_np, clip_min)
            preds.extend(out_np)
            truths.extend(yb.squeeze(-1).cpu().numpy())
    preds_arr = np.array(preds)
    truths_arr = np.array(truths)
    metrics = {
        "MAE": float(mean_absolute_error(truths_arr, preds_arr)),
        "RMSE": float(mean_squared_error(truths_arr, preds_arr) ** 0.5),
        "R2": float(r2_score(truths_arr, preds_arr)),
    }
    return preds_arr, truths_arr, metrics


def evaluate_quantile_model(
    model: nn.Module,
    ds: SequenceDataset,
    device: str,
    quantiles: List[float],
    clip_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    """Evaluate a quantile regressor, returning the full quantile matrix alongside median metrics."""
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    quantile_preds: List[np.ndarray] = []
    truths: List[float] = []
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            out_np = out.cpu().numpy()
            quantile_preds.append(out_np)
            truths.extend(yb.squeeze(-1).cpu().numpy())
    quantile_arr = np.concatenate(quantile_preds, axis=0)
    if clip_min is not None:
        quantile_arr = np.maximum(quantile_arr, clip_min)
    preds_arr = quantile_arr[:, median_idx]
    truths_arr = np.array(truths)
    metrics = {
        "MAE": float(mean_absolute_error(truths_arr, preds_arr)),
        "RMSE": float(mean_squared_error(truths_arr, preds_arr) ** 0.5),
        "R2": float(r2_score(truths_arr, preds_arr)),
    }
    return preds_arr, truths_arr, metrics, quantile_arr


def train_linear_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> pd.DataFrame:
    rows = []
    print(
        f"[Baseline-Easy] LinearRegression: train={len(X_train)}, val={len(X_val)}, features={X_train.shape[1]}",
        flush=True,
    )
    for seed in cfg.seeds:
        model = LinearRegression()
        start = time.monotonic()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        elapsed = time.monotonic() - start
        print(f"[Baseline-Easy] seed={seed}: fit+predict in {elapsed:.2f}s", flush=True)
        rows.append(
            {
                "model": "Linear",
                "seed": seed,
                "MAE": float(mean_absolute_error(y_val, preds)),
                "RMSE": float(mean_squared_error(y_val, preds) ** 0.5),
                "R2": float(r2_score(y_val, preds)),
            }
        )
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("model")
        .agg({"MAE": ["mean", "std"], "RMSE": ["mean", "std"], "R2": ["mean", "std"]})
        .reset_index()
    )
    agg.columns = ["model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]
    return agg


def train_baselines(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> pd.DataFrame:
    rows = []
    print(
        f"[Baselines] Starting fits: train={len(X_train)}, val={len(X_val)}, features={X_train.shape[1]}",
        flush=True,
    )
    for seed in cfg.seeds:
        # Random forest underperforms significantly, we disable this for now
        # models = {
        #     "RandomForest": RandomForestRegressor(
        #         n_estimators=cfg.rf_estimators,
        #         max_depth=None,
        #         n_jobs=-1,
        #         random_state=seed,
        #         max_features="sqrt",
        #         max_samples=cfg.rf_max_samples,
        #     )
        # }
        models = {}

        # Optional XGBoost
        try:
            import xgboost as xgb  # type: ignore

            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=cfg.xgb_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=seed,
            )
        except Exception:
            pass
        # Optional slower baselines (uncomment for final runs)
        # models["GradientBoosting"] = GradientBoostingRegressor(
        #     random_state=seed, n_estimators=cfg.gb_estimators, learning_rate=0.05
        # )
        try:
            import lightgbm as lgb  # type: ignore

            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=cfg.lgb_estimators,
                learning_rate=0.05,
                num_leaves=48,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=seed,
                force_col_wise=True,
                verbose=-1,
            )
        except Exception:
            pass

        print(f"[Baselines] seed={seed}: models={list(models.keys())}", flush=True)
        for name, model in models.items():
            start = time.monotonic()
            print(
                f"[Baselines] seed={seed}, model={name}: fitting...",
                flush=True,
            )
            model.fit(X_train, y_train)
            fit_secs = time.monotonic() - start
            print(
                f"[Baselines] seed={seed}, model={name}: fit done in {fit_secs:.2f}s; predicting...",
                flush=True,
            )
            preds = model.predict(X_val)
            pred_secs = time.monotonic() - start
            mae = float(mean_absolute_error(y_val, preds))
            rmse = float(mean_squared_error(y_val, preds) ** 0.5)
            r2 = float(r2_score(y_val, preds))
            print(
                f"[Baselines] seed={seed}, model={name}: predict+metrics done in {pred_secs:.2f}s "
                f"(MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f})",
                flush=True,
            )
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("model")
        .agg({"MAE": ["mean", "std"], "RMSE": ["mean", "std"], "R2": ["mean", "std"]})
        .reset_index()
    )
    total_secs = sum(rows[i]["RMSE"] * 0 for i in range(len(rows)))  # placeholder to keep structure
    print(f"[Baselines] Completed all fits for seeds={cfg.seeds}", flush=True)
    agg.columns = ["model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]
    return agg


# =============================================================================
# Visualization and reporting
# =============================================================================


def plot_correlations(corr_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(x="abs_spearman", y="sensor", data=corr_df.head(10), color="C0")
    plt.title("Sensor monotonicity with cycle (top-10)")
    plt.xlabel("|Spearman correlation|")
    plt.ylabel("Sensor")
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def plot_threshold_example(df: pd.DataFrame, drift_sensors: List[str], out_path: Path) -> None:
    sample_engine = df["engine_id"].iloc[0]
    g = df[df["engine_id"] == sample_engine].sort_values("cycle")
    plt.figure(figsize=(10, 4))
    for s in drift_sensors:
        plt.plot(g["cycle"], g[s], label=s)
    plt.plot(
        g["cycle"],
        g["time_to_drift"],
        label="Time-to-drift label",
        linestyle="--",
        alpha=0.6,
    )
    plt.xlabel("Cycle")
    plt.ylabel("Sensor value / TTD")
    plt.title(f"Drift sensors and time-to-drift for engine {sample_engine}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def plot_predictions(
    preds: np.ndarray, truths: np.ndarray, out_path: Path, title: str = "Pred vs True"
) -> None:
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=truths, y=preds, alpha=0.4)
    maxv = max(truths.max(), preds.max())
    plt.plot([0, maxv], [0, maxv], "r--", label="ideal")
    plt.xlabel("True TTD")
    plt.ylabel("Predicted TTD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def plot_residuals(preds: np.ndarray, truths: np.ndarray, out_path: Path) -> None:
    residuals = preds - truths
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(x=truths, y=residuals, alpha=0.4, ax=axes[0])
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("True TTD")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs True")

    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title("Residual distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def plot_priority_histogram(schedule_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(schedule_df["predicted_ttd"], bins=30, kde=True)
    plt.xlabel("Predicted TTD")
    plt.ylabel("Count")
    plt.title("Calibration priority distribution (lower = more urgent)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.plots_dpi)
    plt.close()


def save_table(data: pd.DataFrame, out_path: Path) -> None:
    data.to_csv(out_path, index=False)


def build_text_summary(
    drift_sensors: List[str],
    corr_df: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    model_metrics: Dict[str, Dict[str, float]],
    schedule_table: pd.DataFrame,
    policy_costs: pd.DataFrame,
    safety_margin: float,
    risk_quantile: Optional[float],
    out_path: Path,
) -> str:
    lines = []
    lines.append(f"Device: {describe_device()}")
    lines.append(f"Selected drift sensors: {', '.join(drift_sensors)}")
    lines.append(f"Safety margin for predictive policies: {safety_margin}")
    if risk_quantile is not None:
        lines.append(f"Quantile-aware trigger uses q{int(risk_quantile * 100)} TTD estimates.")
    lines.append("Top monotonic sensors (|Spearman|):")
    for _, row in corr_df.head(5).iterrows():
        lines.append(f"  {row['sensor']}: {row['abs_spearman']:.3f}")
    lines.append("")
    lines.append("Baseline metrics:")
    for _, row in baseline_metrics.iterrows():
        lines.append(
            f"  {row['model']}: MAE={row['MAE_mean']:.3f}±{row['MAE_std']:.3f}, "
            f"RMSE={row['RMSE_mean']:.3f}±{row['RMSE_std']:.3f}, "
            f"R2={row['R2_mean']:.3f}±{row['R2_std']:.3f}"
        )
    lines.append("")
    for name, metrics in model_metrics.items():
        lines.append(
            f"{name} metrics: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}"
        )
    lines.append("")
    lines.append("Top-10 calibration priorities (small TTD first):")
    for _, row in schedule_table.head(10).iterrows():
        lines.append(
            f"  Engine {int(row['engine_id'])} | predicted TTD={row['predicted_ttd']:.2f} | "
            f"urgency={row['urgency_score']:.2f}"
        )
    lines.append("")
    lines.append("Policy cost estimate (calibration count, violations, cost):")
    for _, row in policy_costs.iterrows():
        lines.append(
            f"  {row['policy']}: cals={row['calibrations']}, violations={row['violations']}, cost={row['cost']:.2f}"
        )
    text = "\n".join(lines)
    out_path.write_text(text)
    return text


# =============================================================================
# Scheduling
# =============================================================================


def build_schedule(pred_df: pd.DataFrame, capacity: int = 5, risk_col: Optional[str] = None) -> pd.DataFrame:
    pred_df = pred_df.copy()
    score_col = risk_col if risk_col and risk_col in pred_df.columns else "predicted_ttd"
    pred_df["urgency_score"] = 1.0 / (pred_df[score_col] + 1e-6)
    pred_df = pred_df.sort_values(score_col)
    pred_df["scheduled"] = False
    pred_df.loc[pred_df.index[:capacity], "scheduled"] = True
    return pred_df


def estimate_policy_costs(
    preds: np.ndarray,
    truths: np.ndarray,
    margin: Optional[float] = None,
    quantile_preds: Optional[np.ndarray] = None,
    quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    preds = np.maximum(preds, 0.0)
    margin = cfg.safety_margin if margin is None else margin
    violation_mask = truths <= 0.0  # true threshold crossings from labels

    reactive_violations = int(violation_mask.sum())
    reactive_calibs = reactive_violations

    predictive_calibs = int((preds <= margin).sum())
    predictive_violations = int(np.logical_and(violation_mask, preds > margin).sum())

    fixed_interval = max(cfg.fixed_interval, 1)
    fixed_calibs = math.ceil(len(truths) / fixed_interval)
    fixed_violations = reactive_violations

    rows = [
        {"policy": "Reactive", "calibrations": reactive_calibs, "violations": reactive_violations},
        {"policy": "Fixed", "calibrations": fixed_calibs, "violations": fixed_violations},
        {"policy": "Predictive", "calibrations": predictive_calibs, "violations": predictive_violations},
    ]
    if quantile_preds is not None:
        risk_idx = 0
        if quantiles:
            if 0.1 in quantiles:
                risk_idx = quantiles.index(0.1)
            else:
                risk_idx = 0
        risk_preds = np.maximum(quantile_preds[:, risk_idx], 0.0)
        q_calibs = int((risk_preds <= margin).sum())
        q_violations = int(np.logical_and(violation_mask, risk_preds > margin).sum())
        label = "QuantilePredictive"
        if quantiles:
            label = f"Quantile@{int(quantiles[risk_idx]*100)}"
        rows.append({"policy": label, "calibrations": q_calibs, "violations": q_violations})
    df = pd.DataFrame(rows)
    df["cost"] = df["calibrations"] * cfg.calib_cost + df["violations"] * cfg.violation_cost
    return df


# =============================================================================
# Main pipeline
# =============================================================================


def maybe_load_cached_processed(cache_path: Path) -> Optional[Dict[str, object]]:
    if cache_path.exists() and cfg.cache_processed:
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded processed cache from {cache_path}")
            if isinstance(data, dict):
                return data
            if isinstance(data, tuple) and len(data) == 2:
                adapted_df, corr_df = data
                return {"adapted_df": adapted_df, "corr_df": corr_df}
        except Exception:
            return None
    return None


def run_pipeline(dataset_name: Optional[str] = None) -> None:
    global DIRS
    if dataset_name:
        cfg.dataset_name = dataset_name
        cfg.base_dir = Path("calibration_outputs") / dataset_name
    DIRS = ensure_dirs()
    print("Starting calibration scheduling pipeline...")
    print(f"Using device: {describe_device()}")

    cache_path = DIRS["cache"] / "processed.pkl"
    cached = maybe_load_cached_processed(cache_path)
    if cached:
        adapted_df = cached["adapted_df"]
        corr_df = cached["corr_df"]
        drift_sensors = cached.get(
            "drift_sensors", corr_df.sort_values("abs_spearman", ascending=False).head(3)["sensor"].tolist()
        )
    else:
        dataset_root: Optional[Path] = None
        if cfg.prefer_kaggle:
            dataset_root = fetch_via_kagglehub(cfg.kaggle_slug, DIRS["extracted"])

        if dataset_root is None:
            archive_path = DIRS["raw"] / "CMAPSSData.zip"
            try:
                download_file(cfg.dataset_urls, archive_path)
                extract_archive(archive_path, DIRS["extracted"])
                dataset_root = DIRS["extracted"]
            except Exception as e:
                print(f"Dataset download failed ({e}); falling back to synthetic CMAPSS-like data.")
                train_df = generate_synthetic_cmapss()
                dataset_root = None

        if dataset_root is not None:
            train_df = load_cmapss_split(dataset_root, "train", cfg.dataset_name)
        print(f"Loaded train split: {train_df.shape}")
        drift_sensors, corr_df = select_drift_sensors(train_df)
        print(f"Selected drift sensors: {drift_sensors}")
        adapted_df = adapt_dataset(train_df, drift_sensors)
        if cfg.cache_processed:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "adapted_df": adapted_df,
                        "corr_df": corr_df,
                        "drift_sensors": drift_sensors,
                    },
                    f,
                )

    max_monotonicity = float(corr_df["abs_spearman"].max())
    adaptive_margin = cfg.safety_margin
    if max_monotonicity < 0.35:
        adaptive_margin = max(cfg.safety_margin, 8)
        print(
            f"Weak drift signal detected (|Spearman|max={max_monotonicity:.2f}); "
            f"using conservative safety margin {adaptive_margin} cycles for predictive policies."
        )
    else:
        print(f"Top drift monotonicity |Spearman|={max_monotonicity:.2f}; safety margin={adaptive_margin}")

    # Prepare sequences with engine-level split and normalization
    feature_cols = [c for c in adapted_df.columns if c.startswith("sensor_")]
    engine_ids = sorted(adapted_df["engine_id"].unique())
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=1 - cfg.train_fraction, random_state=cfg.seed
    )
    train_df = adapted_df[adapted_df["engine_id"].isin(train_ids)]
    val_df = adapted_df[adapted_df["engine_id"].isin(val_ids)]
    train_df, val_df, norm_stats = standardize_features(train_df, val_df, feature_cols)

    X_train, y_train, train_seq_ids = build_sequences(
        train_df,
        feature_cols=feature_cols,
        label_col="time_to_drift",
        window=cfg.window_size,
        stride=cfg.window_stride,
    )
    X_val, y_val, val_seq_ids = build_sequences(
        val_df,
        feature_cols=feature_cols,
        label_col="time_to_drift",
        window=cfg.window_size,
        stride=cfg.window_stride,
    )
    print(f"Sequence dataset: train {X_train.shape}, val {X_val.shape}")

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("No sequences produced; consider lowering window_size or disabling small_run.")

    if cfg.use_small_run:
        max_train = min(6000, len(X_train))
        max_val = min(3000, len(X_val))
        X_train, y_train = X_train[:max_train], y_train[:max_train]
        X_val, y_val, val_seq_ids = X_val[:max_val], y_val[:max_val], val_seq_ids[:max_val]
        print(f"Using small-run subset: train {len(X_train)}, val {len(X_val)}")

    flat_train = X_train.reshape(len(X_train), -1)
    flat_val = X_val.reshape(len(X_val), -1)
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    # 1) Train our sequence models first for quick signal
    lstm_model, history_lstm = train_lstm(train_ds, val_ds, X_train.shape[-1], cfg.device)
    lstm_quantiles = None
    if cfg.use_quantile_lstm:
        lstm_preds, lstm_truths, lstm_metrics, lstm_quantiles = evaluate_quantile_model(
            lstm_model, val_ds, cfg.device, cfg.quantiles, clip_min=0.0
        )
    else:
        lstm_preds, lstm_truths, lstm_metrics = evaluate_torch_model(
            lstm_model, val_ds, cfg.device, clip_min=0.0
        )
    print(
        f"[Metrics] LSTM (post-train) MAE={lstm_metrics['MAE']:.3f}, RMSE={lstm_metrics['RMSE']:.3f}, R2={lstm_metrics['R2']:.3f}",
        flush=True,
    )

    cnn_model, history_cnn = train_cnn(train_ds, val_ds, X_train.shape[-1], cfg.device)
    cnn_preds, cnn_truths, cnn_metrics = evaluate_torch_model(
        cnn_model, val_ds, cfg.device, clip_min=0.0
    )
    print(
        f"[Metrics] CNN (post-train) MAE={cnn_metrics['MAE']:.3f}, RMSE={cnn_metrics['RMSE']:.3f}, R2={cnn_metrics['R2']:.3f}",
        flush=True,
    )

    transformer_model, history_transformer = train_transformer(
        train_ds, val_ds, X_train.shape[-1], cfg.device, quantiles=cfg.quantiles
    )
    transformer_quantiles = None
    if cfg.quantiles:
        transformer_preds, transformer_truths, transformer_metrics, transformer_quantiles = evaluate_quantile_model(
            transformer_model, val_ds, cfg.device, cfg.quantiles, clip_min=0.0
        )
    else:
        transformer_preds, transformer_truths, transformer_metrics = evaluate_torch_model(
            transformer_model, val_ds, cfg.device, clip_min=0.0
        )
    risk_q_value: Optional[float] = cfg.quantiles[0] if cfg.quantiles else None
    tcn_model, history_tcn = train_tcn(train_ds, val_ds, X_train.shape[-1], cfg.device)
    tcn_preds, tcn_truths, tcn_metrics = evaluate_torch_model(
        tcn_model, val_ds, cfg.device, clip_min=0.0
    )
    print(
        f"[Metrics] TRF (post-train) MAE={transformer_metrics['MAE']:.3f}, RMSE={transformer_metrics['RMSE']:.3f}, R2={transformer_metrics['R2']:.3f}",
        flush=True,
    )
    print(
        f"[Metrics] TCN (post-train) MAE={tcn_metrics['MAE']:.3f}, RMSE={tcn_metrics['RMSE']:.3f}, R2={tcn_metrics['R2']:.3f}",
        flush=True,
    )

    # 2) Easy baseline (Linear) for quick comparison
    easy_baseline = train_linear_baseline(flat_train, y_train, flat_val, y_val).fillna(0)

    # 3) Other baselines (trees/boosting)
    baseline_metrics = train_baselines(flat_train, y_train, flat_val, y_val).fillna(0)

    # Priority schedule using Transformer predictions on validation set
    priority_data = {
        "engine_id": [eid for eid, _ in val_seq_ids],
        "predicted_ttd": np.maximum(transformer_preds, 0.0),
        "true_ttd": transformer_truths,
    }
    risk_col = None
    if transformer_quantiles is not None and cfg.quantiles:
        for idx, q in enumerate(cfg.quantiles):
            q_label = int(round(q * 100))
            priority_data[f"predicted_ttd_q{q_label}"] = transformer_quantiles[:, idx]
        risk_col = f"predicted_ttd_q{int(round(cfg.quantiles[0] * 100))}"
    priority_df = pd.DataFrame(priority_data)
    schedule_df = build_schedule(priority_df, capacity=10, risk_col=risk_col)
    policy_costs = estimate_policy_costs(
        transformer_preds,
        transformer_truths,
        margin=adaptive_margin,
        quantile_preds=transformer_quantiles,
        quantiles=cfg.quantiles,
    )

    # Outputs
    plot_correlations(corr_df, DIRS["plots"] / "drift_sensor_correlations.png")
    plot_threshold_example(adapted_df, drift_sensors, DIRS["plots"] / "threshold_example.png")
    plot_training_curves(history_lstm, DIRS["plots"] / "lstm_training.png", title="LSTM training curves")
    plot_training_curves(history_cnn, DIRS["plots"] / "cnn_training.png", title="CNN training curves")
    plot_training_curves(
        history_transformer, DIRS["plots"] / "transformer_training.png", title="Transformer training curves"
    )
    plot_training_curves(history_tcn, DIRS["plots"] / "tcn_training.png", title="TCN training curves")
    plot_predictions(
        transformer_preds,
        transformer_truths,
        DIRS["plots"] / "transformer_pred_vs_true.png",
        title="Transformer predicted vs true TTD",
    )
    plot_residuals(
        transformer_preds,
        transformer_truths,
        DIRS["plots"] / "transformer_residuals.png",
    )
    plot_predictions(
        lstm_preds,
        lstm_truths,
        DIRS["plots"] / "lstm_pred_vs_true.png",
        title="LSTM predicted vs true TTD",
    )
    plot_residuals(lstm_preds, lstm_truths, DIRS["plots"] / "lstm_residuals.png")
    plot_priority_histogram(schedule_df, DIRS["plots"] / "priority_hist.png")

    baseline_table = pd.concat([easy_baseline, baseline_metrics], ignore_index=True)
    deep_rows = pd.DataFrame(
        [
            {"model": "LSTM", "MAE_mean": lstm_metrics["MAE"], "MAE_std": 0.0, "RMSE_mean": lstm_metrics["RMSE"], "RMSE_std": 0.0, "R2_mean": lstm_metrics["R2"], "R2_std": 0.0},
            {"model": "CNN", "MAE_mean": cnn_metrics["MAE"], "MAE_std": 0.0, "RMSE_mean": cnn_metrics["RMSE"], "RMSE_std": 0.0, "R2_mean": cnn_metrics["R2"], "R2_std": 0.0},
            {"model": "Transformer", "MAE_mean": transformer_metrics["MAE"], "MAE_std": 0.0, "RMSE_mean": transformer_metrics["RMSE"], "RMSE_std": 0.0, "R2_mean": transformer_metrics["R2"], "R2_std": 0.0},
            {"model": "TCN", "MAE_mean": tcn_metrics["MAE"], "MAE_std": 0.0, "RMSE_mean": tcn_metrics["RMSE"], "RMSE_std": 0.0, "R2_mean": tcn_metrics["R2"], "R2_std": 0.0},
        ]
    )
    metrics_table = pd.concat([baseline_table, deep_rows], ignore_index=True)
    save_table(metrics_table, DIRS["tables"] / "model_metrics.csv")
    save_table(schedule_df, DIRS["tables"] / "calibration_schedule.csv")
    save_table(policy_costs, DIRS["tables"] / "policy_costs.csv")

    print(
        f"[Metrics] LSTM MAE={lstm_metrics['MAE']:.3f}, RMSE={lstm_metrics['RMSE']:.3f}, R2={lstm_metrics['R2']:.3f}",
        flush=True,
    )
    print(
        f"[Metrics] CNN  MAE={cnn_metrics['MAE']:.3f}, RMSE={cnn_metrics['RMSE']:.3f}, R2={cnn_metrics['R2']:.3f}",
        flush=True,
    )
    print(
        f"[Metrics] TRF  MAE={transformer_metrics['MAE']:.3f}, RMSE={transformer_metrics['RMSE']:.3f}, R2={transformer_metrics['R2']:.3f}",
        flush=True,
    )
    print("[Metrics] Baselines summary (mean):", flush=True)
    print(metrics_table[["model", "MAE_mean", "RMSE_mean", "R2_mean"]].to_string(index=False), flush=True)

    text_summary = build_text_summary(
        drift_sensors=drift_sensors,
        corr_df=corr_df,
        baseline_metrics=baseline_table,
        model_metrics={
            "LSTM": lstm_metrics,
            "CNN": cnn_metrics,
            "Transformer": transformer_metrics,
            "TCN": tcn_metrics,
        },
        schedule_table=schedule_df,
        policy_costs=policy_costs,
        safety_margin=adaptive_margin,
        risk_quantile=risk_q_value,
        out_path=DIRS["results"] / "summary.txt",
    )
    print("\n=== Summary ===\n")
    print(text_summary)

    # Bundle results for download in Colab
    zip_basename = f"calibration_outputs_{cfg.dataset_name.lower()}"
    zip_target = Path(f"{zip_basename}.zip")
    shutil.make_archive(zip_basename, "zip", DIRS["results"])
    print(f"Zipped outputs at: {zip_target}")
    try:
        from google.colab import files  # type: ignore

        files.download(str(zip_target))
    except Exception:
        print("Colab download not available; zip file saved locally.")

    # Notebook-friendly snippet (executed for convenience)
    try:
        import IPython  # type: ignore

        colab_zip = f"/content/{zip_target.name}"
        snippet = (
            "```\n"
            f"!zip -r {colab_zip} {DIRS['results']}\n"
            "from google.colab import files\n"
            f"files.download('{colab_zip}')\n"
            "```"
        )
        IPython.display.display(IPython.display.Markdown(snippet))
    except Exception:
        pass


if __name__ == "__main__":
    if cfg.dataset_names and len(cfg.dataset_names) > 1:
        for ds_name in cfg.dataset_names:
            print(f"\n=== Running dataset: {ds_name} ===")
            run_pipeline(ds_name)
    else:
        run_pipeline()
