# -*- coding: utf-8 -*-
"""
End-to-end experiment pipeline:
1) generate raw multi-source dataset via generate_bluesky_dataset.py (Python 3.12 subprocess)
2) process confidence via transfer_confidence.py API
3) run droneFusion no-GPS fusion (passive detection trusted tracks)
4) build GPS report tracks + inject anomalies
5) run graph-feature matching anomaly detector (uavMatch/match.py)

Outputs are written under WORK_ROOT.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG (all tunable params here)
# ============================================================

CFG: Dict[str, Any] = {
    # -------- paths --------
    "WORK_ROOT": "drone_sys/app/core/uavMatch/exp_bluesky_100x120_match",
    "RAW_DATASET_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_100x120_match/raw_dataset",
    "PROCESSED_DATASET_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_100x120_match/processed_dataset",
    "OUTPUT_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_100x120_match/output",
    "GENERATOR_PYTHON": r"C:\Python312\python.exe",  # generator needs Py>=3.10 for vendored BlueSky code
    "GENERATOR_USE_VENDORED_BLUESKY": True,
    "BLUESKY_VENDORED_ROOT": "drone_sys/app/dependencies/bluesky",

    # -------- stage control --------
    "RUN_GENERATE": True,
    "RUN_TRANSFER_CONFIDENCE": True,
    "RUN_FUSION": True,
    "RUN_MATCH": True,
    "RUN_PLOT_TRAJ_COMPARE": True,
    "SKIP_IF_EXISTS": True,

    # -------- dataset generation target --------
    "SEED": 20260225,
    "UAV_COUNT": 100,
    "DURATION_S": 120.0,
    "TRUTH_DT_S": 1.0,
    "BATCH_PREFIX": "batch",
    "BATCH_SIZE": 100,
    "GENERATOR_WORKERS": 4,
    "GENERATOR_USE_MULTIPROCESSING": True,
    "START_EPOCH": 1700000000,

    # Improve passive sensing side (stronger radar than default)
    "RADAR_RATE_HZ": 3.0,
    "RADAR_ERROR_SCALE": 0.85,  # < default 1.22 means cleaner radar
    "GENERATOR_CFG_DEEP_OVERRIDE": {},  # deep-merge into generate_bluesky_dataset.default_config()

    # Scenario mix (slightly richer than default but stable)
    "SCENARIO_MIX": {"A": 0.35, "B": 0.30, "C": 0.18, "D": 0.12, "E": 0.05},

    # -------- transfer confidence --------
    "TRANSFER_WORKERS": 8,
    "TRANSFER_SHUFFLE_SEED": 2026,

    # -------- fusion/no-GPS inference --------
    "FUSION_MODEL_PATH": "drone_sys/app/core/droneFusion/model_result/graph_fusion_model_v2.8.pt",
    "FUSION_NORM_PATH": "drone_sys/app/core/droneFusion/model_result/graph_norm_v2.8.pth",
    "FUSION_DISABLE_GPS": True,
    "FUSION_SAVE_FIG": False,
    "MAX_FUSE_UAVS": 0,  # 0 = all
    "ALIGN_TOLERANCE_S_FOR_REPORT": 0.55,
    "PASSIVE_TRACK_PREFIX": "P",
    "PASSIVE_CONFIDENCE_DEFAULT": 0.92,
    "PASSIVE_CONFIDENCE_MIN": 0.55,

    # -------- anomaly injection ratios (based on physical UAV set) --------
    "RATIO_UNREPORTED": 0.12,  # no GPS report for these physical UAVs
    "RATIO_DRIFT": 0.10,
    "RATIO_DEVIATION": 0.10,
    "RATIO_CONSISTENT_MIN": 0.50,  # keep at least this many normal
    "FALSE_REPORT_COUNT": 8,  # extra report-only fake tracks
    "ANOMALY_START_FRAC": 0.55,

    # Drift anomaly (GPS report drift relative to passive)
    "DRIFT_GROWTH_M_PER_S": 6.5,
    "DRIFT_BASE_OFFSET_M": 30.0,
    "DRIFT_DIRECTION_JITTER_DEG": 20.0,

    # Deviation anomaly (GPS path deflection + heading mismatch)
    "DEVIATION_HEADING_ROT_DEG": 55.0,
    "DEVIATION_LATERAL_GROWTH_M_PER_S": 4.5,
    "DEVIATION_BASE_LATERAL_M": 20.0,

    # False report synthesis
    "FALSE_REPORT_OFFSET_RANGE_M": [1800.0, 4500.0],
    "FALSE_REPORT_CONFIDENCE": 0.88,
    "FALSE_REPORT_ID_PREFIX": "FAKE_RPT_",

    # -------- matching config overrides --------
    "MATCH_CONFIG_OVERRIDES": {
        "PRINT_DETAIL": False,
        "WINDOW_DURATION_S": 4.0,
        "MATCH_ACTIVE_GAP_S": 2.5,
        "GRAPH_KNN_FUSION": 3,
        "GRAPH_KNN_REPORT": 3,
        # Tuned for ~100 targets and fusion error around 20-30m:
        # prioritize recall while keeping pair precision high.
        "TOPK_PER_FUSION": 20,
        "MAX_CANDIDATE_PAIRS_FOR_AFFINITY": 5000,
        "CANDIDATE_RADIUS_M": 2500.0,
        "SIGMA_NODE_POS_M": 280.0,
        "SIGMA_NODE_VEL_MPS": 55.0,
        "SIGMA_NODE_PATH_M": 1000.0,
        "SIGMA_EDGE_DP_M": 1200.0,
        "SIGMA_EDGE_DV_MPS": 80.0,
        "AFFINITY_EDGE_WEIGHT": 0.08,
        "MIN_MATCH_NODE_SCORE": 0.002,
        "MIN_MATCH_PAIR_SCORE": 0.005,
        "PAIR_SCORE_THRESHOLD": 0.20,
        "DRIFT_OFFSET_M": 180.0,
        "DEVIATION_OFFSET_M": 140.0,
        "DEVIATION_HEADING_DEG": 30.0,
        "DEVIATION_GROWTH_FRAMES": 2,
        "VOTE_WINDOW_FRAMES": 5,
        "VOTE_MIN_OBS": 3,
        "VOTE_MIN_TRUE_COUNT": 3,
        "VOTE_TRIGGER_RATIO": 0.6,
    },

    # -------- misc --------
    "SAVE_INTERMEDIATE_CSV": True,
    "SAVE_MATCH_RESULTS_JSON": True,
    "SAVE_TRAJ_PLOTS": True,
    "TRAJ_PLOT_DIRNAME": "traj_compare_per_uav",
    "TRAJ_PLOT_DPI": 130,
    "TRAJ_PLOT_FIGSIZE": [7.2, 6.2],
    "TRAJ_PLOT_LIMIT": 0,  # 0 = all
    "TRAJ_PLOT_OVERWRITE": True,
    "TRAJ_TOPN_OVERLAY_N": 20,
    "TRAJ_TOPN_OVERLAY_FILENAME": "traj_compare_top20_overlay.png",
    "TRAJ_TOPN_OVERLAY_DPI": 140,
    "TRAJ_TOPN_OVERLAY_FIGSIZE": [11.5, 9.0],
    "TRAJ_COLOR_FUSED": "#1f77b4",
    "TRAJ_COLOR_REPORT": "#d62728",
    "TRAJ_COLOR_TRUTH": "#2ca02c",
    "VERBOSE": True,
}


# ============================================================
# Path/bootstrap helpers
# ============================================================


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
CORE_ROOT = REPO_ROOT / "drone_sys" / "app" / "core"
DATASET_BUILDER_DIR = CORE_ROOT / "datasetBuilder"
DRONE_FUSION_DIR = CORE_ROOT / "droneFusion"

# `droneFusion/inference.py` uses local imports (`from dataset import ...`)
if str(DRONE_FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_FUSION_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EARTH_R_M = 6378137.0
EPS = 1e-12


def _p(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log(msg: str) -> None:
    if bool(CFG.get("VERBOSE", True)):
        print(msg)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _pick_existing_python(py_path: str) -> str:
    p = _p(py_path)
    if p.exists():
        return str(p)
    return sys.executable


def _normalize_modality_name(name: str) -> str:
    if name == "5g_a":
        return "fiveg"
    return name


def _safe_numeric(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _angle_wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _meters_to_deg_lat(m: float) -> float:
    return float(m / 111320.0)


def _meters_to_deg_lon(m: float, lat_deg: float) -> float:
    return float(m / (111320.0 * math.cos(math.radians(lat_deg)) + 1e-12))


def _nearest_truth_index(ts_obs: np.ndarray, ts_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(ts_truth) == 0 or len(ts_obs) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=float)
    idx = np.searchsorted(ts_truth, ts_obs)
    idx_lo = np.clip(idx - 1, 0, len(ts_truth) - 1)
    idx_hi = np.clip(idx, 0, len(ts_truth) - 1)
    d_lo = np.abs(ts_obs - ts_truth[idx_lo])
    d_hi = np.abs(ts_obs - ts_truth[idx_hi])
    choose_hi = d_hi < d_lo
    best_idx = np.where(choose_hi, idx_hi, idx_lo).astype(np.int64)
    best_diff = np.where(choose_hi, d_hi, d_lo).astype(float)
    return best_idx, best_diff


@dataclass
class FusedTrackResult:
    true_uav_id: str
    timestamps: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    confidence: np.ndarray
    diag: Dict[str, Any]


@dataclass
class ProcessedBatchData:
    batch_name: str
    batch_dir: Path
    truth_df: pd.DataFrame
    mod_frames: Dict[str, pd.DataFrame]


# ============================================================
# Generic column/track helpers
# ============================================================


def _deep_update_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update_dict(base[k], v)
        else:
            base[k] = v
    return base


def _detect_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_id_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df, ["uav_id", "id", "track_id", "tid", "target_id"])


def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df, ["timestamp", "time", "ts", "t"])


def _ensure_numeric_time(df: pd.DataFrame, time_col: str) -> pd.Series:
    s = pd.to_numeric(df[time_col], errors="coerce")
    if s.notna().any():
        return s.astype(float)
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt.notna().any():
        return (dt.astype("int64") / 1e9).astype(float)
    raise ValueError(f"Failed to parse time column: {time_col}")


def _filter_valid_llh(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for c in ["lat", "lon"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "alt" in out.columns:
        out["alt"] = pd.to_numeric(out["alt"], errors="coerce")
    else:
        out["alt"] = 0.0
    if "missing_flag" in out.columns:
        miss = pd.to_numeric(out["missing_flag"], errors="coerce").fillna(0).astype(int)
        out = out[miss == 0]
    out = out[np.isfinite(pd.to_numeric(out["lat"], errors="coerce")) & np.isfinite(pd.to_numeric(out["lon"], errors="coerce"))]
    return out.reset_index(drop=True)


def _enu_to_llh(
    e: np.ndarray,
    n: np.ndarray,
    u: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = ref_lat + np.degrees(n / EARTH_R_M)
    lon = ref_lon + np.degrees(e / (EARTH_R_M * math.cos(math.radians(ref_lat)) + EPS))
    alt = ref_alt + u
    return lat.astype(float), lon.astype(float), alt.astype(float)


def _latlonalt_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> np.ndarray:
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    east = dlon * EARTH_R_M * math.cos(math.radians(ref_lat))
    north = dlat * EARTH_R_M
    up = alt - ref_alt
    return np.stack([east, north, up], axis=1)


def _estimate_velocity_from_llh(
    ts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(ts))
    if n == 0:
        z = np.zeros((0,), dtype=float)
        return z, z, z
    if n == 1:
        z = np.zeros((1,), dtype=float)
        return z, z, z

    ts = np.asarray(ts, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    enu = _latlonalt_to_enu(lat, lon, alt, float(lat[0]), float(lon[0]), float(alt[0]))
    e_arr = enu[:, 0]
    n_arr = enu[:, 1]
    u_arr = enu[:, 2]

    def _grad(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=float)
        dt01 = max(ts[1] - ts[0], 1e-3)
        out[0] = (arr[1] - arr[0]) / dt01
        for i in range(1, len(arr) - 1):
            dtt = max(ts[i + 1] - ts[i - 1], 1e-3)
            out[i] = (arr[i + 1] - arr[i - 1]) / dtt
        dtn = max(ts[-1] - ts[-2], 1e-3)
        out[-1] = (arr[-1] - arr[-2]) / dtn
        return out

    return _grad(e_arr), _grad(n_arr), _grad(u_arr)


def _recompute_velocity_columns(
    df: pd.DataFrame,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    alt_col: str = "alt",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for c in ["vx", "vy", "vz"]:
            if c not in out.columns:
                out[c] = pd.Series(dtype=float)
        return out
    ts = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(out[alt_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vx, vy, vz = _estimate_velocity_from_llh(ts, lat, lon, alt)
    out["vx"] = vx
    out["vy"] = vy
    out["vz"] = vz
    return out


def _standardize_track_df(
    df: pd.DataFrame,
    *,
    out_id: Optional[str] = None,
    default_confidence: float = 0.8,
    force_time_col: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence"])

    id_col = _detect_id_col(df)
    time_col = force_time_col or _detect_time_col(df)
    if id_col is None or time_col is None:
        raise ValueError("track dataframe missing id/time columns")

    out = pd.DataFrame()
    out["id"] = str(out_id) if out_id is not None else df[id_col].astype(str)
    out["time"] = _ensure_numeric_time(df, time_col)
    for c in ["lat", "lon", "alt"]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else (0.0 if c == "alt" else np.nan)

    out["vx"] = pd.to_numeric(df[_detect_col(df, ["vx", "ve", "vel_x"])], errors="coerce") if _detect_col(df, ["vx", "ve", "vel_x"]) else np.nan
    out["vy"] = pd.to_numeric(df[_detect_col(df, ["vy", "vn", "vel_y"])], errors="coerce") if _detect_col(df, ["vy", "vn", "vel_y"]) else np.nan
    out["vz"] = pd.to_numeric(df[_detect_col(df, ["vz", "vu", "vel_z"])], errors="coerce") if _detect_col(df, ["vz", "vu", "vel_z"]) else np.nan
    conf_col = _detect_col(df, ["confidence", "quality", "score"])
    out["confidence"] = pd.to_numeric(df[conf_col], errors="coerce").fillna(default_confidence) if conf_col else float(default_confidence)

    out = out[np.isfinite(out["time"]) & np.isfinite(out["lat"]) & np.isfinite(out["lon"])].copy()
    out["alt"] = out["alt"].fillna(0.0)
    out["confidence"] = out["confidence"].clip(0.0, 1.0)
    out = out.sort_values(["id", "time"]).reset_index(drop=True)
    if not (np.isfinite(out["vx"]).all() and np.isfinite(out["vy"]).all() and np.isfinite(out["vz"]).all()):
        fixed_parts: List[pd.DataFrame] = []
        for tid, g in out.groupby("id", sort=False):
            fixed_parts.append(_recompute_velocity_columns(g.copy().reset_index(drop=True)))
        out = pd.concat(fixed_parts, ignore_index=True) if fixed_parts else out
    return out


def _discover_batch_leaf_dirs(root: Path, batch_prefix: str) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith(batch_prefix) and (p / "truth.csv").exists()]


def _load_processed_batches(processed_root: Path, batch_prefix: str) -> List[ProcessedBatchData]:
    batch_dirs = _discover_batch_leaf_dirs(processed_root, batch_prefix)
    if not batch_dirs and (processed_root / "truth.csv").exists():
        batch_dirs = [processed_root]
    mod_files = {"gps": "gps.csv", "radar": "radar.csv", "fiveg": "5g_a.csv", "tdoa": "tdoa.csv", "acoustic": "acoustic.csv"}
    out: List[ProcessedBatchData] = []
    for bdir in batch_dirs:
        truth_df = pd.read_csv(bdir / "truth.csv")
        mod_frames = {m: (pd.read_csv(bdir / fn) if (bdir / fn).exists() else pd.DataFrame()) for m, fn in mod_files.items()}
        out.append(ProcessedBatchData(batch_name=bdir.name, batch_dir=bdir, truth_df=truth_df, mod_frames=mod_frames))
    return out


def _load_truth_tracks_for_plot(processed_root: Path, raw_root: Path, batch_prefix: str) -> pd.DataFrame:
    roots = [processed_root, raw_root]
    parts: List[pd.DataFrame] = []
    seen_files: set = set()
    for root in roots:
        if root is None or (not Path(root).exists()):
            continue
        batch_dirs = _discover_batch_leaf_dirs(Path(root), batch_prefix)
        if not batch_dirs and (Path(root) / "truth.csv").exists():
            batch_dirs = [Path(root)]
        for bdir in batch_dirs:
            fp = (bdir / "truth.csv").resolve()
            if str(fp) in seen_files or (not fp.exists()):
                continue
            seen_files.add(str(fp))
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            std = _standardize_track_df(df, default_confidence=1.0)
            if std.empty:
                continue
            std["true_uav_id"] = std["id"].astype(str)
            std["track_source"] = "truth"
            parts.append(std)
    if not parts:
        return pd.DataFrame(columns=["id", "true_uav_id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence", "track_source"])
    out = pd.concat(parts, ignore_index=True)
    # Deduplicate if the same truth rows appear in both processed/raw roots.
    dedup_cols = [c for c in ["id", "time", "lat", "lon", "alt"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
    return out.sort_values(["time", "id"]).reset_index(drop=True)


def _insert_vendored_bluesky_if_needed() -> None:
    if not bool(CFG.get("GENERATOR_USE_VENDORED_BLUESKY", True)):
        return
    vendored = _p(str(CFG.get("BLUESKY_VENDORED_ROOT", "")))
    if vendored.exists() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))


# ============================================================
# Stage 1: generate raw BlueSky dataset
# ============================================================


def stage_generate_raw_dataset(raw_root: Path) -> Dict[str, Any]:
    if bool(CFG.get("SKIP_IF_EXISTS", True)):
        if _discover_batch_leaf_dirs(raw_root, str(CFG["BATCH_PREFIX"])) or (raw_root / "truth.csv").exists():
            _log(f"[generate] skip (existing dataset): {raw_root}")
            return {"skipped": True}

    _insert_vendored_bluesky_if_needed()
    from drone_sys.app.core.datasetBuilder import generate_bluesky_dataset as gen

    cfg = deepcopy(gen.default_config())
    cfg["seed"] = int(CFG["SEED"])
    cfg["output_dir"] = str(raw_root)
    cfg["worker_num"] = int(CFG["GENERATOR_WORKERS"])
    cfg["use_multiprocessing"] = bool(CFG["GENERATOR_USE_MULTIPROCESSING"])
    cfg["batching"] = {
        "enabled": True,
        "batch_size": int(CFG["BATCH_SIZE"]),
        "folder_prefix": str(CFG["BATCH_PREFIX"]),
    }
    cfg["simulation"]["uav_count"] = int(CFG["UAV_COUNT"])
    cfg["simulation"]["duration_s"] = float(CFG["DURATION_S"])
    cfg["simulation"]["truth_dt_s"] = float(CFG["TRUTH_DT_S"])
    cfg["simulation"]["start_epoch"] = int(CFG["START_EPOCH"])
    cfg["scenario_mix"] = dict(CFG["SCENARIO_MIX"])
    cfg.setdefault("modalities", {}).setdefault("radar", {})
    cfg["modalities"]["radar"]["rate_hz"] = float(CFG["RADAR_RATE_HZ"])
    cfg.setdefault("modality_error_scale", {})
    cfg["modality_error_scale"]["radar"] = float(CFG["RADAR_ERROR_SCALE"])
    extra_gen_override = CFG.get("GENERATOR_CFG_DEEP_OVERRIDE", {})
    if isinstance(extra_gen_override, dict) and len(extra_gen_override) > 0:
        _deep_update_dict(cfg, deepcopy(extra_gen_override))

    _ensure_dir(raw_root)
    _json_dump(_to_builtin(cfg), raw_root / "generator_config.used.json")
    _log(f"[generate] running BlueSky dataset generation -> {raw_root}")
    gen.run(cfg)
    return {"skipped": False, "output_dir": str(raw_root)}


# ============================================================
# Stage 2: transfer confidence
# ============================================================


def stage_transfer_confidence(raw_root: Path, processed_root: Path) -> Dict[str, Any]:
    if bool(CFG.get("SKIP_IF_EXISTS", True)):
        if _discover_batch_leaf_dirs(processed_root, str(CFG["BATCH_PREFIX"])) or (processed_root / "truth.csv").exists():
            _log(f"[transfer] skip (existing processed dataset): {processed_root}")
            return {"skipped": True}

    from drone_sys.app.core.datasetBuilder import transfer_confidence as tc

    _ensure_dir(processed_root)
    _log(f"[transfer] running confidence transform: {raw_root} -> {processed_root}")
    summary = tc.process_dataset_unit(
        dataset_dir=str(raw_root),
        output_dir=str(processed_root),
        cfg=tc.default_cfg(),
        batch_prefix=str(CFG["BATCH_PREFIX"]),
        worker_num=int(CFG["TRANSFER_WORKERS"]),
    )
    _json_dump(_to_builtin(summary), processed_root / "confidence_transfer.summary.json")
    return {"skipped": False, "summary": summary}


# ============================================================
# Stage 3: no-GPS fusion (passive detection trusted tracks)
# ============================================================


def _configure_fusion_router_model_paths() -> Any:
    from drone_sys.app.routers import fusion as fusion_router

    model_path = _p(str(CFG["FUSION_MODEL_PATH"]))
    norm_path = _p(str(CFG["FUSION_NORM_PATH"]))
    if not model_path.exists():
        raise FileNotFoundError(f"Fusion model not found: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Fusion norm not found: {norm_path}")

    fusion_router._MODEL_PATH = model_path
    fusion_router._NORM_PATH = norm_path
    try:
        fusion_router._load_runtime_bundle.cache_clear()
    except Exception:
        pass
    return fusion_router


def _collect_gps_frames_by_uav(batches: Sequence[ProcessedBatchData]) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, List[pd.DataFrame]] = {}
    for batch in batches:
        gps_df = batch.mod_frames.get("gps", pd.DataFrame())
        if gps_df.empty:
            continue
        id_col = _detect_id_col(gps_df)
        time_col = _detect_time_col(gps_df)
        if id_col is None or time_col is None:
            continue
        tmp = gps_df.copy()
        tmp[id_col] = tmp[id_col].astype(str)
        tmp[time_col] = _ensure_numeric_time(tmp, time_col)
        for uid, g in tmp.groupby(id_col, sort=False):
            parts.setdefault(str(uid), []).append(g.copy())
    out: Dict[str, pd.DataFrame] = {}
    for uid, gs in parts.items():
        merged = pd.concat(gs, ignore_index=True) if len(gs) > 1 else gs[0].copy()
        tcol = _detect_time_col(merged)
        if tcol:
            merged = merged.sort_values(tcol).reset_index(drop=True)
        out[uid] = merged
    return out


def _fuse_one_uav_no_gps(
    fusion_router: Any,
    truth_u: pd.DataFrame,
    mod_frames_u: Dict[str, pd.DataFrame],
    true_uid: str,
    batch_name: str,
) -> Optional[FusedTrackResult]:
    req_frames = {
        "radar": mod_frames_u.get("radar", pd.DataFrame()),
        "fiveg": mod_frames_u.get("fiveg", pd.DataFrame()),
        "tdoa": mod_frames_u.get("tdoa", pd.DataFrame()),
        "acoustic": mod_frames_u.get("acoustic", pd.DataFrame()),
    }
    try:
        out_rows = fusion_router._run_model_inference(truth_df=truth_u, mod_frames_by_request_name=req_frames)
    except Exception as ex:
        _log(f"[fusion] uav={true_uid} failed: {ex}")
        return None
    if not out_rows:
        return None

    out_df = pd.DataFrame(out_rows)
    if out_df.empty or "timestamp" not in out_df.columns:
        return None
    out_df["timestamp"] = pd.to_numeric(out_df["timestamp"], errors="coerce")
    for c in ["lat", "lon", "alt"]:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce")
    out_df = out_df[np.isfinite(out_df["timestamp"]) & np.isfinite(out_df["lat"]) & np.isfinite(out_df["lon"])].copy()
    out_df["alt"] = out_df["alt"].fillna(0.0)
    out_df = out_df.sort_values("timestamp").reset_index(drop=True)
    if out_df.empty:
        return None

    ts = out_df["timestamp"].to_numpy(dtype=float)
    lat = out_df["lat"].to_numpy(dtype=float)
    lon = out_df["lon"].to_numpy(dtype=float)
    alt = out_df["alt"].to_numpy(dtype=float)
    vx, vy, vz = _estimate_velocity_from_llh(ts, lat, lon, alt)
    conf_default = max(float(CFG["PASSIVE_CONFIDENCE_MIN"]), min(1.0, float(CFG["PASSIVE_CONFIDENCE_DEFAULT"])))
    conf = np.full(len(out_df), conf_default, dtype=float)

    return FusedTrackResult(
        true_uav_id=str(true_uid),
        timestamps=ts,
        lat=lat,
        lon=lon,
        alt=alt,
        vx=vx,
        vy=vy,
        vz=vz,
        confidence=conf,
        diag={
            "batch": batch_name,
            "fused_points": int(len(out_df)),
            "modal_rows": {k: int(len(v)) for k, v in req_frames.items()},
        },
    )


def stage_fusion_no_gps(processed_root: Path) -> Tuple[List[FusedTrackResult], Dict[str, pd.DataFrame], Dict[str, Any]]:
    batches = _load_processed_batches(processed_root, str(CFG["BATCH_PREFIX"]))
    if not batches:
        raise FileNotFoundError(f"No processed batches found under: {processed_root}")

    fusion_router = _configure_fusion_router_model_paths()
    _, _, _, _, _, _, _, runtime = fusion_router._load_runtime_bundle()
    runtime_info = {k: runtime.get(k) for k in ["window_size", "stride", "align_tolerance_s", "modalities", "in_dim"]}
    _log(f"[fusion] runtime={runtime_info}")

    gps_frames_by_uid = _collect_gps_frames_by_uav(batches)
    fused_results: List[FusedTrackResult] = []
    total_seen = 0
    max_uavs = int(CFG.get("MAX_FUSE_UAVS", 0))

    for batch in batches:
        truth_df = batch.truth_df.copy()
        id_col = _detect_id_col(truth_df)
        time_col = _detect_time_col(truth_df)
        if id_col is None or time_col is None:
            raise ValueError(f"truth.csv missing id/time columns in {batch.batch_dir}")
        truth_df[id_col] = truth_df[id_col].astype(str)
        truth_df[time_col] = _ensure_numeric_time(truth_df, time_col)

        mod_group_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        for m, mdf in batch.mod_frames.items():
            if mdf is None or mdf.empty:
                mod_group_cache[m] = {}
                continue
            mid = _detect_id_col(mdf)
            mtime = _detect_time_col(mdf)
            if mid is None or mtime is None:
                mod_group_cache[m] = {}
                continue
            tmp = mdf.copy()
            tmp[mid] = tmp[mid].astype(str)
            tmp[mtime] = _ensure_numeric_time(tmp, mtime)
            groups: Dict[str, pd.DataFrame] = {}
            for uid, g in tmp.groupby(mid, sort=False):
                groups[str(uid)] = g.sort_values(mtime).reset_index(drop=True)
            mod_group_cache[m] = groups

        for uid in sorted(truth_df[id_col].unique().tolist()):
            if max_uavs > 0 and total_seen >= max_uavs:
                break
            total_seen += 1
            truth_u = truth_df[truth_df[id_col] == uid].copy().sort_values(time_col).reset_index(drop=True)
            mod_u = {m: mod_group_cache.get(m, {}).get(uid, pd.DataFrame()) for m in ["radar", "fiveg", "tdoa", "acoustic"]}
            res = _fuse_one_uav_no_gps(
                fusion_router=fusion_router,
                truth_u=truth_u,
                mod_frames_u=mod_u,
                true_uid=str(uid),
                batch_name=batch.batch_name,
            )
            if res is not None:
                fused_results.append(res)
                if len(fused_results) % 10 == 0:
                    _log(f"[fusion] completed={len(fused_results)} UAVs")
        if max_uavs > 0 and total_seen >= max_uavs:
            break

    if not fused_results:
        raise RuntimeError("No fused tracks produced.")

    return fused_results, gps_frames_by_uid, {
        "batch_count": len(batches),
        "fused_uav_count": len(fused_results),
        "gps_uav_count": len(gps_frames_by_uid),
        "runtime": runtime_info,
    }


# ============================================================
# Stage 4: build GPS reports + inject anomalies
# ============================================================


def _align_gps_report_to_fused_track(
    true_uid: str,
    fused: FusedTrackResult,
    gps_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    empty_cols = ["id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence"]
    if gps_df is None or gps_df.empty:
        return pd.DataFrame(columns=empty_cols)

    g = _filter_valid_llh(gps_df)
    if g.empty:
        return pd.DataFrame(columns=empty_cols)

    time_col = _detect_time_col(g)
    if time_col is None:
        return pd.DataFrame(columns=empty_cols)
    g[time_col] = _ensure_numeric_time(g, time_col)
    g = g.sort_values(time_col).reset_index(drop=True)

    ts_g = g[time_col].to_numpy(dtype=float)
    ts_ref = fused.timestamps.astype(float)
    nearest_idx, nearest_diff = _nearest_truth_index(ts_ref, ts_g)
    keep = nearest_diff <= float(CFG["ALIGN_TOLERANCE_S_FOR_REPORT"])
    if not np.any(keep):
        return pd.DataFrame(columns=empty_cols)

    vx_col = _detect_col(g, ["vx", "ve", "vel_x"])
    vy_col = _detect_col(g, ["vy", "vn", "vel_y"])
    vz_col = _detect_col(g, ["vz", "vu", "vel_z"])
    conf_col = _detect_col(g, ["confidence", "quality", "score"])

    out = pd.DataFrame(
        {
            "id": str(true_uid),
            "time": ts_ref[keep],
            "src_timestamp": ts_g[nearest_idx[keep]],
            "lat": pd.to_numeric(g["lat"], errors="coerce").to_numpy(dtype=float)[nearest_idx[keep]],
            "lon": pd.to_numeric(g["lon"], errors="coerce").to_numpy(dtype=float)[nearest_idx[keep]],
            "alt": (pd.to_numeric(g["alt"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "alt" in g.columns else np.zeros(len(g), dtype=float))[nearest_idx[keep]],
            "vx": (pd.to_numeric(g[vx_col], errors="coerce").to_numpy(dtype=float) if vx_col else np.full(len(g), np.nan, dtype=float))[nearest_idx[keep]],
            "vy": (pd.to_numeric(g[vy_col], errors="coerce").to_numpy(dtype=float) if vy_col else np.full(len(g), np.nan, dtype=float))[nearest_idx[keep]],
            "vz": (pd.to_numeric(g[vz_col], errors="coerce").to_numpy(dtype=float) if vz_col else np.full(len(g), np.nan, dtype=float))[nearest_idx[keep]],
            "confidence": (
                pd.to_numeric(g[conf_col], errors="coerce").fillna(0.85).to_numpy(dtype=float) if conf_col else np.full(len(g), 0.85, dtype=float)
            )[nearest_idx[keep]],
        }
    )
    out = out[np.isfinite(out["lat"]) & np.isfinite(out["lon"])].copy()
    out["confidence"] = out["confidence"].clip(0.0, 1.0)
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    if out.empty:
        return out
    if not (np.isfinite(out["vx"]).all() and np.isfinite(out["vy"]).all() and np.isfinite(out["vz"]).all()):
        out = _recompute_velocity_columns(out)
    return out


def _compute_case_counts(uids: List[str]) -> Dict[str, int]:
    n = len(uids)
    counts = {
        "unreported": int(round(float(CFG["RATIO_UNREPORTED"]) * n)),
        "drift": int(round(float(CFG["RATIO_DRIFT"]) * n)),
        "deviation": int(round(float(CFG["RATIO_DEVIATION"]) * n)),
    }
    min_consistent = int(math.ceil(float(CFG["RATIO_CONSISTENT_MIN"]) * n))
    max_anom = max(0, n - min_consistent)
    while sum(counts.values()) > max_anom:
        for k in ["deviation", "drift", "unreported"]:
            if sum(counts.values()) <= max_anom:
                break
            if counts[k] > 0:
                counts[k] -= 1
    counts["consistent"] = n - sum(counts.values())
    return counts


def _assign_scenarios(uids: List[str], seed: int) -> Dict[str, str]:
    shuffled = list(uids)
    rnd = random.Random(int(seed))
    rnd.shuffle(shuffled)
    counts = _compute_case_counts(shuffled)
    out: Dict[str, str] = {}
    idx = 0
    for k in ["unreported", "drift", "deviation"]:
        for _ in range(counts[k]):
            if idx >= len(shuffled):
                break
            out[shuffled[idx]] = k
            idx += 1
    while idx < len(shuffled):
        out[shuffled[idx]] = "consistent"
        idx += 1
    return out


def _apply_drift_anomaly(track_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if track_df.empty or len(track_df) < 3:
        return track_df.copy()
    out = track_df.copy().sort_values("time").reset_index(drop=True)
    n = len(out)
    start_idx = min(n - 1, max(1, int(float(CFG["ANOMALY_START_FRAC"]) * n)))
    t0 = float(out.loc[start_idx, "time"])

    vx0 = _safe_float(out.loc[start_idx, "vx"], 0.0)
    vy0 = _safe_float(out.loc[start_idx, "vy"], 0.0)
    heading_deg = math.degrees(math.atan2(vy0, vx0)) if abs(vx0) + abs(vy0) > 1e-6 else 0.0
    rnd = random.Random(int(seed))
    theta = math.radians(heading_deg + rnd.uniform(-float(CFG["DRIFT_DIRECTION_JITTER_DEG"]), float(CFG["DRIFT_DIRECTION_JITTER_DEG"])))
    ux = math.cos(theta)
    uy = math.sin(theta)
    base_off = float(CFG["DRIFT_BASE_OFFSET_M"])
    growth = float(CFG["DRIFT_GROWTH_M_PER_S"])

    for i in range(start_idx, n):
        t = float(out.loc[i, "time"])
        dt = max(0.0, t - t0)
        off = base_off + growth * dt
        lat0 = float(out.loc[i, "lat"])
        lon0 = float(out.loc[i, "lon"])
        out.loc[i, "lat"] = lat0 + _meters_to_deg_lat(off * uy)
        out.loc[i, "lon"] = lon0 + _meters_to_deg_lon(off * ux, lat0)
    return _recompute_velocity_columns(out)


def _apply_deviation_anomaly(track_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if track_df.empty or len(track_df) < 4:
        return _apply_drift_anomaly(track_df, seed)
    out = track_df.copy().sort_values("time").reset_index(drop=True)
    n = len(out)
    start_idx = min(n - 2, max(1, int(float(CFG["ANOMALY_START_FRAC"]) * n)))
    t0 = float(out.loc[start_idx, "time"])

    ref_lat = float(out.loc[start_idx, "lat"])
    ref_lon = float(out.loc[start_idx, "lon"])
    ref_alt = float(out.loc[start_idx, "alt"])
    post = out.iloc[start_idx:].copy().reset_index(drop=True)
    ts = pd.to_numeric(post["time"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(post["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(post["lon"], errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(post["alt"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    enu = _latlonalt_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    rnd = random.Random(int(seed))
    rot_deg = float(CFG["DEVIATION_HEADING_ROT_DEG"]) * (1.0 if rnd.random() >= 0.5 else -1.0)
    rot = math.radians(rot_deg)
    c = math.cos(rot)
    s = math.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=float)
    xy = enu[:, :2] @ R.T

    hvec = xy[1] - xy[0] if len(xy) >= 2 else np.array([1.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(hvec))
    if norm < 1e-6:
        hvec = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    hvec = hvec / norm
    perp = np.array([-hvec[1], hvec[0]], dtype=float)

    dt = np.maximum(0.0, ts - t0)
    lateral = float(CFG["DEVIATION_BASE_LATERAL_M"]) + float(CFG["DEVIATION_LATERAL_GROWTH_M_PER_S"]) * dt
    xy = xy + lateral[:, None] * perp[None, :]

    new_lat, new_lon, new_alt = _enu_to_llh(xy[:, 0], xy[:, 1], enu[:, 2], ref_lat, ref_lon, ref_alt)
    out.loc[start_idx:, "lat"] = new_lat
    out.loc[start_idx:, "lon"] = new_lon
    out.loc[start_idx:, "alt"] = new_alt
    return _recompute_velocity_columns(out)


def _synthesize_false_reports(
    fused_results: Sequence[FusedTrackResult],
    count: int,
    seed: int,
) -> List[pd.DataFrame]:
    if count <= 0 or not fused_results:
        return []
    rnd = random.Random(int(seed))
    dmin, dmax = [float(x) for x in CFG["FALSE_REPORT_OFFSET_RANGE_M"]]
    out: List[pd.DataFrame] = []
    for i in range(int(count)):
        base = fused_results[rnd.randrange(len(fused_results))]
        theta = rnd.uniform(0.0, 2.0 * math.pi)
        dist = rnd.uniform(dmin, dmax)
        dx = dist * math.cos(theta)
        dy = dist * math.sin(theta)
        fake_id = f"{CFG['FALSE_REPORT_ID_PREFIX']}{i + 1:03d}"

        lat = base.lat.copy()
        lon = base.lon.copy()
        alt = base.alt.copy()
        for j in range(len(lat)):
            lat0 = float(lat[j])
            lon0 = float(lon[j])
            jx = dx + rnd.uniform(-30.0, 30.0)
            jy = dy + rnd.uniform(-30.0, 30.0)
            lat[j] = lat0 + _meters_to_deg_lat(jy)
            lon[j] = lon0 + _meters_to_deg_lon(jx, lat0)

        fake_df = pd.DataFrame(
            {
                "id": fake_id,
                "time": base.timestamps.astype(float),
                "lat": lat.astype(float),
                "lon": lon.astype(float),
                "alt": alt.astype(float),
                "confidence": float(CFG["FALSE_REPORT_CONFIDENCE"]),
                "synthetic_false_report": 1,
                "source_true_uav_ref": str(base.true_uav_id),
            }
        )
        out.append(_recompute_velocity_columns(fake_df))
    return out


def stage_build_tracks_and_inject_anomalies(
    fused_results: Sequence[FusedTrackResult],
    gps_frames_by_uid: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not fused_results:
        raise RuntimeError("No fused results to build tracks.")

    fused_by_uid = {r.true_uav_id: r for r in fused_results}
    case_map = _assign_scenarios(list(fused_by_uid.keys()), seed=int(CFG["SEED"]) + 101)
    aligned_report_cache: Dict[str, pd.DataFrame] = {}
    for uid, fused in fused_by_uid.items():
        aligned_report_cache[uid] = _align_gps_report_to_fused_track(uid, fused, gps_frames_by_uid.get(uid))

    report_parts: List[pd.DataFrame] = []
    assignment_rows: List[Dict[str, Any]] = []
    for idx, fused in enumerate(fused_results, start=1):
        uid = fused.true_uav_id
        case = case_map.get(uid, "consistent")
        rep = aligned_report_cache.get(uid, pd.DataFrame()).copy()
        if case == "unreported":
            rep = rep.iloc[0:0].copy()
        elif case == "drift":
            rep = _apply_drift_anomaly(rep, seed=int(CFG["SEED"]) + 2000 + idx)
        elif case == "deviation":
            rep = _apply_deviation_anomaly(rep, seed=int(CFG["SEED"]) + 3000 + idx)

        if not rep.empty:
            rep["scenario_case"] = case
            rep["source_true_uav_id"] = uid
            report_parts.append(rep)

        assignment_rows.append(
            {
                "true_uav_id": uid,
                "scenario_case": case,
                "aligned_report_points": int(len(aligned_report_cache.get(uid, pd.DataFrame()))),
                "final_report_points": int(len(rep)),
            }
        )

    false_reports = _synthesize_false_reports(
        fused_results=fused_results,
        count=int(CFG["FALSE_REPORT_COUNT"]),
        seed=int(CFG["SEED"]) + 4001,
    )
    for fr in false_reports:
        ff = fr.copy()
        ff["scenario_case"] = "false_report"
        ff["source_true_uav_id"] = ""
        report_parts.append(ff)

    report_df = pd.concat(report_parts, ignore_index=True) if report_parts else pd.DataFrame()
    if not report_df.empty:
        report_df["id"] = report_df["id"].astype(str)
        report_df["time"] = pd.to_numeric(report_df["time"], errors="coerce")
        for c in ["lat", "lon", "alt", "vx", "vy", "vz", "confidence"]:
            if c in report_df.columns:
                report_df[c] = pd.to_numeric(report_df[c], errors="coerce")
        report_df["alt"] = report_df["alt"].fillna(0.0)
        report_df["confidence"] = report_df["confidence"].fillna(float(CFG["FALSE_REPORT_CONFIDENCE"])).clip(0.0, 1.0)
        report_df = report_df[np.isfinite(report_df["time"]) & np.isfinite(report_df["lat"]) & np.isfinite(report_df["lon"])].copy()
        report_df = report_df.sort_values(["time", "id"]).reset_index(drop=True)

    passive_parts: List[pd.DataFrame] = []
    passive_map_rows: List[Dict[str, Any]] = []
    for idx, fused in enumerate(fused_results, start=1):
        passive_id = f"{CFG['PASSIVE_TRACK_PREFIX']}{idx:03d}"
        p = pd.DataFrame(
            {
                "id": passive_id,
                "time": fused.timestamps.astype(float),
                "lat": fused.lat.astype(float),
                "lon": fused.lon.astype(float),
                "alt": fused.alt.astype(float),
                "vx": fused.vx.astype(float),
                "vy": fused.vy.astype(float),
                "vz": fused.vz.astype(float),
                "confidence": fused.confidence.astype(float),
                "track_source": "passive_fusion",
            }
        )
        passive_parts.append(p)
        passive_map_rows.append(
            {
                "passive_track_id": passive_id,
                "true_uav_id": fused.true_uav_id,
                "scenario_case": case_map.get(fused.true_uav_id, "consistent"),
                "fused_points": int(len(fused.timestamps)),
                "batch": fused.diag.get("batch", ""),
            }
        )

    passive_df = pd.concat(passive_parts, ignore_index=True) if passive_parts else pd.DataFrame()
    if not passive_df.empty:
        passive_df = passive_df.sort_values(["time", "id"]).reset_index(drop=True)

    passive_map_df = pd.DataFrame(passive_map_rows).sort_values("passive_track_id").reset_index(drop=True)
    assignment_df = pd.DataFrame(assignment_rows).sort_values("true_uav_id").reset_index(drop=True)
    case_counts = assignment_df["scenario_case"].value_counts().to_dict() if not assignment_df.empty else {}
    summary = {
        "physical_uav_count": int(len(fused_results)),
        "case_counts": {str(k): int(v) for k, v in case_counts.items()},
        "false_report_track_count": int(len(false_reports)),
        "passive_rows": int(len(passive_df)),
        "report_rows": int(len(report_df)),
        "passive_track_count": int(passive_df["id"].nunique()) if not passive_df.empty else 0,
        "report_track_count": int(report_df["id"].nunique()) if not report_df.empty else 0,
    }
    return passive_df, report_df, passive_map_df, assignment_df, summary


def _fused_results_to_truth_df(fused_results: Sequence[FusedTrackResult]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fused in fused_results:
        g = pd.DataFrame(
            {
                "id": str(fused.true_uav_id),
                "true_uav_id": str(fused.true_uav_id),
                "time": fused.timestamps.astype(float),
                "lat": fused.lat.astype(float),
                "lon": fused.lon.astype(float),
                "alt": fused.alt.astype(float),
                "vx": fused.vx.astype(float),
                "vy": fused.vy.astype(float),
                "vz": fused.vz.astype(float),
                "confidence": fused.confidence.astype(float),
                "track_source": "fusion_no_gps",
                "source_batch": str(fused.diag.get("batch", "")),
            }
        )
        parts.append(g)
    if not parts:
        return pd.DataFrame(
            columns=["id", "true_uav_id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence", "track_source", "source_batch"]
        )
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["time", "id"]).reset_index(drop=True)


def _reconstruct_fused_truth_from_passive(passive_df: pd.DataFrame, passive_map_df: pd.DataFrame) -> pd.DataFrame:
    if passive_df is None or passive_df.empty or passive_map_df is None or passive_map_df.empty:
        return pd.DataFrame()
    p = passive_df.copy()
    m = passive_map_df.copy()
    p["id"] = p["id"].astype(str)
    m["passive_track_id"] = m["passive_track_id"].astype(str)
    m["true_uav_id"] = m["true_uav_id"].astype(str)
    out = p.merge(m[["passive_track_id", "true_uav_id", "scenario_case"]], left_on="id", right_on="passive_track_id", how="left")
    if out.empty:
        return pd.DataFrame()
    out["id"] = out["true_uav_id"].fillna(out["id"]).astype(str)
    out["track_source"] = out.get("track_source", pd.Series(["fusion_no_gps"] * len(out))).fillna("fusion_no_gps")
    cols = [c for c in ["id", "true_uav_id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence", "track_source", "scenario_case"] if c in out.columns]
    return out[cols].sort_values(["time", "id"]).reset_index(drop=True)


def _xy_offset_stats_m(fused_u: pd.DataFrame, report_u: pd.DataFrame) -> Dict[str, Any]:
    if fused_u.empty or report_u.empty:
        return {"overlap_points": 0, "mean_xy_m": None, "median_xy_m": None, "max_xy_m": None}
    fu = fused_u.copy()
    ru = report_u.copy()
    fu["time_key"] = pd.to_numeric(fu["time"], errors="coerce").round(6)
    ru["time_key"] = pd.to_numeric(ru["time"], errors="coerce").round(6)
    common = sorted(set(fu["time_key"].tolist()) & set(ru["time_key"].tolist()))
    if not common:
        return {"overlap_points": 0, "mean_xy_m": None, "median_xy_m": None, "max_xy_m": None}
    fidx = fu.set_index("time_key").loc[common]
    ridx = ru.set_index("time_key").loc[common]
    ref_lat = float(pd.to_numeric(fidx["lat"], errors="coerce").iloc[0])
    ref_lon = float(pd.to_numeric(fidx["lon"], errors="coerce").iloc[0])
    ref_alt = float(pd.to_numeric(fidx.get("alt", 0.0), errors="coerce").fillna(0.0).iloc[0]) if "alt" in fidx.columns else 0.0
    f_enu = _latlonalt_to_enu(
        pd.to_numeric(fidx["lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(fidx["lon"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(fidx.get("alt", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) if "alt" in fidx.columns else np.zeros(len(fidx)),
        ref_lat,
        ref_lon,
        ref_alt,
    )
    r_enu = _latlonalt_to_enu(
        pd.to_numeric(ridx["lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(ridx["lon"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(ridx.get("alt", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) if "alt" in ridx.columns else np.zeros(len(ridx)),
        ref_lat,
        ref_lon,
        ref_alt,
    )
    d = np.linalg.norm(f_enu[:, :2] - r_enu[:, :2], axis=1)
    return {
        "overlap_points": int(len(d)),
        "mean_xy_m": float(np.mean(d)) if len(d) else None,
        "median_xy_m": float(np.median(d)) if len(d) else None,
        "max_xy_m": float(np.max(d)) if len(d) else None,
    }


def stage_plot_fused_vs_report_per_uav(
    fused_truth_df: pd.DataFrame,
    report_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    if not bool(CFG.get("SAVE_TRAJ_PLOTS", True)):
        return {"skipped_by_flag": True}
    if fused_truth_df is None or fused_truth_df.empty:
        return {"skipped": True, "reason": "fused_truth_df empty"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = _ensure_dir(output_dir / str(CFG["TRAJ_PLOT_DIRNAME"]))
    fig_w, fig_h = [float(x) for x in CFG.get("TRAJ_PLOT_FIGSIZE", [7.2, 6.2])]
    dpi = int(CFG.get("TRAJ_PLOT_DPI", 130))
    overwrite = bool(CFG.get("TRAJ_PLOT_OVERWRITE", True))
    limit = int(CFG.get("TRAJ_PLOT_LIMIT", 0))

    fused = fused_truth_df.copy()
    fused_id_col = "true_uav_id" if "true_uav_id" in fused.columns else "id"
    fused[fused_id_col] = fused[fused_id_col].astype(str)
    fused["time"] = pd.to_numeric(fused["time"], errors="coerce")
    for c in ["lat", "lon", "alt"]:
        if c in fused.columns:
            fused[c] = pd.to_numeric(fused[c], errors="coerce")
    fused = fused[np.isfinite(fused["time"]) & np.isfinite(fused["lat"]) & np.isfinite(fused["lon"])].copy()

    rep = report_df.copy() if report_df is not None else pd.DataFrame()
    if not rep.empty:
        rep["id"] = rep["id"].astype(str)
        rep["time"] = pd.to_numeric(rep["time"], errors="coerce")
        for c in ["lat", "lon", "alt"]:
            if c in rep.columns:
                rep[c] = pd.to_numeric(rep[c], errors="coerce")
        rep = rep[np.isfinite(rep["time"]) & np.isfinite(rep["lat"]) & np.isfinite(rep["lon"])].copy()

    truth = truth_df.copy() if truth_df is not None else pd.DataFrame()
    if not truth.empty:
        truth["id"] = truth["id"].astype(str)
        truth["time"] = pd.to_numeric(truth["time"], errors="coerce")
        for c in ["lat", "lon", "alt"]:
            if c in truth.columns:
                truth[c] = pd.to_numeric(truth[c], errors="coerce")
        truth = truth[np.isfinite(truth["time"]) & np.isfinite(truth["lat"]) & np.isfinite(truth["lon"])].copy()

    case_map: Dict[str, str] = {}
    if assignment_df is not None and not assignment_df.empty and "true_uav_id" in assignment_df.columns and "scenario_case" in assignment_df.columns:
        case_map = dict(zip(assignment_df["true_uav_id"].astype(str), assignment_df["scenario_case"].astype(str)))

    uids = sorted(fused[fused_id_col].astype(str).unique().tolist())
    if limit > 0:
        uids = uids[:limit]

    stats_rows: List[Dict[str, Any]] = []
    saved = 0
    fused_color = str(CFG.get("TRAJ_COLOR_FUSED", "#1f77b4"))
    report_color = str(CFG.get("TRAJ_COLOR_REPORT", "#d62728"))
    truth_color = str(CFG.get("TRAJ_COLOR_TRUTH", "#2ca02c"))
    for idx, uid in enumerate(uids, start=1):
        fu = fused[fused[fused_id_col].astype(str) == uid].copy().sort_values("time").reset_index(drop=True)
        ru = rep[rep["id"].astype(str) == uid].copy().sort_values("time").reset_index(drop=True) if not rep.empty else pd.DataFrame()
        tu = truth[truth["id"].astype(str) == uid].copy().sort_values("time").reset_index(drop=True) if not truth.empty else pd.DataFrame()
        if fu.empty:
            continue

        out_png = plot_dir / f"{uid}.png"
        if out_png.exists() and (not overwrite):
            saved += 1
            continue

        ref_lat = float(fu["lat"].iloc[0])
        ref_lon = float(fu["lon"].iloc[0])
        ref_alt = float(fu["alt"].fillna(0.0).iloc[0]) if "alt" in fu.columns else 0.0
        fu_enu = _latlonalt_to_enu(
            fu["lat"].to_numpy(dtype=float),
            fu["lon"].to_numpy(dtype=float),
            fu["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in fu.columns else np.zeros(len(fu), dtype=float),
            ref_lat,
            ref_lon,
            ref_alt,
        )
        ru_enu = None
        if not ru.empty:
            ru_enu = _latlonalt_to_enu(
                ru["lat"].to_numpy(dtype=float),
                ru["lon"].to_numpy(dtype=float),
                ru["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in ru.columns else np.zeros(len(ru), dtype=float),
                ref_lat,
                ref_lon,
                ref_alt,
            )
        tu_enu = None
        if not tu.empty:
            tu_enu = _latlonalt_to_enu(
                tu["lat"].to_numpy(dtype=float),
                tu["lon"].to_numpy(dtype=float),
                tu["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in tu.columns else np.zeros(len(tu), dtype=float),
                ref_lat,
                ref_lon,
                ref_alt,
            )

        offs = _xy_offset_stats_m(fu, ru)
        case_name = case_map.get(uid, "")
        stats_rows.append({"true_uav_id": uid, "scenario_case": case_name, **offs, "fused_points": int(len(fu)), "report_points": int(len(ru))})

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        if tu_enu is not None and len(tu_enu) > 0:
            ax.plot(tu_enu[:, 0], tu_enu[:, 1], color=truth_color, lw=2.0, alpha=0.95, label=f"truth n={len(tu)}")
            ax.scatter([tu_enu[0, 0]], [tu_enu[0, 1]], color=truth_color, s=26, marker="o", zorder=5)
            ax.scatter([tu_enu[-1, 0]], [tu_enu[-1, 1]], color=truth_color, s=26, marker="s", zorder=5)
        ax.plot(fu_enu[:, 0], fu_enu[:, 1], color=fused_color, lw=2.0, label=f"fused(no-gps) n={len(fu)}")
        ax.scatter([fu_enu[0, 0]], [fu_enu[0, 1]], color=fused_color, s=28, marker="o", zorder=5)
        ax.scatter([fu_enu[-1, 0]], [fu_enu[-1, 1]], color=fused_color, s=28, marker="s", zorder=5)
        if ru_enu is not None and len(ru_enu) > 0:
            ax.plot(ru_enu[:, 0], ru_enu[:, 1], color=report_color, lw=1.8, alpha=0.9, label=f"report(gps) n={len(ru)}")
            ax.scatter([ru_enu[0, 0]], [ru_enu[0, 1]], color=report_color, s=24, marker="o", zorder=5)
            ax.scatter([ru_enu[-1, 0]], [ru_enu[-1, 1]], color=report_color, s=24, marker="s", zorder=5)
        else:
            ax.text(0.02, 0.92, "report missing", transform=ax.transAxes, fontsize=10, color="#d62728")

        title = f"{uid}"
        if case_name:
            title += f" | {case_name}"
        if offs["overlap_points"] > 0 and offs["mean_xy_m"] is not None:
            title += f" | mean={offs['mean_xy_m']:.1f}m med={offs['median_xy_m']:.1f}m max={offs['max_xy_m']:.1f}m"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("East (m, local ENU)")
        ax.set_ylabel("North (m, local ENU)")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        saved += 1

        if idx % 10 == 0:
            _log(f"[plot] saved {idx}/{len(uids)} trajectory compare plots")

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df.to_csv(output_dir / "traj_compare_stats_by_uav.csv", index=False)
        by_case = (
            stats_df.groupby("scenario_case")[["mean_xy_m", "median_xy_m", "max_xy_m"]]
            .median(numeric_only=True)
            .reset_index()
        )
        by_case.to_csv(output_dir / "traj_compare_stats_by_case.csv", index=False)

    summary = {
        "plot_dir": str(plot_dir),
        "plotted_uav_count": int(saved),
        "requested_uav_count": int(len(uids)),
        "stats_csv": str(output_dir / "traj_compare_stats_by_uav.csv"),
        "stats_by_case_csv": str(output_dir / "traj_compare_stats_by_case.csv"),
    }
    _json_dump(_to_builtin(summary), output_dir / "traj_plot_summary.json")
    return summary


def stage_plot_topn_overlay_compare(
    fused_truth_df: pd.DataFrame,
    report_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    if not bool(CFG.get("SAVE_TRAJ_PLOTS", True)):
        return {"skipped_by_flag": True}
    if fused_truth_df is None or fused_truth_df.empty:
        return {"skipped": True, "reason": "fused_truth_df empty"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fused = fused_truth_df.copy()
    fused_id_col = "true_uav_id" if "true_uav_id" in fused.columns else "id"
    fused[fused_id_col] = fused[fused_id_col].astype(str)
    fused["time"] = pd.to_numeric(fused["time"], errors="coerce")
    for c in ["lat", "lon", "alt"]:
        if c in fused.columns:
            fused[c] = pd.to_numeric(fused[c], errors="coerce")
    fused = fused[np.isfinite(fused["time"]) & np.isfinite(fused["lat"]) & np.isfinite(fused["lon"])].copy()
    if fused.empty:
        return {"skipped": True, "reason": "fused_truth_df no valid llh points"}

    rep = report_df.copy() if report_df is not None else pd.DataFrame()
    if not rep.empty:
        rep["id"] = rep["id"].astype(str)
        rep["time"] = pd.to_numeric(rep["time"], errors="coerce")
        for c in ["lat", "lon", "alt"]:
            if c in rep.columns:
                rep[c] = pd.to_numeric(rep[c], errors="coerce")
        rep = rep[np.isfinite(rep["time"]) & np.isfinite(rep["lat"]) & np.isfinite(rep["lon"])].copy()

    truth = truth_df.copy() if truth_df is not None else pd.DataFrame()
    if not truth.empty:
        truth["id"] = truth["id"].astype(str)
        truth["time"] = pd.to_numeric(truth["time"], errors="coerce")
        for c in ["lat", "lon", "alt"]:
            if c in truth.columns:
                truth[c] = pd.to_numeric(truth[c], errors="coerce")
        truth = truth[np.isfinite(truth["time"]) & np.isfinite(truth["lat"]) & np.isfinite(truth["lon"])].copy()

    case_map: Dict[str, str] = {}
    if assignment_df is not None and not assignment_df.empty and "true_uav_id" in assignment_df.columns and "scenario_case" in assignment_df.columns:
        case_map = dict(zip(assignment_df["true_uav_id"].astype(str), assignment_df["scenario_case"].astype(str)))

    uids = sorted(fused[fused_id_col].astype(str).unique().tolist())
    topn = int(CFG.get("TRAJ_TOPN_OVERLAY_N", 20))
    if topn <= 0:
        topn = 20
    uids = uids[:topn]
    if not uids:
        return {"skipped": True, "reason": "no uids selected"}

    # Common ENU reference for all selected tracks so they can be overlaid in one axes.
    first_u = fused[fused[fused_id_col].astype(str) == uids[0]].sort_values("time").reset_index(drop=True)
    ref_lat = float(first_u["lat"].iloc[0])
    ref_lon = float(first_u["lon"].iloc[0])
    ref_alt = float(first_u["alt"].fillna(0.0).iloc[0]) if "alt" in first_u.columns else 0.0

    fig_w, fig_h = [float(x) for x in CFG.get("TRAJ_TOPN_OVERLAY_FIGSIZE", [11.5, 9.0])]
    dpi = int(CFG.get("TRAJ_TOPN_OVERLAY_DPI", 140))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    index_rows: List[Dict[str, Any]] = []
    fused_color = str(CFG.get("TRAJ_COLOR_FUSED", "#1f77b4"))
    report_color = str(CFG.get("TRAJ_COLOR_REPORT", "#d62728"))
    truth_color = str(CFG.get("TRAJ_COLOR_TRUTH", "#2ca02c"))

    for seq, uid in enumerate(uids, start=1):
        fu = fused[fused[fused_id_col].astype(str) == uid].copy().sort_values("time").reset_index(drop=True)
        ru = rep[rep["id"].astype(str) == uid].copy().sort_values("time").reset_index(drop=True) if not rep.empty else pd.DataFrame()
        tu = truth[truth["id"].astype(str) == uid].copy().sort_values("time").reset_index(drop=True) if not truth.empty else pd.DataFrame()
        if fu.empty:
            continue

        fu_enu = _latlonalt_to_enu(
            fu["lat"].to_numpy(dtype=float),
            fu["lon"].to_numpy(dtype=float),
            fu["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in fu.columns else np.zeros(len(fu), dtype=float),
            ref_lat,
            ref_lon,
            ref_alt,
        )
        if not tu.empty:
            tu_enu = _latlonalt_to_enu(
                tu["lat"].to_numpy(dtype=float),
                tu["lon"].to_numpy(dtype=float),
                tu["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in tu.columns else np.zeros(len(tu), dtype=float),
                ref_lat,
                ref_lon,
                ref_alt,
            )
            ax.plot(tu_enu[:, 0], tu_enu[:, 1], color=truth_color, lw=1.35, alpha=0.75)
        else:
            tu_enu = None

        ax.plot(fu_enu[:, 0], fu_enu[:, 1], color=fused_color, lw=1.5, alpha=0.8)

        ru_enu = None
        if not ru.empty:
            ru_enu = _latlonalt_to_enu(
                ru["lat"].to_numpy(dtype=float),
                ru["lon"].to_numpy(dtype=float),
                ru["alt"].fillna(0.0).to_numpy(dtype=float) if "alt" in ru.columns else np.zeros(len(ru), dtype=float),
                ref_lat,
                ref_lon,
                ref_alt,
            )
            ax.plot(ru_enu[:, 0], ru_enu[:, 1], color=report_color, lw=1.2, alpha=0.8)

        # Label with sequence index (as requested) near fused start and report start if available.
        fx, fy = float(fu_enu[0, 0]), float(fu_enu[0, 1])
        ax.text(fx, fy, str(seq), color=fused_color, fontsize=8, weight="bold", alpha=0.95)
        ax.scatter([fx], [fy], color=fused_color, s=10, alpha=0.9)
        if ru_enu is not None and len(ru_enu) > 0:
            rx, ry = float(ru_enu[0, 0]), float(ru_enu[0, 1])
            ax.text(rx, ry, str(seq), color=report_color, fontsize=8, alpha=0.9)
            ax.scatter([rx], [ry], color=report_color, s=9, alpha=0.9)

        offs = _xy_offset_stats_m(fu, ru)
        index_rows.append(
            {
                "seq_no": int(seq),
                "true_uav_id": str(uid),
                "scenario_case": str(case_map.get(uid, "")),
                "fused_points": int(len(fu)),
                "report_points": int(len(ru)),
                "overlap_points": int(offs.get("overlap_points", 0) or 0),
                "mean_xy_m": offs.get("mean_xy_m"),
                "median_xy_m": offs.get("median_xy_m"),
                "max_xy_m": offs.get("max_xy_m"),
            }
        )

    legend_handles = [
        Line2D([0], [0], color=truth_color, lw=2.0, label="truth"),
        Line2D([0], [0], color=fused_color, lw=2.0, label="fused(no-gps)"),
        Line2D([0], [0], color=report_color, lw=2.0, label="report(gps)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    ax.set_title(f"Top {len(uids)} UAV Overlay (Blue=fused, Red=report, labels=seq no.)", fontsize=11)
    ax.set_xlabel("East (m, common ENU)")
    ax.set_ylabel("North (m, common ENU)")
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    out_png = output_dir / str(CFG.get("TRAJ_TOPN_OVERLAY_FILENAME", "traj_compare_top20_overlay.png"))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    idx_df = pd.DataFrame(index_rows)
    idx_csv = output_dir / "traj_compare_topn_index_map.csv"
    if not idx_df.empty:
        idx_df.to_csv(idx_csv, index=False)

    return {
        "plot_path": str(out_png),
        "index_map_csv": str(idx_csv),
        "uav_count": int(len(index_rows)),
    }


# ============================================================
# Stage 5: graph feature matching
# ============================================================


def _summarize_match_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"frames": 0}
    voted_union: Dict[str, set] = {}
    event_type_counts: Dict[str, int] = {}
    total_matches = 0
    for fr in results:
        total_matches += int(len(fr.get("matches", [])))
        for k, vals in (fr.get("voted_anomalies", {}) or {}).items():
            s = voted_union.setdefault(str(k), set())
            if isinstance(vals, list):
                for v in vals:
                    s.add(str(v))
        for ev in (fr.get("events", []) or []):
            et = str(ev.get("type", ev.get("event_type", "unknown")))
            event_type_counts[et] = int(event_type_counts.get(et, 0)) + 1
    last = results[-1]
    return {
        "frames": int(len(results)),
        "total_match_pairs_over_frames": int(total_matches),
        "last_frame_time": float(last.get("time", 0.0)),
        "last_global_score": float(last.get("global_score", 0.0)),
        "last_global_anomaly": bool(last.get("global_anomaly", False)),
        "last_abnormal_entities": [str(x) for x in (last.get("abnormal_entities", []) or [])],
        "voted_anomaly_counts_union": {k: int(len(v)) for k, v in voted_union.items()},
        "voted_anomaly_entities_union": {k: sorted(list(v)) for k, v in voted_union.items()},
        "event_type_counts": event_type_counts,
    }


def _build_match_frame_stats_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fr in results:
        matches = fr.get("matches", []) or []
        rows.append(
            {
                "time": float(fr.get("time", 0.0)),
                "num_fusion_nodes": int(fr.get("num_fusion_nodes", 0)),
                "num_report_nodes": int(fr.get("num_report_nodes", 0)),
                "match_count": int(len(matches)),
                "unreported_count": int(len(fr.get("unreported", []) or [])),
                "false_reports_count": int(len(fr.get("false_reports", []) or [])),
                "duplicate_reports_count": int(len(fr.get("duplicate_reports", []) or [])),
                "drift_candidates_count": int(len(fr.get("drift_candidates", []) or [])),
                "deviation_candidates_count": int(len(fr.get("deviation_candidates", []) or [])),
                "global_score": float(fr.get("global_score", 0.0)),
                "global_anomaly": bool(fr.get("global_anomaly", False)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "num_fusion_nodes",
                "num_report_nodes",
                "match_count",
                "unreported_count",
                "false_reports_count",
                "duplicate_reports_count",
                "drift_candidates_count",
                "deviation_candidates_count",
                "global_score",
                "global_anomaly",
            ]
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _build_match_true_pair_eval_df(
    results: List[Dict[str, Any]],
    passive_map_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    if passive_map_df is None or passive_map_df.empty:
        return pd.DataFrame()
    pmap = passive_map_df.copy()
    pmap["passive_track_id"] = pmap["passive_track_id"].astype(str)
    pmap["true_uav_id"] = pmap["true_uav_id"].astype(str)
    p2u = dict(zip(pmap["passive_track_id"], pmap["true_uav_id"]))
    case_map: Dict[str, str] = {}
    if assignment_df is not None and not assignment_df.empty and "true_uav_id" in assignment_df.columns and "scenario_case" in assignment_df.columns:
        case_map = dict(zip(assignment_df["true_uav_id"].astype(str), assignment_df["scenario_case"].astype(str)))

    rows: List[Dict[str, Any]] = []
    for fr in results:
        t = float(fr.get("time", 0.0))
        for m in (fr.get("matches", []) or []):
            fid = str(m.get("fusion_id", ""))
            rid = str(m.get("report_id", ""))
            true_uid = p2u.get(fid, "")
            rows.append(
                {
                    "time": t,
                    "fusion_id": fid,
                    "report_id": rid,
                    "fusion_true_uav_id": true_uid,
                    "is_true_match": bool(true_uid != "" and rid == true_uid),
                    "scenario_case": case_map.get(true_uid, ""),
                    "node_score": _safe_float(m.get("node_score"), np.nan),
                    "edge_score": _safe_float(m.get("edge_score"), np.nan),
                    "pair_score": _safe_float(m.get("pair_score"), np.nan),
                    "spectral_score": _safe_float(m.get("spectral_score"), np.nan),
                    "offset_m": _safe_float(m.get("offset_m"), np.nan),
                    "vel_offset_mps": _safe_float(m.get("vel_offset_mps"), np.nan),
                    "heading_diff_deg": _safe_float(m.get("heading_diff_deg"), np.nan),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "fusion_id",
                "report_id",
                "fusion_true_uav_id",
                "is_true_match",
                "scenario_case",
                "node_score",
                "edge_score",
                "pair_score",
                "spectral_score",
                "offset_m",
                "vel_offset_mps",
                "heading_diff_deg",
            ]
        )
    return pd.DataFrame(rows).sort_values(["time", "fusion_id", "report_id"]).reset_index(drop=True)


def _summarize_match_true_pair_eval(eval_df: pd.DataFrame) -> Dict[str, Any]:
    if eval_df is None or eval_df.empty:
        return {"matched_pairs": 0}
    out: Dict[str, Any] = {
        "matched_pairs": int(len(eval_df)),
        "true_matches": int(eval_df["is_true_match"].sum()),
        "pair_precision": float(eval_df["is_true_match"].mean()),
    }
    by_case = (
        eval_df.groupby("scenario_case")["is_true_match"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "matched_pairs", "sum": "true_matches", "mean": "pair_precision"})
    )
    out["by_case"] = [
        {
            "scenario_case": str(r["scenario_case"]),
            "matched_pairs": int(r["matched_pairs"]),
            "true_matches": int(r["true_matches"]),
            "pair_precision": float(r["pair_precision"]),
        }
        for _, r in by_case.iterrows()
    ]
    # Per-frame pair precision distribution.
    per_frame = eval_df.groupby("time")["is_true_match"].mean()
    out["per_frame_pair_precision"] = {
        "mean": float(per_frame.mean()),
        "median": float(per_frame.median()),
        "p10": float(per_frame.quantile(0.1)),
        "p90": float(per_frame.quantile(0.9)),
    }
    return out


def stage_run_matching(passive_df: pd.DataFrame, report_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from drone_sys.app.core.uavMatch import match as match_mod

    if passive_df is None or passive_df.empty:
        raise RuntimeError("passive_df is empty")
    if report_df is None:
        report_df = pd.DataFrame(columns=["id", "time", "lat", "lon", "alt", "vx", "vy", "vz", "confidence"])

    pass_std = _standardize_track_df(passive_df, default_confidence=float(CFG["PASSIVE_CONFIDENCE_DEFAULT"]))
    rep_std = _standardize_track_df(report_df, default_confidence=0.85)

    mc = deepcopy(match_mod.CONFIG)
    mc.update(dict(CFG.get("MATCH_CONFIG_OVERRIDES", {})))
    detector = match_mod.GraphTrackAnomalyDetector(match_mod.MatchConfig(mc))
    _log(f"[match] passive_rows={len(pass_std)} report_rows={len(rep_std)}")
    results = detector.process_stream(df_fusion=pass_std, df_report=rep_std)
    return results, _summarize_match_results(results)


# ============================================================
# Orchestration / output
# ============================================================


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_existing_tracks_from_output(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    passive_path = output_dir / "passive_tracks.csv"
    report_path = output_dir / "report_tracks.csv"
    if not passive_path.exists() or not report_path.exists():
        raise FileNotFoundError("Cached passive/report CSVs not found under output dir.")
    return pd.read_csv(passive_path), pd.read_csv(report_path)


def run_pipeline(cfg_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CFG
    if cfg_override:
        merged = deepcopy(CFG)
        _deep_update_dict(merged, cfg_override)
        CFG = merged

    random.seed(int(CFG["SEED"]))
    np.random.seed(int(CFG["SEED"]))

    work_root = _ensure_dir(_p(str(CFG["WORK_ROOT"])))
    raw_root = _ensure_dir(_p(str(CFG["RAW_DATASET_DIR"])))
    processed_root = _ensure_dir(_p(str(CFG["PROCESSED_DATASET_DIR"])))
    output_dir = _ensure_dir(_p(str(CFG["OUTPUT_DIR"])))
    _json_dump(_to_builtin(CFG), work_root / "pipeline_config.effective.json")

    stage_info: Dict[str, Any] = {}
    if bool(CFG.get("RUN_GENERATE", True)):
        stage_info["generate"] = stage_generate_raw_dataset(raw_root)
    else:
        stage_info["generate"] = {"skipped_by_flag": True}

    if bool(CFG.get("RUN_TRANSFER_CONFIDENCE", True)):
        stage_info["transfer_confidence"] = stage_transfer_confidence(raw_root, processed_root)
    else:
        stage_info["transfer_confidence"] = {"skipped_by_flag": True}

    passive_df = pd.DataFrame()
    report_df = pd.DataFrame()
    passive_map_df = pd.DataFrame()
    assignment_df = pd.DataFrame()
    fused_truth_df = pd.DataFrame()
    truth_tracks_df = pd.DataFrame()

    if bool(CFG.get("RUN_FUSION", True)):
        fused_results, gps_frames_by_uid, fusion_summary = stage_fusion_no_gps(processed_root)
        stage_info["fusion"] = fusion_summary
        fused_truth_df = _fused_results_to_truth_df(fused_results)
        passive_df, report_df, passive_map_df, assignment_df, scenario_summary = stage_build_tracks_and_inject_anomalies(
            fused_results=fused_results,
            gps_frames_by_uid=gps_frames_by_uid,
        )
        stage_info["scenario"] = scenario_summary
        if bool(CFG.get("SAVE_INTERMEDIATE_CSV", True)):
            _save_df(fused_truth_df, output_dir / "fused_tracks_truth_ids.csv")
            _save_df(passive_df, output_dir / "passive_tracks.csv")
            _save_df(report_df, output_dir / "report_tracks.csv")
            _save_df(passive_map_df, output_dir / "passive_track_truth_map.csv")
            _save_df(assignment_df, output_dir / "scenario_assignment.csv")
    else:
        passive_df, report_df = _load_existing_tracks_from_output(output_dir)
        stage_info["fusion"] = {"skipped_by_flag": True, "loaded_cached_tracks": True}
        if (output_dir / "fused_tracks_truth_ids.csv").exists():
            fused_truth_df = pd.read_csv(output_dir / "fused_tracks_truth_ids.csv")
        if (output_dir / "passive_track_truth_map.csv").exists():
            passive_map_df = pd.read_csv(output_dir / "passive_track_truth_map.csv")
        if (output_dir / "scenario_assignment.csv").exists():
            assignment_df = pd.read_csv(output_dir / "scenario_assignment.csv")
        if fused_truth_df.empty and (not passive_df.empty) and (not passive_map_df.empty):
            fused_truth_df = _reconstruct_fused_truth_from_passive(passive_df, passive_map_df)
            if bool(CFG.get("SAVE_INTERMEDIATE_CSV", True)) and not fused_truth_df.empty:
                _save_df(fused_truth_df, output_dir / "fused_tracks_truth_ids.csv")

    if bool(CFG.get("RUN_MATCH", True)):
        match_results, match_summary = stage_run_matching(passive_df=passive_df, report_df=report_df)
        frame_stats_df = _build_match_frame_stats_df(match_results)
        pair_eval_df = _build_match_true_pair_eval_df(match_results, passive_map_df=passive_map_df, assignment_df=assignment_df)
        pair_eval_summary = _summarize_match_true_pair_eval(pair_eval_df)
        match_summary["frame_stats"] = {
            "avg_matches_per_frame": float(frame_stats_df["match_count"].mean()) if not frame_stats_df.empty else 0.0,
            "median_matches_per_frame": float(frame_stats_df["match_count"].median()) if not frame_stats_df.empty else 0.0,
            "max_matches_per_frame": int(frame_stats_df["match_count"].max()) if not frame_stats_df.empty else 0,
        }
        match_summary["true_pair_eval"] = pair_eval_summary
        stage_info["match"] = match_summary
        if bool(CFG.get("SAVE_MATCH_RESULTS_JSON", True)):
            _json_dump(_to_builtin(match_results), output_dir / "match_results.json")
            _json_dump(_to_builtin(match_summary), output_dir / "match_summary.json")
        _save_df(frame_stats_df, output_dir / "match_frame_stats.csv")
        _save_df(pair_eval_df, output_dir / "match_true_pair_eval.csv")
        _json_dump(_to_builtin(pair_eval_summary), output_dir / "match_true_pair_eval_summary.json")
    else:
        stage_info["match"] = {"skipped_by_flag": True}

    if bool(CFG.get("RUN_PLOT_TRAJ_COMPARE", True)):
        truth_tracks_df = _load_truth_tracks_for_plot(
            processed_root=processed_root,
            raw_root=raw_root,
            batch_prefix=str(CFG["BATCH_PREFIX"]),
        )
        if bool(CFG.get("SAVE_INTERMEDIATE_CSV", True)) and (not truth_tracks_df.empty):
            _save_df(truth_tracks_df, output_dir / "truth_tracks_truth_ids.csv")
        per_uav_plot_summary = stage_plot_fused_vs_report_per_uav(
            fused_truth_df=fused_truth_df,
            report_df=report_df,
            truth_df=truth_tracks_df,
            assignment_df=assignment_df,
            output_dir=output_dir,
        )
        topn_overlay_summary = stage_plot_topn_overlay_compare(
            fused_truth_df=fused_truth_df,
            report_df=report_df,
            truth_df=truth_tracks_df,
            assignment_df=assignment_df,
            output_dir=output_dir,
        )
        stage_info["traj_plot"] = {
            "per_uav": per_uav_plot_summary,
            "topn_overlay": topn_overlay_summary,
        }
    else:
        stage_info["traj_plot"] = {"skipped_by_flag": True}

    final_summary = {
        "work_root": str(work_root),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "rows": {
            "fused_truth": int(len(fused_truth_df)),
            "truth": int(len(truth_tracks_df)),
            "passive": int(len(passive_df)),
            "report": int(len(report_df)),
            "fused_truth_tracks": int(fused_truth_df["id"].nunique()) if not fused_truth_df.empty else 0,
            "truth_tracks": int(truth_tracks_df["id"].nunique()) if not truth_tracks_df.empty else 0,
            "passive_tracks": int(passive_df["id"].nunique()) if not passive_df.empty else 0,
            "report_tracks": int(report_df["id"].nunique()) if not report_df.empty else 0,
        },
        "stage_info": _to_builtin(stage_info),
    }
    _json_dump(_to_builtin(final_summary), output_dir / "pipeline_run_summary.json")
    _log(f"[done] outputs -> {output_dir}")
    return final_summary


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
