import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


# ==========================================================
# CONFIG (edit here)
# ==========================================================
BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_FUSION_DIR = os.path.normpath(os.path.join(BASELINE_DIR, ".."))
DATASET_BATCH_DIR = os.path.normpath(
    os.path.join(
        BASELINE_DIR,
        "..",
        "..",
        "datasetBuilder",
        "dataset-processed",
        "test-datasets",
        "scenario_ultra_precision_eval_100x60",
        "batch01",
    )
)

# UAV selection / batch evaluation
# If EVAL_ALL_UAVS=True, TARGET_UAV_ID/TARGET_UAV_INDEX are ignored unless you add custom filtering.
EVAL_ALL_UAVS = True
MAX_UAVS = 0  # 0 means all UAVs in the batch
PLOT_FIRST_N_UAVS = 20

# Fallback single-UAV selectors (used when EVAL_ALL_UAVS=False)
TARGET_UAV_ID: Optional[str] = None
TARGET_UAV_INDEX: int = 0

# "Your fusion method" (graph fusion model) paths.
USER_METHOD_LABEL = "GraphFusion"
USER_MODEL_PATH = os.path.normpath(os.path.join(DRONE_FUSION_DIR, "model_result", "graph_fusion_model_v2.8.pt"))
USER_NORM_PATH = os.path.normpath(os.path.join(DRONE_FUSION_DIR, "model_result", "graph_norm_v2.8.pth"))
# Match evaluate.py behavior: use stride=1 at evaluation time (None -> use checkpoint/runtime stride)
USER_EVAL_STRIDE_OVERRIDE: Optional[int] = 1

# Inference blending params for your graph fusion method (match evaluate.py by default)
MERGE_EDGE_TAPER_MIN = 0.12
WARMUP_POINTS = 28
WARMUP_MIN_COVERAGE = 5.0
TAIL_POINTS = 40

# EKF / UKF shared config
ALIGN_TOLERANCE_S = 0.55
PROCESS_ACCEL_SIGMA = 6.0
USE_VELOCITY_UPDATE = False
USE_SPEED_UPDATE = False
USE_CONFIDENCE_SCALE = True
USE_QUALITY_COLUMNS = True
USE_MODALITY_Q_HINT = True
MODALITY_Q_HINT_STRENGTH = 1.0

# UKF sigma-point params
UKF_ALPHA = 1e-2
UKF_BETA = 2.0
UKF_KAPPA = 0.0

# IRLS params
IRLS_MAX_ITER = 8
IRLS_HUBER_K = 2.5
IRLS_TOL_M = 1e-3
IRLS_USE_TEMPORAL_PRIOR = True
IRLS_USE_VELOCITY_PRED = True
IRLS_TEMPORAL_PRIOR_SIGMA_M = 120.0

# OSPA (multi-target set metric across all UAVs at each timestamp)
OSPA_ENABLE = True
OSPA_P = 2
OSPA_CUTOFFS_M = (20.0, 50.0)
OSPA_INCLUDE_XY = True

# Outputs
# `OUTPUT_ROOT_DIR/<dataset_name>/...` will be used, where dataset_name is derived from DATASET_BATCH_DIR.
OUTPUT_ROOT_DIR = os.path.join(BASELINE_DIR, "outputs")
PLOT_FILENAME = "compare_ekf_ukf_graphfusion_vs_rawgps_single.png"
PLOTS_DIRNAME = "compare_ekf_ukf_graphfusion_vs_rawgps_first20"
METRICS_CSV_FILENAME = "compare_ekf_ukf_graphfusion_vs_rawgps_all_uavs.csv"
SAVE_PLOT = True
SHOW_PLOT = False


# ==========================================================
# Imports from baseline modules
# ==========================================================
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)
if DRONE_FUSION_DIR not in sys.path:
    sys.path.insert(0, DRONE_FUSION_DIR)

try:
    from . import ekf_fusion, irls_fusion, ukf_fusion  # type: ignore
    from .kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        _align_modality_rows,
        _detect_id_col,
        _load_batch_frames,
        _uav_list,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )
except ImportError:
    import ekf_fusion  # type: ignore
    import irls_fusion  # type: ignore
    import ukf_fusion  # type: ignore
    from kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        _align_modality_rows,
        _detect_id_col,
        _load_batch_frames,
        _uav_list,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )

def _select_uav(df_truth: pd.DataFrame, target_uav_id: Optional[str], target_uav_index: int) -> Tuple[str, str, pd.DataFrame]:
    id_col, uavs = _uav_list(df_truth)
    if target_uav_id is not None:
        if target_uav_id not in uavs:
            raise RuntimeError(f"TARGET_UAV_ID not found in truth.csv: {target_uav_id}")
        uav = target_uav_id
    else:
        if target_uav_index < 0 or target_uav_index >= len(uavs):
            raise RuntimeError(f"TARGET_UAV_INDEX out of range: {target_uav_index}, total={len(uavs)}")
        uav = uavs[int(target_uav_index)]
    df_t = df_truth[df_truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
    if len(df_t) == 0:
        raise RuntimeError(f"No rows in truth.csv for UAV: {uav}")
    return id_col, uav, df_t


def _list_uavs(df_truth: pd.DataFrame) -> Tuple[str, List[str]]:
    id_col, uavs = _uav_list(df_truth)
    return id_col, list(uavs)


def _truth_enu(df_t: pd.DataFrame) -> Tuple[np.ndarray, float, float, float]:
    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]
    e_gt, n_gt, u_gt = latlon_to_enu(
        df_t["lat"].values,
        df_t["lon"].values,
        df_t["alt"].values,
        lat0,
        lon0,
        alt0,
    )
    truth_enu = np.stack([e_gt, n_gt, u_gt], axis=-1).astype(np.float32)
    return truth_enu, float(lat0), float(lon0), float(alt0)


def _raw_gps_on_truth_timeline(
    df_truth_u: pd.DataFrame,
    gps_df: pd.DataFrame,
    lat0: float,
    lon0: float,
    alt0: float,
    align_tolerance_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, _ = _align_modality_rows(df_truth_u=df_truth_u, df_mod_u=gps_df, align_tolerance_s=float(align_tolerance_s))
    t_total = len(df_truth_u)
    gps_enu = np.full((t_total, 3), np.nan, dtype=np.float32)
    valid = np.zeros((t_total,), dtype=bool)
    for i, row in enumerate(rows):
        if row is None:
            continue
        try:
            if int(float(row.get("missing_flag", 0))) > 0:
                continue
        except Exception:
            pass
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
            alt = float(row.get("alt"))
        except Exception:
            continue
        if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
            continue
        e, n, u = latlon_to_enu(lat, lon, alt, lat0, lon0, alt0)
        gps_enu[i] = np.array([e, n, u], dtype=np.float32)
        valid[i] = True
    return gps_enu, valid


def _metrics_row(method: str, pred_enu: np.ndarray, truth_enu: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    if valid_mask is None:
        valid_mask = np.ones((len(truth_enu),), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != len(truth_enu):
        raise RuntimeError(f"valid_mask length mismatch for {method}")
    n_total = int(len(truth_enu))
    n_eval = int(np.sum(valid_mask))
    if n_eval <= 0:
        metrics = {"RMSE": np.nan, "MAE": np.nan, "MEDAE": np.nan, "P95": np.nan, "MAX": np.nan}
    else:
        metrics = calc_err(np.asarray(pred_enu)[valid_mask], np.asarray(truth_enu)[valid_mask])
    return {
        "method": method,
        "RMSE": float(metrics["RMSE"]) if np.isfinite(metrics["RMSE"]) else np.nan,
        "MAE": float(metrics["MAE"]) if np.isfinite(metrics["MAE"]) else np.nan,
        "MEDAE": float(metrics["MEDAE"]) if np.isfinite(metrics["MEDAE"]) else np.nan,
        "P95": float(metrics["P95"]) if np.isfinite(metrics["P95"]) else np.nan,
        "MAX": float(metrics["MAX"]) if np.isfinite(metrics["MAX"]) else np.nan,
        "n_eval": n_eval,
        "n_total": n_total,
        "coverage_ratio": float(n_eval / max(n_total, 1)),
    }


def _add_batch_uav_fields(row: Dict[str, object], batch_dir: str, uav_id: str, row_type: str = "per_uav") -> Dict[str, object]:
    out = dict(row)
    out["row_type"] = row_type
    out["batch_dir"] = batch_dir
    out["batch"] = os.path.basename(batch_dir.rstrip("\\/"))
    out["uav_id"] = uav_id
    return out


def _safe_name_token(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "unknown"
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _dataset_output_dir(batch_dir: str) -> str:
    batch_dir = os.path.normpath(batch_dir)
    batch_name = _safe_name_token(os.path.basename(batch_dir.rstrip("\\/")))
    parent_name = _safe_name_token(os.path.basename(os.path.dirname(batch_dir)))
    if parent_name and parent_name != "unknown" and parent_name != batch_name:
        dataset_folder = f"{parent_name}__{batch_name}"
    else:
        dataset_folder = batch_name
    return os.path.join(OUTPUT_ROOT_DIR, dataset_folder)


def _build_mean_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if len(metrics_df) == 0:
        return pd.DataFrame()
    per_uav = metrics_df[metrics_df["row_type"] == "per_uav"].copy()
    if len(per_uav) == 0:
        return pd.DataFrame()

    group_cols = ["method"]
    num_cols = ["RMSE", "MAE", "MEDAE", "P95", "MAX", "n_eval", "n_total", "coverage_ratio"]
    agg = per_uav.groupby(group_cols, dropna=False)[num_cols].mean(numeric_only=True).reset_index()
    agg["row_type"] = "mean_all_uavs"
    agg["uav_id"] = "__MEAN__"
    agg["batch"] = per_uav["batch"].iloc[0]
    agg["batch_dir"] = per_uav["batch_dir"].iloc[0]
    cols = ["row_type", "batch_dir", "batch", "uav_id", "method"] + [c for c in per_uav.columns if c not in {"row_type", "batch_dir", "batch", "uav_id", "method"}]
    for c in cols:
        if c not in agg.columns:
            agg[c] = np.nan
    return agg[cols]


def _hungarian_min_cost_rect(cost: np.ndarray) -> float:
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        raise ValueError("cost must be 2D")
    m, n = c.shape
    if m == 0:
        return 0.0
    if n == 0:
        return float(np.inf)
    if m > n:
        c = c.T
        m, n = c.shape

    u = np.zeros(m + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, m + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = np.full(m, -1, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] > 0:
            row_to_col[int(p[j]) - 1] = j - 1
    if np.any(row_to_col < 0):
        return float(np.inf)
    return float(c[np.arange(m), row_to_col].sum())


def ospa_distance(pred_set: np.ndarray, gt_set: np.ndarray, p: int = 2, cutoff_m: float = 20.0) -> float:
    p = max(int(p), 1)
    c = max(float(cutoff_m), 1e-6)

    x = np.asarray(pred_set, dtype=float)
    y = np.asarray(gt_set, dtype=float)
    if x.ndim != 2:
        x = x.reshape((-1, x.shape[-1] if x.ndim > 0 else 1))
    if y.ndim != 2:
        y = y.reshape((-1, y.shape[-1] if y.ndim > 0 else 1))

    if x.shape[0] == 0 and y.shape[0] == 0:
        return 0.0
    if x.shape[0] == 0 or y.shape[0] == 0:
        return c
    if x.shape[1] != y.shape[1]:
        raise ValueError("pred_set and gt_set must have same point dimension")

    m = int(x.shape[0])
    n = int(y.shape[0])
    if m <= n:
        a, b = x, y
        denom = n
    else:
        a, b = y, x
        m, n = n, m
        denom = n

    dmat = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    dmat = np.minimum(dmat, c) ** p
    assign_cost = _hungarian_min_cost_rect(dmat)
    total = (assign_cost + (n - m) * (c**p)) / max(denom, 1)
    return float(total ** (1.0 / p))


def _cutoff_key(cutoff_m: float) -> str:
    c = float(cutoff_m)
    if abs(c - round(c)) < 1e-9:
        return str(int(round(c)))
    return str(c).replace(".", "p")


def _compute_batch_ospa_from_tracks(
    batch_name: str,
    tracks: Sequence[Dict[str, object]],
    p: int = 2,
    cutoffs_m: Sequence[float] = (20.0, 50.0),
    include_xy: bool = True,
) -> Dict[str, object]:
    if len(tracks) == 0:
        return {"batch": batch_name, "ospa_frame_count": 0, "ospa_uav_count": 0}

    pred_map: Dict[float, List[np.ndarray]] = {}
    gt_map: Dict[float, List[np.ndarray]] = {}

    for tr in tracks:
        ts = np.asarray(tr.get("timestamps", []), dtype=float)
        pred = np.asarray(tr.get("pred_enu_batch", []), dtype=float)
        gt = np.asarray(tr.get("gt_enu_batch", []), dtype=float)
        if ts.ndim != 1 or pred.ndim != 2 or gt.ndim != 2:
            continue
        n = min(len(ts), len(pred), len(gt))
        if n <= 0:
            continue
        ts = ts[:n]
        pred = pred[:n]
        gt = gt[:n]
        for i in range(n):
            t = float(ts[i])
            if not np.isfinite(t):
                continue
            pv = pred[i]
            gv = gt[i]
            if np.all(np.isfinite(pv)):
                pred_map.setdefault(t, []).append(np.asarray(pv, dtype=float))
            if np.all(np.isfinite(gv)):
                gt_map.setdefault(t, []).append(np.asarray(gv, dtype=float))

    frame_keys = sorted(set(pred_map.keys()) | set(gt_map.keys()))
    if len(frame_keys) == 0:
        return {"batch": batch_name, "ospa_frame_count": 0, "ospa_uav_count": len(tracks)}

    frame_rows: List[Dict[str, object]] = []
    for t in frame_keys:
        pred_pts = np.asarray(pred_map.get(t, []), dtype=float)
        gt_pts = np.asarray(gt_map.get(t, []), dtype=float)
        if pred_pts.ndim == 1:
            pred_pts = pred_pts.reshape((-1, 3)) if pred_pts.size > 0 else np.zeros((0, 3), dtype=float)
        if gt_pts.ndim == 1:
            gt_pts = gt_pts.reshape((-1, 3)) if gt_pts.size > 0 else np.zeros((0, 3), dtype=float)
        row: Dict[str, object] = {
            "timestamp": float(t),
            "pred_cardinality": int(pred_pts.shape[0]),
            "gt_cardinality": int(gt_pts.shape[0]),
        }
        for c in cutoffs_m:
            c_key = _cutoff_key(float(c))
            row[f"ospa3d_p{int(p)}_c{c_key}"] = ospa_distance(pred_pts, gt_pts, p=p, cutoff_m=float(c))
            if bool(include_xy):
                row[f"ospa_xy_p{int(p)}_c{c_key}"] = ospa_distance(pred_pts[:, :2], gt_pts[:, :2], p=p, cutoff_m=float(c))
        frame_rows.append(row)

    df_f = pd.DataFrame(frame_rows)
    out: Dict[str, object] = {
        "batch": batch_name,
        "ospa_frame_count": int(len(df_f)),
        "ospa_uav_count": int(len(tracks)),
        "ospa_p": int(p),
    }
    metric_cols = [c for c in df_f.columns if str(c).startswith("ospa")]
    for col in metric_cols:
        vals = pd.to_numeric(df_f[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{col}_mean"] = float(np.mean(vals)) if vals.size > 0 else np.nan
        out[f"{col}_median"] = float(np.median(vals)) if vals.size > 0 else np.nan
        out[f"{col}_p95"] = float(np.percentile(vals, 95)) if vals.size > 0 else np.nan
    return out


def _build_method_mod_frames(batch_dir: str, modalities: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for m in modalities:
        fname = "5g_a.csv" if m == "5g_a" else f"{m}.csv"
        p = os.path.join(batch_dir, fname)
        out[m] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return out


def _load_graph_fusion_runner(batch_dir: str) -> Dict[str, object]:
    try:
        import inference as model_inf  # type: ignore
    except ImportError as ex:
        raise RuntimeError(
            "Failed to import `droneFusion/inference.py` for your fusion method. "
            "This usually means the current environment is missing dependencies like torch/matplotlib."
        ) from ex

    try:
        import torch  # local import to avoid hard failure when only EKF/UKF are used
    except ImportError as ex:
        raise RuntimeError(
            "PyTorch (`torch`) is required to run your graph fusion method. "
            "Please run this script in the same environment used for your model inference."
        ) from ex

    model, x_mean, x_std, y_mean, y_std, runtime = model_inf.load_model_and_runtime(
        model_path=USER_MODEL_PATH,
        norm_path=USER_NORM_PATH,
        device=model_inf.DEVICE,
    )

    modalities = list(runtime["modalities"])
    mod_frames = _build_method_mod_frames(batch_dir, modalities)
    return {
        "model_inf": model_inf,
        "torch": torch,
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "runtime": runtime,
        "mod_frames": mod_frames,
        "batch_dir": batch_dir,
    }


def run_graph_fusion_uav(batch_dir: str, df_truth_u: pd.DataFrame, uav_id: str, runner: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    runner_local = runner if runner is not None else _load_graph_fusion_runner(batch_dir)
    model_inf = runner_local["model_inf"]
    torch = runner_local["torch"]
    model = runner_local["model"]
    x_mean = runner_local["x_mean"]
    x_std = runner_local["x_std"]
    y_mean = runner_local["y_mean"]
    y_std = runner_local["y_std"]
    runtime = runner_local["runtime"]
    mod_frames = runner_local["mod_frames"]

    in_dim = int(runtime["in_dim"])
    window_size = int(runtime["window_size"])
    stride = int(USER_EVAL_STRIDE_OVERRIDE) if USER_EVAL_STRIDE_OVERRIDE is not None else int(runtime["stride"])
    align_tolerance_s = float(runtime["align_tolerance_s"])
    modalities = list(runtime["modalities"])

    truth_enu, lat0, lon0, alt0 = _truth_enu(df_truth_u)

    if in_dim >= model_inf.NODE_FEAT_DIM:
        windows, starts = model_inf.build_sparse_windows_new(
            df_truth_u=df_truth_u,
            mod_frames=mod_frames,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            modalities=modalities,
            window_size=window_size,
            stride=stride,
            align_tolerance_s=align_tolerance_s,
        )
    else:
        windows, starts = model_inf.build_sparse_windows_legacy(
            df_truth_u=df_truth_u,
            mod_frames=mod_frames,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            modalities=modalities,
            window_size=window_size,
            stride=stride,
            in_dim=in_dim,
        )

    if len(windows) == 0:
        raise RuntimeError(f"No valid sparse windows for user method UAV={uav_id}")

    obs_fallback_enu, obs_fallback_w = model_inf.build_obs_fallback_series(
        df_truth_u=df_truth_u,
        mod_frames=mod_frames,
        modalities=modalities,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        align_tolerance_s=align_tolerance_s,
    )

    preds = []
    window_weights = []
    for w in windows:
        window_weights.append(model_inf.estimate_window_quality(w["node_feat"]))
        node_feat = torch.tensor(w["node_feat"], dtype=torch.float32, device=model_inf.DEVICE)
        node_t = torch.tensor(w["node_t"], dtype=torch.long, device=model_inf.DEVICE)
        node_m = torch.tensor(w["node_m"], dtype=torch.long, device=model_inf.DEVICE)
        node_feat = model_inf.fit_feature_dim(node_feat, int(x_mean.numel()))
        node_feat = (node_feat - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            pred_norm = model(
                node_feat=node_feat.unsqueeze(0),
                node_t=node_t.unsqueeze(0),
                node_m=node_m.unsqueeze(0),
                node_mask=torch.ones((1, node_feat.shape[0]), dtype=torch.float32, device=model_inf.DEVICE),
                window_size=window_size,
            )[0]
        pred = pred_norm * y_std + y_mean
        preds.append(pred.detach().cpu().numpy())

    fusion_enu, cover_weight, cover_count = model_inf.merge_windows(
        np.array(preds),
        starts,
        t_total=len(truth_enu),
        window=window_size,
        window_weights=window_weights,
        edge_taper_min=float(MERGE_EDGE_TAPER_MIN),
    )
    fusion_enu, warmup_replaced = model_inf.apply_warmup_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        warmup_points=int(WARMUP_POINTS),
        min_coverage=float(WARMUP_MIN_COVERAGE),
    )
    fusion_enu, tail_replaced = model_inf.apply_tail_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        tail_points=int(TAIL_POINTS),
        min_coverage=float(WARMUP_MIN_COVERAGE),
    )

    return {
        "pred_enu": np.asarray(fusion_enu, dtype=np.float32),
        "truth_enu": truth_enu,
        "lat0": lat0,
        "lon0": lon0,
        "alt0": alt0,
        "runtime": runtime,
        "merge_info": {
            "windows": len(windows),
            "warmup_blended": int(warmup_replaced),
            "tail_blended": int(tail_replaced),
            "cover_weight": np.asarray(cover_weight, dtype=np.float32),
            "cover_count": np.asarray(cover_count, dtype=np.float32),
        },
    }


def _plot_comparison(
    out_png: str,
    uav_id: str,
    batch_dir: str,
    lat0: float,
    lon0: float,
    alt0: float,
    truth_enu: np.ndarray,
    graph_pred_enu: np.ndarray,
    ekf_pred_enu: np.ndarray,
    ukf_pred_enu: np.ndarray,
    irls_pred_enu: Optional[np.ndarray],
    gps_raw_enu: np.ndarray,
    truth_timestamps: Optional[np.ndarray] = None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as ex:
        raise RuntimeError(
            "matplotlib is required for plotting comparison figure. "
            "Install it in the current environment or set SAVE_PLOT=False/SHOW_PLOT=False and remove plotting call."
        ) from ex

    # Plot in ENU meters (East/North). This makes small trajectory differences visible,
    # unlike lat/lon plots where meter-level gaps are visually compressed.
    truth_x = np.asarray(truth_enu[:, 0], dtype=float)
    truth_y = np.asarray(truth_enu[:, 1], dtype=float)
    graph_x = np.asarray(graph_pred_enu[:, 0], dtype=float)
    graph_y = np.asarray(graph_pred_enu[:, 1], dtype=float)
    ekf_x = np.asarray(ekf_pred_enu[:, 0], dtype=float)
    ekf_y = np.asarray(ekf_pred_enu[:, 1], dtype=float)
    ukf_x = np.asarray(ukf_pred_enu[:, 0], dtype=float)
    ukf_y = np.asarray(ukf_pred_enu[:, 1], dtype=float)
    if irls_pred_enu is not None:
        irls_x = np.asarray(irls_pred_enu[:, 0], dtype=float)
        irls_y = np.asarray(irls_pred_enu[:, 1], dtype=float)
    else:
        irls_x = irls_y = None

    gps_x = np.asarray(gps_raw_enu[:, 0], dtype=float)
    gps_y = np.asarray(gps_raw_enu[:, 1], dtype=float)
    valid_gps = np.isfinite(gps_raw_enu[:, 0]) & np.isfinite(gps_raw_enu[:, 1]) & np.isfinite(gps_raw_enu[:, 2])

    truth_z = np.asarray(truth_enu[:, 2], dtype=float)
    graph_z = np.asarray(graph_pred_enu[:, 2], dtype=float)
    ekf_z = np.asarray(ekf_pred_enu[:, 2], dtype=float)
    ukf_z = np.asarray(ukf_pred_enu[:, 2], dtype=float)
    gps_z = np.asarray(gps_raw_enu[:, 2], dtype=float)
    if irls_pred_enu is not None:
        irls_z = np.asarray(irls_pred_enu[:, 2], dtype=float)
    else:
        irls_z = None

    batch_norm = os.path.normpath(batch_dir)
    batch_name = os.path.basename(batch_norm.rstrip("\\/"))
    dataset_name = os.path.basename(os.path.dirname(batch_norm))
    title_prefix = f"{dataset_name} | {batch_name} | {uav_id}"

    n_t = int(len(truth_enu))
    if truth_timestamps is not None:
        t = np.asarray(truth_timestamps, dtype=float).reshape(-1)
        if t.shape[0] != n_t:
            t = np.arange(n_t, dtype=float)
    else:
        t = np.arange(n_t, dtype=float)
    finite_t = np.isfinite(t)
    if np.any(finite_t):
        t0 = float(t[finite_t][0])
        t_plot = t.astype(float) - t0
    else:
        t_plot = np.arange(n_t, dtype=float)
    t_plot = np.where(np.isfinite(t_plot), t_plot, np.nan)

    def _err_series(pred: np.ndarray) -> np.ndarray:
        p = np.asarray(pred, dtype=float)
        gt = np.asarray(truth_enu, dtype=float)
        if p.shape != gt.shape:
            m = min(len(p), len(gt))
            p = p[:m]
            gt = gt[:m]
        err = np.linalg.norm(p - gt, axis=1)
        valid = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(gt), axis=1)
        err = np.asarray(err, dtype=float)
        err[~valid] = np.nan
        return err

    def _save_or_show(fig, path: str):
        if SAVE_PLOT:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)

    out_base, out_ext = os.path.splitext(out_png)
    if not out_ext:
        out_ext = ".png"
    xy_path = f"{out_base}_xy{out_ext}"
    xyz_path = f"{out_base}_xyz{out_ext}"
    err_path = f"{out_base}_err{out_ext}"

    # ---- Figure 1: XY trajectory (meters) ----
    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    ax.plot(truth_x, truth_y, color="black", linewidth=2.2, label="Truth")
    ax.plot(graph_x, graph_y, color="#c0392b", linewidth=1.8, alpha=0.95, label=USER_METHOD_LABEL)
    ax.plot(ekf_x, ekf_y, color="#2980b9", linewidth=1.6, alpha=0.95, label="EKF")
    ax.plot(ukf_x, ukf_y, color="#27ae60", linewidth=1.6, alpha=0.95, label="UKF")
    if irls_x is not None and irls_y is not None:
        ax.plot(irls_x, irls_y, color="#8e44ad", linewidth=1.5, alpha=0.95, label="IRLS")
    ax.scatter(gps_x[valid_gps], gps_y[valid_gps], s=18, color="#f39c12", alpha=0.55, label="RAW GPS")

    all_x_parts = [truth_x, graph_x, ekf_x, ukf_x, gps_x[valid_gps]]
    all_y_parts = [truth_y, graph_y, ekf_y, ukf_y, gps_y[valid_gps]]
    if irls_x is not None and irls_y is not None:
        all_x_parts.append(irls_x)
        all_y_parts.append(irls_y)
    all_x = np.concatenate([a[np.isfinite(a)] for a in all_x_parts if a is not None and np.any(np.isfinite(a))]) if any(
        a is not None and np.any(np.isfinite(a)) for a in all_x_parts
    ) else np.array([0.0])
    all_y = np.concatenate([a[np.isfinite(a)] for a in all_y_parts if a is not None and np.any(np.isfinite(a))]) if any(
        a is not None and np.any(np.isfinite(a)) for a in all_y_parts
    ) else np.array([0.0])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    margin_x = max(0.05 * x_span, 20.0)
    margin_y = max(0.05 * y_span, 20.0)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    ax.set_title(f"XY Trajectory (ENU meters) | {title_prefix}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    _save_or_show(fig, xy_path)

    # ---- Figure 2: XYZ vs time ----
    fig_xyz, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=140, sharex=True)
    series_by_method = [
        ("Truth", "#000000", (truth_x, truth_y, truth_z)),
        (USER_METHOD_LABEL, "#c0392b", (graph_x, graph_y, graph_z)),
        ("EKF", "#2980b9", (ekf_x, ekf_y, ekf_z)),
        ("UKF", "#27ae60", (ukf_x, ukf_y, ukf_z)),
    ]
    if irls_x is not None and irls_y is not None and irls_z is not None:
        series_by_method.append(("IRLS", "#8e44ad", (irls_x, irls_y, irls_z)))
    series_by_method.append(("RAW GPS", "#f39c12", (gps_x, gps_y, gps_z)))
    axis_names = ["East (m)", "North (m)", "Up (m)"]

    for axis_idx, ax_i in enumerate(axes):
        for name, color, comp in series_by_method:
            y = np.asarray(comp[axis_idx], dtype=float)
            if name == "RAW GPS":
                ax_i.plot(t_plot, y, color=color, linewidth=1.0, alpha=0.7, marker="o", markersize=2.5, label=name)
            elif name == "Truth":
                ax_i.plot(t_plot, y, color=color, linewidth=2.0, alpha=0.95, label=name)
            else:
                ax_i.plot(t_plot, y, color=color, linewidth=1.3, alpha=0.92, label=name)
        ax_i.set_ylabel(axis_names[axis_idx])
        ax_i.grid(True, alpha=0.25)
        ax_i.ticklabel_format(style="plain", useOffset=False)
    axes[0].set_title(f"XYZ Components vs Time | {title_prefix}")
    axes[-1].set_xlabel("Time (s, relative)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", ncol=3, fontsize=9)
    fig_xyz.tight_layout()
    _save_or_show(fig_xyz, xyz_path)

    # ---- Figure 3: Position error vs time ----
    fig_err, ax_err = plt.subplots(figsize=(13, 6.5), dpi=140)
    err_series = [
        (USER_METHOD_LABEL, "#c0392b", _err_series(graph_pred_enu)),
        ("EKF", "#2980b9", _err_series(ekf_pred_enu)),
        ("UKF", "#27ae60", _err_series(ukf_pred_enu)),
        ("RAW GPS", "#f39c12", _err_series(gps_raw_enu)),
    ]
    if irls_pred_enu is not None:
        err_series.insert(3, ("IRLS", "#8e44ad", _err_series(irls_pred_enu)))
    for name, color, e in err_series:
        if name == "RAW GPS":
            ax_err.plot(t_plot[: len(e)], e, color=color, linewidth=1.1, alpha=0.8, marker="o", markersize=2.5, label=name)
        else:
            ax_err.plot(t_plot[: len(e)], e, color=color, linewidth=1.5, alpha=0.95, label=name)
    ax_err.set_title(f"Position Error vs Time | {title_prefix}")
    ax_err.set_xlabel("Time (s, relative)")
    ax_err.set_ylabel("3D Position Error (m)")
    ax_err.ticklabel_format(style="plain", useOffset=False)
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="best", ncol=3)
    fig_err.tight_layout()
    _save_or_show(fig_err, err_path)

    if SAVE_PLOT:
        print(f"[Output] plots saved -> {xy_path} | {xyz_path} | {err_path}")


def main():
    run_output_dir = _dataset_output_dir(DATASET_BATCH_DIR)
    os.makedirs(run_output_dir, exist_ok=True)
    truth, mod_frames = _load_batch_frames(DATASET_BATCH_DIR)
    id_col, all_uavs = _list_uavs(truth)
    if EVAL_ALL_UAVS:
        uavs = list(all_uavs[: int(MAX_UAVS)]) if int(MAX_UAVS) > 0 else list(all_uavs)
    else:
        _, uav_single, _ = _select_uav(truth, TARGET_UAV_ID, TARGET_UAV_INDEX)
        uavs = [uav_single]

    print(f"[Compare] batch={DATASET_BATCH_DIR}")
    print(f"[Compare] uav_count={len(uavs)} (available={len(all_uavs)}) | eval_all={EVAL_ALL_UAVS}")
    print(f"[Compare] user_method={USER_METHOD_LABEL} | model={USER_MODEL_PATH}")
    if USER_EVAL_STRIDE_OVERRIDE is not None:
        print(f"[Compare] user_method_eval_stride_override={USER_EVAL_STRIDE_OVERRIDE}")

    # Load graph-fusion model/runtime once and reuse across UAVs.
    graph_runner = _load_graph_fusion_runner(DATASET_BATCH_DIR)
    graph_runtime = graph_runner.get("runtime", {})
    try:
        print(
            f"[Compare] graph_runtime | in_dim={graph_runtime.get('in_dim')} | "
            f"window={graph_runtime.get('window_size')} | stride={graph_runtime.get('stride')} | "
            f"mods={graph_runtime.get('modalities')}"
        )
    except Exception:
        pass

    # EKF
    ekf_cfg = ekf_fusion.EkfConfig(
        align_tolerance_s=float(ALIGN_TOLERANCE_S),
        process_accel_sigma_mps2=float(PROCESS_ACCEL_SIGMA),
        use_velocity_measurement=bool(USE_VELOCITY_UPDATE),
        use_speed_measurement=bool(USE_SPEED_UPDATE),
        use_confidence_scaling=bool(USE_CONFIDENCE_SCALE),
        use_quality_columns=bool(USE_QUALITY_COLUMNS),
        use_modality_q_hint=bool(USE_MODALITY_Q_HINT),
        modality_q_hint_strength=float(MODALITY_Q_HINT_STRENGTH),
    )

    # UKF
    ukf_cfg = ukf_fusion.UkfConfig(
        align_tolerance_s=float(ALIGN_TOLERANCE_S),
        process_accel_sigma_mps2=float(PROCESS_ACCEL_SIGMA),
        use_velocity_measurement=bool(USE_VELOCITY_UPDATE),
        use_speed_measurement=bool(USE_SPEED_UPDATE),
        use_confidence_scaling=bool(USE_CONFIDENCE_SCALE),
        use_quality_columns=bool(USE_QUALITY_COLUMNS),
        use_modality_q_hint=bool(USE_MODALITY_Q_HINT),
        modality_q_hint_strength=float(MODALITY_Q_HINT_STRENGTH),
        ukf_alpha=float(UKF_ALPHA),
        ukf_beta=float(UKF_BETA),
        ukf_kappa=float(UKF_KAPPA),
    )
    irls_cfg = irls_fusion.IrlsConfig(
        align_tolerance_s=float(ALIGN_TOLERANCE_S),
        use_confidence_scaling=bool(USE_CONFIDENCE_SCALE),
        use_quality_columns=bool(USE_QUALITY_COLUMNS),
        use_modality_q_hint=bool(USE_MODALITY_Q_HINT),
        modality_q_hint_strength=float(MODALITY_Q_HINT_STRENGTH),
        irls_max_iter=int(IRLS_MAX_ITER),
        irls_huber_k=float(IRLS_HUBER_K),
        irls_tol_m=float(IRLS_TOL_M),
        use_temporal_prior=bool(IRLS_USE_TEMPORAL_PRIOR),
        use_velocity_prediction=bool(IRLS_USE_VELOCITY_PRED),
        temporal_prior_sigma_m=float(IRLS_TEMPORAL_PRIOR_SIGMA_M),
    )
    per_uav_metric_rows: List[Dict[str, object]] = []
    method_order = [USER_METHOD_LABEL, "EKF", "UKF", "IRLS", "RAW_GPS"]
    ospa_tracks_by_method: Dict[str, List[Dict[str, object]]] = {m: [] for m in method_order} if OSPA_ENABLE else {}
    plot_dir = os.path.join(run_output_dir, PLOTS_DIRNAME)

    for idx, uav_id in enumerate(uavs, start=1):
        df_truth_u = truth[truth[id_col] == uav_id].sort_values("timestamp").reset_index(drop=True)
        if len(df_truth_u) == 0:
            continue
        truth_enu, lat0, lon0, alt0 = _truth_enu(df_truth_u)
        truth_ts_np = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)

        ekf_res = ekf_fusion.ekf_fuse_uav(df_truth_u=df_truth_u, mod_frames=mod_frames, cfg=ekf_cfg)
        ukf_res = ukf_fusion.ukf_fuse_uav(df_truth_u=df_truth_u, mod_frames=mod_frames, cfg=ukf_cfg)
        irls_res = irls_fusion.irls_fuse_uav(df_truth_u=df_truth_u, mod_frames=mod_frames, cfg=irls_cfg)
        graph_res = run_graph_fusion_uav(
            batch_dir=DATASET_BATCH_DIR,
            df_truth_u=df_truth_u,
            uav_id=uav_id,
            runner=graph_runner,
        )

        gps_df = mod_frames.get("gps", pd.DataFrame()).copy()
        id_col_g = _detect_id_col(gps_df)
        if id_col_g is not None:
            gps_df = gps_df[gps_df[id_col_g] == uav_id].sort_values("timestamp").reset_index(drop=True)
        raw_gps_enu, raw_gps_valid = _raw_gps_on_truth_timeline(
            df_truth_u=df_truth_u,
            gps_df=gps_df,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            align_tolerance_s=float(ALIGN_TOLERANCE_S),
        )

        m_graph = _metrics_row(USER_METHOD_LABEL, graph_res["pred_enu"], truth_enu)
        m_ekf = _metrics_row("EKF", ekf_res["pred_enu"], truth_enu)
        m_ukf = _metrics_row("UKF", ukf_res["pred_enu"], truth_enu)
        m_irls = _metrics_row("IRLS", irls_res["pred_enu"], truth_enu)
        m_gps = _metrics_row("RAW_GPS", raw_gps_enu, truth_enu, valid_mask=raw_gps_valid)

        if OSPA_ENABLE:
            ospa_tracks_by_method[USER_METHOD_LABEL].append(
                {"uav_id": uav_id, "timestamps": truth_ts_np, "pred_enu_batch": np.asarray(graph_res["pred_enu"]), "gt_enu_batch": truth_enu}
            )
            ospa_tracks_by_method["EKF"].append(
                {"uav_id": uav_id, "timestamps": truth_ts_np, "pred_enu_batch": np.asarray(ekf_res["pred_enu"]), "gt_enu_batch": truth_enu}
            )
            ospa_tracks_by_method["UKF"].append(
                {"uav_id": uav_id, "timestamps": truth_ts_np, "pred_enu_batch": np.asarray(ukf_res["pred_enu"]), "gt_enu_batch": truth_enu}
            )
            ospa_tracks_by_method["IRLS"].append(
                {"uav_id": uav_id, "timestamps": truth_ts_np, "pred_enu_batch": np.asarray(irls_res["pred_enu"]), "gt_enu_batch": truth_enu}
            )
            ospa_tracks_by_method["RAW_GPS"].append(
                {"uav_id": uav_id, "timestamps": truth_ts_np, "pred_enu_batch": np.asarray(raw_gps_enu), "gt_enu_batch": truth_enu}
            )

        per_uav_metric_rows.extend(
            [
                _add_batch_uav_fields(m_graph, DATASET_BATCH_DIR, uav_id),
                _add_batch_uav_fields(m_ekf, DATASET_BATCH_DIR, uav_id),
                _add_batch_uav_fields(m_ukf, DATASET_BATCH_DIR, uav_id),
                _add_batch_uav_fields(m_irls, DATASET_BATCH_DIR, uav_id),
                _add_batch_uav_fields(m_gps, DATASET_BATCH_DIR, uav_id),
            ]
        )

        print(
            f"[{idx:03d}/{len(uavs):03d}] {uav_id} | "
            f"{USER_METHOD_LABEL}:RMSE={float(m_graph['RMSE']):.3f},MAE={float(m_graph['MAE']):.3f} | "
            f"EKF:RMSE={float(m_ekf['RMSE']):.3f},MAE={float(m_ekf['MAE']):.3f} | "
            f"UKF:RMSE={float(m_ukf['RMSE']):.3f},MAE={float(m_ukf['MAE']):.3f} | "
            f"IRLS:RMSE={float(m_irls['RMSE']):.3f},MAE={float(m_irls['MAE']):.3f} | "
            f"RAW_GPS:RMSE={float(m_gps['RMSE']):.3f},MAE={float(m_gps['MAE']):.3f}"
        )

        if EVAL_ALL_UAVS and SAVE_PLOT and int(PLOT_FIRST_N_UAVS) > 0 and idx <= int(PLOT_FIRST_N_UAVS):
            out_png = os.path.join(plot_dir, f"{idx:03d}_{uav_id}.png")
            _plot_comparison(
                out_png=out_png,
                uav_id=uav_id,
                batch_dir=DATASET_BATCH_DIR,
                lat0=lat0,
                lon0=lon0,
                alt0=alt0,
                truth_enu=truth_enu,
                graph_pred_enu=np.asarray(graph_res["pred_enu"]),
                ekf_pred_enu=np.asarray(ekf_res["pred_enu"]),
                ukf_pred_enu=np.asarray(ukf_res["pred_enu"]),
                irls_pred_enu=np.asarray(irls_res["pred_enu"]),
                gps_raw_enu=raw_gps_enu,
                truth_timestamps=truth_ts_np,
            )
        elif (not EVAL_ALL_UAVS) and SAVE_PLOT:
            # single-uav mode keeps the original single plot name
            out_png = os.path.join(run_output_dir, PLOT_FILENAME)
            _plot_comparison(
                out_png=out_png,
                uav_id=uav_id,
                batch_dir=DATASET_BATCH_DIR,
                lat0=lat0,
                lon0=lon0,
                alt0=alt0,
                truth_enu=truth_enu,
                graph_pred_enu=np.asarray(graph_res["pred_enu"]),
                ekf_pred_enu=np.asarray(ekf_res["pred_enu"]),
                ukf_pred_enu=np.asarray(ukf_res["pred_enu"]),
                irls_pred_enu=np.asarray(irls_res["pred_enu"]),
                gps_raw_enu=raw_gps_enu,
                truth_timestamps=truth_ts_np,
            )

    per_uav_df = pd.DataFrame(per_uav_metric_rows)
    mean_df = _build_mean_rows(per_uav_df)
    ospa_rows: List[Dict[str, object]] = []
    if OSPA_ENABLE:
        batch_name = os.path.basename(DATASET_BATCH_DIR.rstrip("\\/"))
        for m in method_order:
            tracks_m = ospa_tracks_by_method.get(m, [])
            ospa_row = _compute_batch_ospa_from_tracks(
                batch_name=batch_name,
                tracks=tracks_m,
                p=int(OSPA_P),
                cutoffs_m=tuple(float(c) for c in OSPA_CUTOFFS_M),
                include_xy=bool(OSPA_INCLUDE_XY),
            )
            ospa_row["method"] = m
            ospa_row["ospa_partial_uav_eval"] = float(len(tracks_m) < len(uavs))
            ospa_row["ospa_uav_count_expected"] = int(len(uavs))
            ospa_rows.append(_add_batch_uav_fields(ospa_row, DATASET_BATCH_DIR, "__ALL_UAVS_SET__", row_type="ospa_batch"))
    ospa_df = pd.DataFrame(ospa_rows)
    metric_parts = [per_uav_df]
    if len(mean_df) > 0:
        metric_parts.append(mean_df)
    if len(ospa_df) > 0:
        metric_parts.append(ospa_df)
    metrics_df = pd.concat(metric_parts, ignore_index=True) if len(metric_parts) > 0 else pd.DataFrame()

    metrics_csv_path = os.path.join(run_output_dir, METRICS_CSV_FILENAME)
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Output] metrics csv saved -> {metrics_csv_path}")
    if SAVE_PLOT and int(PLOT_FIRST_N_UAVS) > 0 and EVAL_ALL_UAVS:
        print(f"[Output] plots saved (first {min(int(PLOT_FIRST_N_UAVS), len(uavs))} UAVs) -> {plot_dir}")

    # Console summary: mean RMSE/MAE across all UAVs for the four methods
    print("\n=== Mean RMSE / MAE Across All UAVs (same dataset) ===")
    if len(mean_df) == 0:
        print("No metrics to summarize.")
    else:
        mean_show = mean_df.copy()
        mean_show["_ord"] = mean_show["method"].apply(lambda x: method_order.index(x) if x in method_order else 999)
        mean_show = mean_show.sort_values(["_ord", "method"]).drop(columns=["_ord"])
        for _, r in mean_show.iterrows():
            print(
                f"{str(r['method']):>12s} | RMSE={float(r['RMSE']):9.3f} m | "
                f"MAE={float(r['MAE']):9.3f} m | coverage={float(r['coverage_ratio']):.3f} "
                f"({float(r['n_eval']):.1f}/{float(r['n_total']):.1f} avg)"
            )

    if OSPA_ENABLE and len(ospa_df) > 0:
        print(f"\n=== OSPA Across All UAVs (set metric, per-timestamp, p={int(OSPA_P)}) ===")
        ospa_show = ospa_df.copy()
        ospa_show["_ord"] = ospa_show["method"].apply(lambda x: method_order.index(x) if x in method_order else 999)
        ospa_show = ospa_show.sort_values(["_ord", "method"]).drop(columns=["_ord"])
        for _, r in ospa_show.iterrows():
            parts = []
            for c in OSPA_CUTOFFS_M:
                ck = _cutoff_key(float(c))
                key3_mean = f"ospa3d_p{int(OSPA_P)}_c{ck}_mean"
                key3_p95 = f"ospa3d_p{int(OSPA_P)}_c{ck}_p95"
                if key3_mean in r.index:
                    v_mean = r.get(key3_mean, np.nan)
                    v_p95 = r.get(key3_p95, np.nan)
                    parts.append(f"3D@c{float(c):g}:mean={float(v_mean):.3f},p95={float(v_p95):.3f}")
            print(
                f"{str(r['method']):>12s} | frames={int(float(r.get('ospa_frame_count', 0) or 0))} | "
                + " | ".join(parts)
            )

    # Save run config snapshot for reproducibility (same csv request remains metrics; this is optional print only)
    print("\n[Config]")
    print(
        {
            "DATASET_BATCH_DIR": DATASET_BATCH_DIR,
            "RUN_OUTPUT_DIR": run_output_dir,
            "EVAL_ALL_UAVS": EVAL_ALL_UAVS,
            "MAX_UAVS": MAX_UAVS,
            "PLOT_FIRST_N_UAVS": PLOT_FIRST_N_UAVS,
            "TARGET_UAV_ID": TARGET_UAV_ID,
            "TARGET_UAV_INDEX": TARGET_UAV_INDEX,
            "USER_METHOD_LABEL": USER_METHOD_LABEL,
            "USER_MODEL_PATH": USER_MODEL_PATH,
            "USER_NORM_PATH": USER_NORM_PATH,
            "USER_EVAL_STRIDE_OVERRIDE": USER_EVAL_STRIDE_OVERRIDE,
            "ALIGN_TOLERANCE_S": ALIGN_TOLERANCE_S,
            "PROCESS_ACCEL_SIGMA": PROCESS_ACCEL_SIGMA,
            "USE_VELOCITY_UPDATE": USE_VELOCITY_UPDATE,
            "USE_SPEED_UPDATE": USE_SPEED_UPDATE,
            "USE_MODALITY_Q_HINT": USE_MODALITY_Q_HINT,
            "MODALITY_Q_HINT_STRENGTH": MODALITY_Q_HINT_STRENGTH,
            "UKF_ALPHA": UKF_ALPHA,
            "UKF_BETA": UKF_BETA,
            "UKF_KAPPA": UKF_KAPPA,
            "IRLS_MAX_ITER": IRLS_MAX_ITER,
            "IRLS_HUBER_K": IRLS_HUBER_K,
            "IRLS_TOL_M": IRLS_TOL_M,
            "IRLS_USE_TEMPORAL_PRIOR": IRLS_USE_TEMPORAL_PRIOR,
            "IRLS_USE_VELOCITY_PRED": IRLS_USE_VELOCITY_PRED,
            "IRLS_TEMPORAL_PRIOR_SIGMA_M": IRLS_TEMPORAL_PRIOR_SIGMA_M,
            "OSPA_ENABLE": OSPA_ENABLE,
            "OSPA_P": OSPA_P,
            "OSPA_CUTOFFS_M": OSPA_CUTOFFS_M,
            "OSPA_INCLUDE_XY": OSPA_INCLUDE_XY,
        }
    )


if __name__ == "__main__":
    main()
