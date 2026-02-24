import os
from typing import Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # register 3D projection

import inference as inf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"../datasetBuilder/dataset-processed/test-datasets/scenario_ultra_precision_eval_100x60/"
MODEL_PATH = os.path.join(BASE_DIR, "./model_result/graph_fusion_model_v2.8.pt")
NORM_PATH = os.path.join(BASE_DIR, "./model_result/graph_norm_v2.8.pth")
DEVICE = inf.DEVICE

OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results_v2.8_ultra")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_FIG = True
MAX_UAVS = 20  # 0 means no limit
EVAL_STRIDE_OVERRIDE = 1  # int or None

# Boundary blend (same strategy as inference)
MERGE_EDGE_TAPER_MIN = 0.12
WARMUP_POINTS = 28
WARMUP_MIN_COVERAGE = 5.0
TAIL_POINTS = 40

# Advanced metrics
RPE_HORIZONS = (1, 5, 10)
OUTLIER_THRESHOLDS_M = (20.0, 50.0, 100.0, 200.0)
BOUNDARY_K = 20

# 3D trajectory plot settings (fixed Z-axis ticks for readability)
TRAJ3D_Z_TICK_STEP_M = 10.0
TRAJ3D_Z_MARGIN_M = 5.0

# OSPA (multi-target set metric) settings
OSPA_ENABLE = True
OSPA_P = 2
OSPA_CUTOFFS_M = (20.0, 50.0)


# ==============================================================
# METRICS
# ==============================================================
def _nan() -> float:
    return float(np.nan)


def calc_err(pred, gt):
    if pred is None or gt is None or len(pred) == 0:
        return {"MSE": _nan(), "RMSE": _nan(), "MAE": _nan(), "MEDAE": _nan(), "P90": _nan(), "P95": _nan(), "MAX": _nan()}
    diff = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    dist = np.linalg.norm(diff, axis=1)
    return {
        "MSE": float(np.mean(dist**2)),
        "RMSE": float(np.sqrt(np.mean(dist**2))),
        "MAE": float(np.mean(dist)),
        "MEDAE": float(np.median(dist)),
        "P90": float(np.percentile(dist, 90)),
        "P95": float(np.percentile(dist, 95)),
        "MAX": float(np.max(dist)),
    }


def calc_z_err(pred_z, gt_z):
    if pred_z is None or gt_z is None or len(pred_z) == 0:
        return {"MSE": _nan(), "RMSE": _nan(), "MAE": _nan(), "MEDAE": _nan(), "P90": _nan(), "P95": _nan(), "MAX": _nan()}
    diff = np.abs(np.asarray(pred_z, dtype=float) - np.asarray(gt_z, dtype=float))
    return {
        "MSE": float(np.mean(diff**2)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "MAE": float(np.mean(diff)),
        "MEDAE": float(np.median(diff)),
        "P90": float(np.percentile(diff, 90)),
        "P95": float(np.percentile(diff, 95)),
        "MAX": float(np.max(diff)),
    }


def _ospa_console_summary(batch_name: str, ospa_row: Dict[str, float]) -> str:
    parts = [f"[OSPA] {batch_name}"]
    p = int(OSPA_P)
    for c in OSPA_CUTOFFS_M:
        c_key = str(int(round(float(c))))
        xy_key = f"ospa_xy_p{p}_c{c_key}_mean"
        d3_key = f"ospa3d_p{p}_c{c_key}_mean"
        if xy_key in ospa_row and np.isfinite(float(ospa_row[xy_key])):
            parts.append(f"XY@{c_key}={float(ospa_row[xy_key]):.3f}")
        if d3_key in ospa_row and np.isfinite(float(ospa_row[d3_key])):
            parts.append(f"3D@{c_key}={float(ospa_row[d3_key]):.3f}")
    return " | ".join(parts)


def _rpe_rmse(pred: np.ndarray, gt: np.ndarray, horizon: int) -> float:
    if len(pred) <= horizon:
        return _nan()
    rel_pred = pred[horizon:] - pred[:-horizon]
    rel_gt = gt[horizon:] - gt[:-horizon]
    d = np.linalg.norm(rel_pred - rel_gt, axis=1)
    return float(np.sqrt(np.mean(d**2)))


def _dtw_distance_2d(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) == 0 or len(q) == 0:
        return _nan()
    n, m = len(p), len(q)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(p[i - 1] - q[j - 1])
            dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def _discrete_frechet_2d(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) == 0 or len(q) == 0:
        return _nan()
    n, m = len(p), len(q)
    ca = np.full((n, m), np.inf, dtype=float)
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(p[i] - q[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[-1, -1])


def _hausdorff_2d(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) == 0 or len(q) == 0:
        return _nan()
    dmat = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    h_pq = np.max(np.min(dmat, axis=1))
    h_qp = np.max(np.min(dmat, axis=0))
    return float(max(h_pq, h_qp))


def _along_cross_track(pred_xy: np.ndarray, gt_xy: np.ndarray) -> Dict[str, float]:
    if len(pred_xy) == 0:
        return {"along_rmse": _nan(), "cross_rmse": _nan(), "along_bias": _nan(), "cross_bias": _nan()}

    tang = np.gradient(gt_xy, axis=0)
    denom = np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    t_hat = tang / denom
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)

    err = pred_xy - gt_xy
    along = np.sum(err * t_hat, axis=1)
    cross = np.sum(err * n_hat, axis=1)

    return {
        "along_rmse": float(np.sqrt(np.mean(along**2))),
        "cross_rmse": float(np.sqrt(np.mean(cross**2))),
        "along_bias": float(np.mean(along)),
        "cross_bias": float(np.mean(cross)),
    }

def _boundary_rmse(dist: np.ndarray, k: int):
    if len(dist) == 0:
        return _nan(), _nan()
    kk = min(max(int(k), 1), len(dist))
    return (
        float(np.sqrt(np.mean(dist[:kk] ** 2))),
        float(np.sqrt(np.mean(dist[-kk:] ** 2))),
    )


def _outlier_ratios(dist: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    if len(dist) == 0:
        return {f"outlier_{int(th)}m_ratio": _nan() for th in thresholds}
    return {f"outlier_{int(th)}m_ratio": float(np.mean(dist > float(th))) for th in thresholds}


def _jerk_mean(traj_xyz: np.ndarray) -> float:
    if len(traj_xyz) < 4:
        return _nan()
    v = np.diff(traj_xyz, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)
    return float(np.mean(np.linalg.norm(j, axis=1)))


def _hungarian_min_cost_rect(cost: np.ndarray) -> float:
    """
    Exact minimum-cost assignment for a rectangular cost matrix using a
    Hungarian-style O(n^3) algorithm.
    Returns min sum over assigning each row to a unique column.
    """
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        raise ValueError("cost must be 2D")
    m, n = c.shape
    if m == 0:
        return 0.0
    if n == 0:
        return float(np.inf)
    if m > n:
        # Ensure rows <= cols; transposition preserves optimal total cost.
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
    """
    OSPA distance between two finite sets of points (XY or XYZ rows).
    """
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

    m = int(x.shape[0])
    n = int(y.shape[0])
    if x.shape[1] != y.shape[1]:
        raise ValueError("pred_set and gt_set must have same point dimension")

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


def _compute_batch_ospa_from_tracks(
    batch_name: str,
    tracks: Sequence[Dict],
    p: int = 2,
    cutoffs_m: Sequence[float] = (20.0, 50.0),
) -> Dict[str, float]:
    if len(tracks) == 0:
        return {"batch": batch_name, "ospa_frame_count": 0, "ospa_uav_count": 0}

    pred_map: Dict[float, list] = {}
    gt_map: Dict[float, list] = {}

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
            pv = pred[i]
            gv = gt[i]
            if np.all(np.isfinite(pv)):
                pred_map.setdefault(t, []).append(pv)
            if np.all(np.isfinite(gv)):
                gt_map.setdefault(t, []).append(gv)

    frame_keys = sorted(set(pred_map.keys()) | set(gt_map.keys()))
    if len(frame_keys) == 0:
        return {"batch": batch_name, "ospa_frame_count": 0, "ospa_uav_count": len(tracks)}

    frame_rows = []
    for t in frame_keys:
        pred_pts = np.asarray(pred_map.get(t, []), dtype=float)
        gt_pts = np.asarray(gt_map.get(t, []), dtype=float)
        if pred_pts.ndim == 1:
            pred_pts = pred_pts.reshape((-1, 3)) if pred_pts.size > 0 else np.zeros((0, 3), dtype=float)
        if gt_pts.ndim == 1:
            gt_pts = gt_pts.reshape((-1, 3)) if gt_pts.size > 0 else np.zeros((0, 3), dtype=float)
        row = {
            "timestamp": t,
            "pred_cardinality": int(pred_pts.shape[0]),
            "gt_cardinality": int(gt_pts.shape[0]),
        }
        for c in cutoffs_m:
            c_key = str(int(round(float(c))))
            row[f"ospa3d_p{int(p)}_c{c_key}"] = ospa_distance(pred_pts, gt_pts, p=p, cutoff_m=float(c))
            row[f"ospa_xy_p{int(p)}_c{c_key}"] = ospa_distance(pred_pts[:, :2], gt_pts[:, :2], p=p, cutoff_m=float(c))
        frame_rows.append(row)

    df_f = pd.DataFrame(frame_rows)
    out = {
        "batch": batch_name,
        "ospa_frame_count": int(len(df_f)),
        "ospa_uav_count": int(len(tracks)),
    }
    metric_cols = [c for c in df_f.columns if c.startswith("ospa")]
    for c in metric_cols:
        vals = pd.to_numeric(df_f[c], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{c}_mean"] = float(np.mean(vals)) if vals.size > 0 else np.nan
        out[f"{c}_median"] = float(np.median(vals)) if vals.size > 0 else np.nan
        out[f"{c}_p95"] = float(np.percentile(vals, 95)) if vals.size > 0 else np.nan
    return out


def compute_advanced_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    dist3 = np.linalg.norm(diff, axis=1)
    pred_xy = pred[:, :2]
    gt_xy = gt[:, :2]
    dist2 = np.linalg.norm(pred_xy - gt_xy, axis=1)

    out: Dict[str, float] = {
        "ade_3d": float(np.mean(dist3)),
        "fde_3d": float(dist3[-1]),
        "ade_xy": float(np.mean(dist2)),
        "fde_xy": float(dist2[-1]),
        "dtw_xy": _dtw_distance_2d(pred_xy, gt_xy),
        "frechet_xy": _discrete_frechet_2d(pred_xy, gt_xy),
        "hausdorff_xy": _hausdorff_2d(pred_xy, gt_xy),
        "jerk_mean_pred": _jerk_mean(pred),
        "jerk_mean_gt": _jerk_mean(gt),
    }

    head_rmse, tail_rmse = _boundary_rmse(dist3, BOUNDARY_K)
    out["head_rmse_3d"] = head_rmse
    out["tail_rmse_3d"] = tail_rmse

    out.update(_outlier_ratios(dist3, OUTLIER_THRESHOLDS_M))
    out.update(_along_cross_track(pred_xy, gt_xy))

    for h in RPE_HORIZONS:
        out[f"rpe{h}_rmse_3d"] = _rpe_rmse(pred, gt, h)

    if np.isfinite(out["jerk_mean_pred"]) and np.isfinite(out["jerk_mean_gt"]) and out["jerk_mean_gt"] > 1e-9:
        out["jerk_ratio_pred_gt"] = float(out["jerk_mean_pred"] / out["jerk_mean_gt"])
    else:
        out["jerk_ratio_pred_gt"] = _nan()

    return out


# ==============================================================
# VIS
# ==============================================================
def plot_modality_bar(uav, batch_name, fusion, mod_errs):
    labels = ["Fusion"] + [m.upper() for m in mod_errs.keys()]
    values = [fusion.get("RMSE", np.nan)] + [mod_errs[m].get("RMSE", np.nan) for m in mod_errs.keys()]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("RMSE (m)")
    plt.title(f"RMSE Comparison - {batch_name} - {uav}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{batch_name}_{uav}_modality_rmse.png"))
    plt.close()


def plot_traj_and_error(uav, batch_name, gt, pred):
    dist = np.linalg.norm(pred - gt, axis=1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "k-", label="Truth")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "r--", label="Fusion")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(f"Trajectory XYZ - {batch_name} - {uav}")

    # Use fixed Z tick spacing (not auto ticks) and snap limits to that grid.
    z_all = np.concatenate([gt[:, 2], pred[:, 2]], axis=0).astype(float)
    z_all = z_all[np.isfinite(z_all)]
    if z_all.size > 0:
        zmin = float(np.min(z_all)) - float(TRAJ3D_Z_MARGIN_M)
        zmax = float(np.max(z_all)) + float(TRAJ3D_Z_MARGIN_M)
        step = max(float(TRAJ3D_Z_TICK_STEP_M), 1e-6)
        zmin = np.floor(zmin / step) * step
        zmax = np.ceil(zmax / step) * step
        if zmax <= zmin:
            zmax = zmin + step
        ax.set_zlim(zmin, zmax)
        ax.zaxis.set_major_locator(MultipleLocator(base=step))

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{batch_name}_{uav}_traj_xyz.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(gt[:, 0], gt[:, 1], "k-", label="Truth")
    plt.plot(pred[:, 0], pred[:, 1], "r--", label="Fusion")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title(f"Trajectory - {batch_name} - {uav}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{batch_name}_{uav}_traj_xy.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(dist, color="tab:red")
    plt.xlabel("Time Index")
    plt.ylabel("3D Error (m)")
    plt.title(f"Error Curve - {batch_name} - {uav}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{batch_name}_{uav}_error_curve.png"))
    plt.close()


def modality_metrics(df_truth_u, df_mod_u, lat0, lon0, alt0, align_tolerance_s):
    gt_m, obs_m = inf.modality_series_enu(df_truth_u, df_mod_u, lat0, lon0, alt0, align_tolerance_s)
    err3d = calc_err(obs_m, gt_m)
    try:
        obs_arr = np.asarray(obs_m, dtype=float)
        gt_arr = np.asarray(gt_m, dtype=float)
    except Exception:
        obs_arr = np.zeros((0, 3), dtype=float)
        gt_arr = np.zeros((0, 3), dtype=float)

    if obs_arr.ndim != 2 or gt_arr.ndim != 2 or obs_arr.shape[0] == 0 or gt_arr.shape[0] == 0:
        err_xy = {"RMSE": _nan()}
        err_z = {"RMSE": _nan()}
    else:
        n = min(len(obs_arr), len(gt_arr))
        obs_arr = obs_arr[:n]
        gt_arr = gt_arr[:n]
        if obs_arr.shape[1] >= 2 and gt_arr.shape[1] >= 2:
            err_xy = calc_err(obs_arr[:, :2], gt_arr[:, :2])
        else:
            err_xy = {"RMSE": _nan()}
        if obs_arr.shape[1] >= 3 and gt_arr.shape[1] >= 3:
            err_z = calc_z_err(obs_arr[:, 2], gt_arr[:, 2])
        else:
            err_z = {"RMSE": _nan()}

    # Keep existing keys (MSE/RMSE/MAE/P95...) as 3D metrics for compatibility,
    # and add explicit aliases/components for CSV comparison.
    out = dict(err3d)
    out["MAE_3D"] = float(err3d.get("MAE", np.nan))
    out["RMSE_XY"] = float(err_xy.get("RMSE", np.nan))
    out["RMSE_Z"] = float(err_z.get("RMSE", np.nan))
    return out

# ==============================================================
# EVALUATION
# ==============================================================
def evaluate_uav_advanced(model, batch_dir, uav, x_mean, x_std, y_mean, y_std, runtime, batch_origin_llh=None):
    truth = pd.read_csv(os.path.join(batch_dir, "truth.csv"))
    id_col = inf._detect_id_col(truth)
    if id_col is None:
        return None

    modalities = list(runtime["modalities"])
    mod_frames = {}
    for m in modalities:
        fname = "5g_a.csv" if m == "5g_a" else f"{m}.csv"
        p = os.path.join(batch_dir, fname)
        mod_frames[m] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    df_t = truth[truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
    if len(df_t) < int(runtime["window_size"]):
        return None

    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]
    e_gt, n_gt, u_gt = inf.latlon_to_enu(df_t["lat"].values, df_t["lon"].values, df_t["alt"].values, lat0, lon0, alt0)
    gt = np.stack([e_gt, n_gt, u_gt], axis=1).astype(np.float32)

    stride = int(EVAL_STRIDE_OVERRIDE) if EVAL_STRIDE_OVERRIDE is not None else int(runtime["stride"])
    in_dim = int(runtime["in_dim"])
    if in_dim >= inf.NODE_FEAT_DIM:
        windows, starts = inf.build_sparse_windows_new(
            df_truth_u=df_t,
            mod_frames=mod_frames,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            modalities=modalities,
            window_size=int(runtime["window_size"]),
            stride=stride,
            align_tolerance_s=float(runtime["align_tolerance_s"]),
        )
    else:
        windows, starts = inf.build_sparse_windows_legacy(
            df_truth_u=df_t,
            mod_frames=mod_frames,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            modalities=modalities,
            window_size=int(runtime["window_size"]),
            stride=stride,
            in_dim=in_dim,
        )
    if len(windows) == 0:
        return None

    obs_fallback_enu, obs_fallback_w = inf.build_obs_fallback_series(
        df_truth_u=df_t,
        mod_frames=mod_frames,
        modalities=modalities,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        align_tolerance_s=float(runtime["align_tolerance_s"]),
    )

    preds = []
    window_weights = []
    for w in windows:
        window_weights.append(inf.estimate_window_quality(w["node_feat"]))
        node_feat = torch.tensor(w["node_feat"], dtype=torch.float32, device=DEVICE)
        node_t = torch.tensor(w["node_t"], dtype=torch.long, device=DEVICE)
        node_m = torch.tensor(w["node_m"], dtype=torch.long, device=DEVICE)

        node_feat = inf.fit_feature_dim(node_feat, int(x_mean.numel()))
        node_feat = (node_feat - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            pred_norm = model(
                node_feat=node_feat.unsqueeze(0),
                node_t=node_t.unsqueeze(0),
                node_m=node_m.unsqueeze(0),
                node_mask=torch.ones((1, node_feat.shape[0]), dtype=torch.float32, device=DEVICE),
                window_size=int(runtime["window_size"]),
            )[0]
        preds.append((pred_norm * y_std + y_mean).cpu().numpy())

    fusion_enu, cover_weight, cover_count = inf.merge_windows(
        np.array(preds),
        starts,
        t_total=len(gt),
        window=int(runtime["window_size"]),
        window_weights=window_weights,
        edge_taper_min=MERGE_EDGE_TAPER_MIN,
    )
    fusion_enu, warmup_replaced = inf.apply_warmup_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        warmup_points=WARMUP_POINTS,
        min_coverage=WARMUP_MIN_COVERAGE,
    )
    fusion_enu, tail_replaced = inf.apply_tail_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        tail_points=TAIL_POINTS,
        min_coverage=WARMUP_MIN_COVERAGE,
    )

    fusion_xyz = calc_err(fusion_enu, gt)
    fusion_xy = calc_err(fusion_enu[:, :2], gt[:, :2])
    z_err = calc_z_err(fusion_enu[:, 2], gt[:, 2])
    adv = compute_advanced_metrics(fusion_enu, gt)

    mod_errs = {}
    for m in modalities:
        df_m_u = mod_frames[m]
        id_col_m = inf._detect_id_col(df_m_u)
        if id_col_m is not None:
            df_m_u = df_m_u[df_m_u[id_col_m] == uav]
        mod_errs[m] = modality_metrics(df_t, df_m_u, lat0, lon0, alt0, float(runtime["align_tolerance_s"]))

    win_cov = []
    win_full = []
    win_confq = []
    win_trans = []
    for w in windows:
        sm = w.get("sample_meta", {})
        try:
            if isinstance(sm, dict):
                if "coverage_ratio" in sm:
                    win_cov.append(float(sm["coverage_ratio"]))
                if "full_modal_ratio" in sm:
                    win_full.append(float(sm["full_modal_ratio"]))
                if "conf_quality_ratio" in sm:
                    win_confq.append(float(sm["conf_quality_ratio"]))
                if "transition_ratio" in sm:
                    win_trans.append(float(sm["transition_ratio"]))
        except Exception:
            pass

    best_single_rmse = np.nan
    best_single_name = None
    for m in modalities:
        rm = float(mod_errs.get(m, {}).get("RMSE", np.nan))
        if np.isfinite(rm) and (not np.isfinite(best_single_rmse) or rm < best_single_rmse):
            best_single_rmse = rm
            best_single_name = m
    fusion_vs_best_single_gap = (
        float(fusion_xyz["RMSE"] - best_single_rmse)
        if np.isfinite(best_single_rmse) and np.isfinite(fusion_xyz.get("RMSE", np.nan))
        else np.nan
    )

    ospa_track = None
    if batch_origin_llh is not None:
        try:
            b_lat0, b_lon0, b_alt0 = [float(v) for v in batch_origin_llh]
            pred_lat, pred_lon, pred_alt = inf.enu_to_llh(
                fusion_enu[:, 0],
                fusion_enu[:, 1],
                fusion_enu[:, 2],
                float(lat0),
                float(lon0),
                float(alt0),
            )
            pred_be, pred_bn, pred_bu = inf.latlon_to_enu(
                pred_lat,
                pred_lon,
                pred_alt,
                b_lat0,
                b_lon0,
                b_alt0,
            )
            gt_be, gt_bn, gt_bu = inf.latlon_to_enu(
                df_t["lat"].to_numpy(dtype=float),
                df_t["lon"].to_numpy(dtype=float),
                df_t["alt"].to_numpy(dtype=float),
                b_lat0,
                b_lon0,
                b_alt0,
            )
            ospa_track = {
                "uav": str(uav),
                "timestamps": pd.to_numeric(df_t["timestamp"], errors="coerce").to_numpy(dtype=float),
                "pred_enu_batch": np.stack([pred_be, pred_bn, pred_bu], axis=1).astype(np.float32),
                "gt_enu_batch": np.stack([gt_be, gt_bn, gt_bu], axis=1).astype(np.float32),
            }
        except Exception:
            ospa_track = None

    if SAVE_FIG:
        batch_name = os.path.basename(batch_dir)
        plot_modality_bar(uav=uav, batch_name=batch_name, fusion=fusion_xyz, mod_errs=mod_errs)
        plot_traj_and_error(uav=uav, batch_name=batch_name, gt=gt, pred=fusion_enu)

    return {
        "fusion_xyz": fusion_xyz,
        "fusion_xy": fusion_xy,
        "z_err": z_err,
        "advanced": adv,
        "mod_errs": mod_errs,
        "ospa_track": ospa_track,
        "diag": {
            "num_points": int(len(gt)),
            "num_windows": int(len(windows)),
            "warmup_blended": int(warmup_replaced),
            "tail_blended": int(tail_replaced),
            "head_cover_count_mean": float(np.mean(cover_count[: min(20, len(cover_count))])),
            "tail_cover_count_mean": float(np.mean(cover_count[-min(20, len(cover_count)) :])),
            "head_cover_weight_mean": float(np.mean(cover_weight[: min(20, len(cover_weight))])),
            "tail_cover_weight_mean": float(np.mean(cover_weight[-min(20, len(cover_weight)) :])),
            "win_coverage_ratio_mean": float(np.mean(win_cov)) if len(win_cov) > 0 else np.nan,
            "win_full_modal_ratio_mean": float(np.mean(win_full)) if len(win_full) > 0 else np.nan,
            "win_conf_quality_ratio_mean": float(np.mean(win_confq)) if len(win_confq) > 0 else np.nan,
            "win_transition_ratio_mean": float(np.mean(win_trans)) if len(win_trans) > 0 else np.nan,
            "best_single_rmse_3d": float(best_single_rmse) if np.isfinite(best_single_rmse) else np.nan,
            "fusion_minus_best_single_rmse_3d": fusion_vs_best_single_gap,
            "fusion_beats_best_single_rmse": float(fusion_vs_best_single_gap < 0) if np.isfinite(fusion_vs_best_single_gap) else np.nan,
            "best_single_modality": best_single_name if best_single_name is not None else "",
        },
    }


def main():
    torch.set_grad_enabled(False)

    model, x_mean, x_std, y_mean, y_std, runtime = inf.load_model_and_runtime(
        model_path=MODEL_PATH,
        norm_path=NORM_PATH,
        device=DEVICE,
    )
    print(
        f"[Runtime] model={os.path.basename(MODEL_PATH)} | "
        f"in_dim={runtime['in_dim']} | window={runtime['window_size']} | stride={runtime['stride']} | "
        f"mods={runtime['modalities']}"
    )
    if isinstance(runtime.get("load_info"), dict) and (not bool(runtime["load_info"].get("strict", True))):
        print(
            f"[Runtime][WARN] non-strict model load | missing={runtime['load_info'].get('missing_keys', 0)} "
            f"unexpected={runtime['load_info'].get('unexpected_keys', 0)}"
        )

    rows = []
    ospa_batch_rows = []
    cnt = 0
    for batch in sorted(os.listdir(DATA_ROOT)):
        batch_dir = os.path.join(DATA_ROOT, batch)
        if not os.path.isdir(batch_dir):
            continue
        truth_path = os.path.join(batch_dir, "truth.csv")
        if not os.path.exists(truth_path):
            continue

        truth = pd.read_csv(truth_path)
        id_col = inf._detect_id_col(truth)
        if id_col is None:
            continue
        batch_origin = None
        if len(truth) > 0 and all(c in truth.columns for c in ["lat", "lon", "alt"]):
            first_row = truth.iloc[0]
            batch_origin = (float(first_row["lat"]), float(first_row["lon"]), float(first_row["alt"]))
        ospa_tracks = []
        batch_total_uavs = int(truth[id_col].dropna().nunique())

        for uav in truth[id_col].dropna().unique():
            if MAX_UAVS > 0 and cnt >= MAX_UAVS:
                break

            out = evaluate_uav_advanced(
                model=model,
                batch_dir=batch_dir,
                uav=uav,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                runtime=runtime,
                batch_origin_llh=batch_origin,
            )
            if out is None:
                continue

            fusion_xyz = out["fusion_xyz"]
            adv = out["advanced"]
            print(
                f"[Eval] {batch} - {uav} | RMSE3D={fusion_xyz['RMSE']:.3f} | "
                f"MSE3D={fusion_xyz.get('MSE', np.nan):.3f} | "
                f"P95={fusion_xyz['P95']:.3f} | FDE_XY={adv.get('fde_xy', np.nan):.3f} | "
                f"DTW_XY={adv.get('dtw_xy', np.nan):.3f}"
            )
            if isinstance(out.get("ospa_track"), dict):
                ospa_tracks.append(out["ospa_track"])

            row = {
                "uav": uav,
                "batch": batch,
                "fusion_mse_3d": out["fusion_xyz"]["MSE"],
                "fusion_rmse_3d": out["fusion_xyz"]["RMSE"],
                "fusion_mae_3d": out["fusion_xyz"]["MAE"],
                "fusion_medae_3d": out["fusion_xyz"]["MEDAE"],
                "fusion_p90_3d": out["fusion_xyz"]["P90"],
                "fusion_p95_3d": out["fusion_xyz"]["P95"],
                "fusion_max_3d": out["fusion_xyz"]["MAX"],
                "fusion_mse_xy": out["fusion_xy"]["MSE"],
                "fusion_rmse_xy": out["fusion_xy"]["RMSE"],
                "fusion_mse_z": out["z_err"]["MSE"],
                "fusion_rmse_z": out["z_err"]["RMSE"],
                "fusion_p95_z": out["z_err"]["P95"],
                **out["advanced"],
                **out["diag"],
            }
            for m in runtime["modalities"]:
                key = m.replace("5g_a", "fiveg")
                row[f"{key}_mse"] = out["mod_errs"].get(m, {}).get("MSE", np.nan)
                row[f"{key}_rmse"] = out["mod_errs"].get(m, {}).get("RMSE", np.nan)
                row[f"{key}_mae_3d"] = out["mod_errs"].get(m, {}).get("MAE_3D", np.nan)
                row[f"{key}_rmse_xy"] = out["mod_errs"].get(m, {}).get("RMSE_XY", np.nan)
                row[f"{key}_rmse_z"] = out["mod_errs"].get(m, {}).get("RMSE_Z", np.nan)
                row[f"{key}_p95"] = out["mod_errs"].get(m, {}).get("P95", np.nan)
            rows.append(row)
            cnt += 1

        if MAX_UAVS > 0 and cnt >= MAX_UAVS:
            if OSPA_ENABLE and len(ospa_tracks) > 0:
                ospa_row = _compute_batch_ospa_from_tracks(
                    batch_name=batch,
                    tracks=ospa_tracks,
                    p=OSPA_P,
                    cutoffs_m=OSPA_CUTOFFS_M,
                )
                ospa_row["batch_total_uavs"] = int(batch_total_uavs)
                ospa_row["ospa_partial_uav_eval"] = float(len(ospa_tracks) < batch_total_uavs)
                ospa_batch_rows.append(ospa_row)
                print(_ospa_console_summary(batch, ospa_row))
            break

        if OSPA_ENABLE and len(ospa_tracks) > 0:
            ospa_row = _compute_batch_ospa_from_tracks(
                batch_name=batch,
                tracks=ospa_tracks,
                p=OSPA_P,
                cutoffs_m=OSPA_CUTOFFS_M,
            )
            ospa_row["batch_total_uavs"] = int(batch_total_uavs)
            ospa_row["ospa_partial_uav_eval"] = float(len(ospa_tracks) < batch_total_uavs)
            ospa_batch_rows.append(ospa_row)
            print(_ospa_console_summary(batch, ospa_row))

    if len(rows) == 0:
        print("[Eval] no valid UAV samples")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "fusion_eval_advanced.csv")
    df.to_csv(csv_path, index=False)

    metric_cols = [
        c
        for c in df.columns
        if any(k in c for k in ["mse", "rmse", "mae", "medae", "p90", "p95", "max", "ade", "fde", "dtw", "frechet", "hausdorff", "rpe", "outlier", "jerk", "cross", "along"])
    ]
    summary = df[metric_cols].agg(["mean", "median", "std"])
    summary_path = os.path.join(OUTPUT_DIR, "fusion_eval_advanced_summary.csv")
    summary.to_csv(summary_path)

    worst = df.sort_values("fusion_rmse_3d", ascending=False).head(min(20, len(df)))
    worst_path = os.path.join(OUTPUT_DIR, "fusion_eval_worst_cases.csv")
    worst.to_csv(worst_path, index=False)

    ospa_batch_path = None
    ospa_summary_path = None
    if OSPA_ENABLE and len(ospa_batch_rows) > 0:
        df_ospa = pd.DataFrame(ospa_batch_rows)
        ospa_batch_path = os.path.join(OUTPUT_DIR, "fusion_eval_ospa_batch.csv")
        df_ospa.to_csv(ospa_batch_path, index=False)
        ospa_metric_cols = [c for c in df_ospa.columns if c.startswith("ospa") and "_c" in c]
        if len(ospa_metric_cols) > 0:
            ospa_summary = df_ospa[ospa_metric_cols].agg(["mean", "median", "std"])
            ospa_summary_path = os.path.join(OUTPUT_DIR, "fusion_eval_ospa_summary.csv")
            ospa_summary.to_csv(ospa_summary_path)

    print("\n[Done] advanced evaluation finished")
    print(f"[Save] {csv_path}")
    print(f"[Save] {summary_path}")
    print(f"[Save] {worst_path}")
    if ospa_batch_path:
        print(f"[Save] {ospa_batch_path}")
    if ospa_summary_path:
        print(f"[Save] {ospa_summary_path}")


if __name__ == "__main__":
    main()
