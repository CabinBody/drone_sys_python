import os
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import inference as inf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"../datasetBuilder/dataset-processed/test-datasets/scenario_multi_source_100x60/"
MODEL_PATH = os.path.join(BASE_DIR, "graph_fusion_model_processed.pt")
NORM_PATH = os.path.join(BASE_DIR, "graph_norm_stats_processed_sparse_enu.pth")
DEVICE = inf.DEVICE

OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_FIG = True
MAX_UAVS = 20  # 0 means no limit
EVAL_STRIDE_OVERRIDE = None  # int or None

# Boundary blend (same strategy as inference)
MERGE_EDGE_TAPER_MIN = 0.25
WARMUP_POINTS = 20
WARMUP_MIN_COVERAGE = 3.0
TAIL_POINTS = 20

# Advanced metrics
RPE_HORIZONS = (1, 5, 10)
OUTLIER_THRESHOLDS_M = (20.0, 50.0, 100.0, 200.0)
BOUNDARY_K = 20


# ==============================================================
# METRICS
# ==============================================================
def _nan() -> float:
    return float(np.nan)


def calc_err(pred, gt):
    if pred is None or gt is None or len(pred) == 0:
        return {"RMSE": _nan(), "MAE": _nan(), "MEDAE": _nan(), "P90": _nan(), "P95": _nan(), "MAX": _nan()}
    diff = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    dist = np.linalg.norm(diff, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(dist**2))),
        "MAE": float(np.mean(dist)),
        "MEDAE": float(np.median(dist)),
        "P90": float(np.percentile(dist, 90)),
        "P95": float(np.percentile(dist, 95)),
        "MAX": float(np.max(dist)),
    }


def calc_z_err(pred_z, gt_z):
    if pred_z is None or gt_z is None or len(pred_z) == 0:
        return {"RMSE": _nan(), "MAE": _nan(), "MEDAE": _nan(), "P90": _nan(), "P95": _nan(), "MAX": _nan()}
    diff = np.abs(np.asarray(pred_z, dtype=float) - np.asarray(gt_z, dtype=float))
    return {
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "MAE": float(np.mean(diff)),
        "MEDAE": float(np.median(diff)),
        "P90": float(np.percentile(diff, 90)),
        "P95": float(np.percentile(diff, 95)),
        "MAX": float(np.max(diff)),
    }


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
    return calc_err(obs_m, gt_m)

# ==============================================================
# EVALUATION
# ==============================================================
def evaluate_uav_advanced(model, batch_dir, uav, x_mean, x_std, y_mean, y_std, runtime):
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
        "diag": {
            "num_points": int(len(gt)),
            "num_windows": int(len(windows)),
            "warmup_blended": int(warmup_replaced),
            "tail_blended": int(tail_replaced),
            "head_cover_count_mean": float(np.mean(cover_count[: min(20, len(cover_count))])),
            "tail_cover_count_mean": float(np.mean(cover_count[-min(20, len(cover_count)) :])),
            "head_cover_weight_mean": float(np.mean(cover_weight[: min(20, len(cover_weight))])),
            "tail_cover_weight_mean": float(np.mean(cover_weight[-min(20, len(cover_weight)) :])),
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

    rows = []
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
            )
            if out is None:
                continue

            fusion_xyz = out["fusion_xyz"]
            adv = out["advanced"]
            print(
                f"[Eval] {batch} - {uav} | RMSE3D={fusion_xyz['RMSE']:.3f} | "
                f"P95={fusion_xyz['P95']:.3f} | FDE_XY={adv.get('fde_xy', np.nan):.3f} | "
                f"DTW_XY={adv.get('dtw_xy', np.nan):.3f}"
            )

            row = {
                "uav": uav,
                "batch": batch,
                "fusion_rmse_3d": out["fusion_xyz"]["RMSE"],
                "fusion_mae_3d": out["fusion_xyz"]["MAE"],
                "fusion_medae_3d": out["fusion_xyz"]["MEDAE"],
                "fusion_p90_3d": out["fusion_xyz"]["P90"],
                "fusion_p95_3d": out["fusion_xyz"]["P95"],
                "fusion_max_3d": out["fusion_xyz"]["MAX"],
                "fusion_rmse_xy": out["fusion_xy"]["RMSE"],
                "fusion_rmse_z": out["z_err"]["RMSE"],
                "fusion_p95_z": out["z_err"]["P95"],
                **out["advanced"],
                **out["diag"],
            }
            for m in runtime["modalities"]:
                key = m.replace("5g_a", "fiveg")
                row[f"{key}_rmse"] = out["mod_errs"].get(m, {}).get("RMSE", np.nan)
                row[f"{key}_p95"] = out["mod_errs"].get(m, {}).get("P95", np.nan)
            rows.append(row)
            cnt += 1

        if MAX_UAVS > 0 and cnt >= MAX_UAVS:
            break

    if len(rows) == 0:
        print("[Eval] no valid UAV samples")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "fusion_eval_advanced.csv")
    df.to_csv(csv_path, index=False)

    metric_cols = [
        c
        for c in df.columns
        if any(k in c for k in ["rmse", "mae", "medae", "p90", "p95", "max", "ade", "fde", "dtw", "frechet", "hausdorff", "rpe", "outlier", "jerk", "cross", "along"])
    ]
    summary = df[metric_cols].agg(["mean", "median", "std"])
    summary_path = os.path.join(OUTPUT_DIR, "fusion_eval_advanced_summary.csv")
    summary.to_csv(summary_path)

    worst = df.sort_values("fusion_rmse_3d", ascending=False).head(min(20, len(df)))
    worst_path = os.path.join(OUTPUT_DIR, "fusion_eval_worst_cases.csv")
    worst.to_csv(worst_path, index=False)

    print("\n[Done] advanced evaluation finished")
    print(f"[Save] {csv_path}")
    print(f"[Save] {summary_path}")
    print(f"[Save] {worst_path}")


if __name__ == "__main__":
    main()
