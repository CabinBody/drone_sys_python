import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        BASE_POS_SIGMA_M,
        BASE_VEL_SIGMA_MPS,
        R_EARTH,
        _align_modality_rows,
        _confidence_scale,
        _detect_id_col,
        _load_batch_frames,
        _mean_metrics,
        _meas_weighted_state_init,
        _multi_modal_q_scale,
        _row_quality_sigma_hint,
        _safe_float,
        _uav_list,
        _valid_position_row,
        _valid_velocity_row,
        _vel_meas_cov,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )
except ImportError:
    from kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        BASE_POS_SIGMA_M,
        BASE_VEL_SIGMA_MPS,
        R_EARTH,
        _align_modality_rows,
        _confidence_scale,
        _detect_id_col,
        _load_batch_frames,
        _mean_metrics,
        _meas_weighted_state_init,
        _multi_modal_q_scale,
        _row_quality_sigma_hint,
        _safe_float,
        _uav_list,
        _valid_position_row,
        _valid_velocity_row,
        _vel_meas_cov,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )


RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


@dataclass
class EkfConfig:
    align_tolerance_s: float = 0.55
    process_accel_sigma_mps2: float = 6.0
    init_pos_sigma_m: float = 50.0
    init_vel_sigma_mps: float = 30.0
    use_velocity_measurement: bool = False
    use_speed_measurement: bool = False
    use_confidence_scaling: bool = True
    use_quality_columns: bool = True
    use_modality_q_hint: bool = True
    modality_q_hint_strength: float = 1.0
    gate_pos_chi2: Optional[float] = None
    gate_vel_chi2: Optional[float] = None
    gate_speed_chi2: Optional[float] = None


def _cv_F_Q(dt: float, accel_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = max(float(dt), 1e-3)
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    q = float(accel_sigma) ** 2
    q1 = (dt**4) / 4.0 * q
    q2 = (dt**3) / 2.0 * q
    q3 = (dt**2) * q
    Q1 = np.array([[q1, q2], [q2, q3]], dtype=float)
    Q = np.zeros((6, 6), dtype=float)
    Q[np.ix_([0, 3], [0, 3])] = Q1
    Q[np.ix_([1, 4], [1, 4])] = Q1
    Q[np.ix_([2, 5], [2, 5])] = Q1
    return F, Q


def _linear_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    gate_chi2: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    y = z - H @ x
    S = H @ P @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    if gate_chi2 is not None:
        d2 = float(y.T @ S_inv @ y)
        if not np.isfinite(d2) or d2 > float(gate_chi2):
            return x, P, False

    K = P @ H.T @ S_inv
    x_new = x + K @ y
    I = np.eye(P.shape[0], dtype=float)
    I_KH = I - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, True


def _ekf_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    h_x: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    gate_chi2: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    y = np.asarray(z, dtype=float).reshape(-1) - np.asarray(h_x, dtype=float).reshape(-1)
    S = H @ P @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    if gate_chi2 is not None:
        d2 = float(y.T @ S_inv @ y)
        if not np.isfinite(d2) or d2 > float(gate_chi2):
            return x, P, False

    K = P @ H.T @ S_inv
    x_new = x + K @ y
    I = np.eye(P.shape[0], dtype=float)
    I_KH = I - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, True


def _h_pos_llh(x: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    e = float(x[0])
    n = float(x[1])
    u = float(x[2])

    lat0_r = np.radians(float(lat0))
    lon0_r = np.radians(float(lon0))
    lat_r = lat0_r + n / R_EARTH
    cos_lat = np.cos(lat_r)
    cos_lat = np.clip(cos_lat, 1e-6, None)
    lon_r = lon0_r + e / (R_EARTH * cos_lat)

    return np.array([lat_r * RAD2DEG, lon_r * RAD2DEG, float(alt0) + u], dtype=float)


def _H_pos_llh(x: np.ndarray, lat0: float) -> np.ndarray:
    e = float(x[0])
    n = float(x[1])

    lat0_r = np.radians(float(lat0))
    lat_r = lat0_r + n / R_EARTH
    sin_lat = np.sin(lat_r)
    cos_lat = np.clip(np.cos(lat_r), 1e-6, None)

    H = np.zeros((3, 6), dtype=float)
    H[0, 1] = RAD2DEG / R_EARTH
    H[1, 0] = RAD2DEG / (R_EARTH * cos_lat)
    H[1, 1] = RAD2DEG * (e * sin_lat) / (R_EARTH**2 * (cos_lat**2))
    H[2, 2] = 1.0
    return H


def _pos_meas_cov_llh(
    modality: str,
    row: pd.Series,
    time_diff_s: float,
    cfg: EkfConfig,
    pred_lat_deg: float,
) -> np.ndarray:
    sigma_m = float(BASE_POS_SIGMA_M.get(modality, 80.0))
    sigma_m *= _confidence_scale(row, cfg)
    if cfg.use_quality_columns:
        hint = _row_quality_sigma_hint(row)
        if hint is not None and np.isfinite(hint):
            sigma_m = max(sigma_m, 0.6 * sigma_m + 0.4 * float(hint))
    sigma_m *= 1.0 + min(float(abs(time_diff_s)) / max(cfg.align_tolerance_s, 1e-3), 1.0)
    sigma_m = max(sigma_m, 1.0)

    lat_sigma_deg = (sigma_m / R_EARTH) * RAD2DEG
    cos_lat = np.clip(np.cos(np.radians(float(pred_lat_deg))), 1e-6, None)
    lon_sigma_deg = (sigma_m / (R_EARTH * cos_lat)) * RAD2DEG
    alt_sigma_m = max(1.2 * sigma_m, 1.0)
    return np.diag([lat_sigma_deg**2, lon_sigma_deg**2, alt_sigma_m**2]).astype(float)


def _valid_speed_row(row: Optional[pd.Series]) -> bool:
    if row is None:
        return False
    if int(_safe_float(row.get("missing_flag", 0), 0)) > 0:
        return False
    s = _safe_float(row.get("speed", np.nan), np.nan)
    return np.isfinite(s)


def _speed_meas_var(modality: str, row: pd.Series, time_diff_s: float, cfg: EkfConfig) -> float:
    sigma = float(BASE_VEL_SIGMA_MPS.get(modality, 12.0)) * 1.4
    sigma *= _confidence_scale(row, cfg)
    sigma *= 1.0 + 0.5 * min(float(abs(time_diff_s)) / max(cfg.align_tolerance_s, 1e-3), 1.0)
    sigma = max(sigma, 0.5)
    return float(sigma**2)


def _h_speed(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x[3:6], dtype=float)
    s = float(np.linalg.norm(v))
    return np.array([s], dtype=float)


def _H_speed(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x[3:6], dtype=float)
    s = float(np.linalg.norm(v))
    H = np.zeros((1, 6), dtype=float)
    if s < 1e-6:
        return H
    H[0, 3:6] = v / s
    return H


def _acoustic_q_scale(row: Optional[pd.Series], cfg: EkfConfig) -> Tuple[float, bool]:
    if (not bool(getattr(cfg, "use_modality_q_hint", True))) or row is None:
        return 1.0, False

    strength = float(np.clip(getattr(cfg, "modality_q_hint_strength", 1.0), 0.0, 1.0))
    if strength <= 0.0:
        return 1.0, False

    missing_flag = int(_safe_float(row.get("missing_flag", 0), 0))
    detected = float(np.clip(_safe_float(row.get("detected_flag", 0), 0.0), 0.0, 1.0))
    conf = float(np.clip(_safe_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0))
    energy = _safe_float(row.get("acoustic_energy", 0.0), 0.0)
    energy = float(np.clip(energy if np.isfinite(energy) else 0.0, 0.0, 1.0))

    if missing_flag > 0 or detected <= 0.0:
        return 1.0 + 0.12 * strength, False

    signal = 0.6 * conf + 0.4 * energy
    scale = 1.0 - 0.25 * strength * float(np.clip(signal, 0.0, 1.0))
    return float(max(scale, 0.70)), True


def _default_batch_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(
            base_dir,
            "..",
            "..",
            "datasetBuilder",
            "dataset-processed",
            "test-datasets",
            "scenario_eval_high_missing_mixed_100x60",
            "batch01",
        )
    )


def ekf_fuse_uav(
    df_truth_u: pd.DataFrame,
    mod_frames: Dict[str, pd.DataFrame],
    cfg: EkfConfig,
) -> Dict[str, object]:
    if len(df_truth_u) == 0:
        return {"pred_enu": np.zeros((0, 3), dtype=np.float32), "truth_enu": np.zeros((0, 3), dtype=np.float32)}

    df_truth_u = df_truth_u.sort_values("timestamp").reset_index(drop=True)
    lat0, lon0, alt0 = df_truth_u.iloc[0][["lat", "lon", "alt"]]
    t_total = len(df_truth_u)
    truth_ts = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)

    e_gt, n_gt, u_gt = latlon_to_enu(
        df_truth_u["lat"].values,
        df_truth_u["lon"].values,
        df_truth_u["alt"].values,
        lat0,
        lon0,
        alt0,
    )
    truth_enu = np.stack([e_gt, n_gt, u_gt], axis=-1).astype(np.float32)

    id_col_truth = _detect_id_col(df_truth_u)
    uav_value = df_truth_u.iloc[0][id_col_truth] if id_col_truth else None

    aligned_rows: Dict[str, List[Optional[pd.Series]]] = {}
    aligned_deltas: Dict[str, np.ndarray] = {}
    modality_usage = {
        m: {"pos_updates": 0, "vel_updates": 0, "speed_updates": 0, "gated": 0, "aligned": 0, "q_hint_hits": 0}
        for m in ALL_MODALITIES
    }

    for m in ALL_MODALITIES:
        d = mod_frames.get(m, pd.DataFrame())
        if len(d) == 0:
            aligned_rows[m] = [None] * t_total
            aligned_deltas[m] = np.full((t_total,), np.inf, dtype=float)
            continue
        d = d.copy()
        id_col_m = _detect_id_col(d)
        if id_col_m is not None and uav_value is not None:
            d = d[d[id_col_m] == uav_value]
        d = d.sort_values("timestamp").reset_index(drop=True)
        rows, deltas = _align_modality_rows(df_truth_u, d, cfg.align_tolerance_s)
        modality_usage[m]["aligned"] = int(np.sum(np.isfinite(deltas)))
        aligned_rows[m] = rows
        aligned_deltas[m] = deltas

    x, P = _meas_weighted_state_init(aligned_rows, aligned_deltas, lat0, lon0, alt0, cfg)
    pred_enu = np.zeros((t_total, 3), dtype=np.float32)
    pred_cov_trace = np.zeros((t_total,), dtype=np.float32)
    q_scale_trace = np.ones((t_total,), dtype=np.float32)

    H_vel = np.zeros((3, 6), dtype=float)
    H_vel[:, 3:6] = np.eye(3, dtype=float)

    for k in range(t_total):
        if k == 0:
            dt = 1.0
            if t_total > 1 and np.isfinite(truth_ts[1]) and np.isfinite(truth_ts[0]):
                dt = max(float(truth_ts[1] - truth_ts[0]), 1e-3)
        else:
            dt = max(float(truth_ts[k] - truth_ts[k - 1]), 1e-3)

        q_scale, q_hit_mods = _multi_modal_q_scale({m: aligned_rows[m][k] for m in ALL_MODALITIES}, cfg)
        for m in q_hit_mods:
            modality_usage[m]["q_hint_hits"] += 1
        q_scale_trace[k] = float(q_scale)
        accel_sigma = float(cfg.process_accel_sigma_mps2) * np.sqrt(max(float(q_scale), 1e-6))

        F, Q = _cv_F_Q(dt, accel_sigma)
        x = F @ x
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        for m in ALL_MODALITIES:
            row = aligned_rows[m][k]
            if row is None:
                continue
            time_diff = float(aligned_deltas[m][k]) if np.isfinite(aligned_deltas[m][k]) else np.inf

            if _valid_position_row(row):
                z_pos = np.array(
                    [
                        _safe_float(row.get("lat"), np.nan),
                        _safe_float(row.get("lon"), np.nan),
                        _safe_float(row.get("alt"), np.nan),
                    ],
                    dtype=float,
                )
                h_pos = _h_pos_llh(x, lat0, lon0, alt0)
                H_pos = _H_pos_llh(x, lat0)
                R_pos = _pos_meas_cov_llh(
                    modality=m,
                    row=row,
                    time_diff_s=time_diff,
                    cfg=cfg,
                    pred_lat_deg=float(h_pos[0]),
                )
                x_new, P_new, accepted = _ekf_update(
                    x=x,
                    P=P,
                    z=z_pos,
                    h_x=h_pos,
                    H=H_pos,
                    R=R_pos,
                    gate_chi2=cfg.gate_pos_chi2,
                )
                if accepted:
                    x, P = x_new, P_new
                    modality_usage[m]["pos_updates"] += 1
                else:
                    modality_usage[m]["gated"] += 1

            if cfg.use_velocity_measurement and _valid_velocity_row(row):
                z_v = np.array(
                    [
                        _safe_float(row.get("vx"), 0.0),
                        _safe_float(row.get("vy"), 0.0),
                        _safe_float(row.get("vz"), 0.0),
                    ],
                    dtype=float,
                )
                R_v = _vel_meas_cov(m, row, time_diff, cfg)
                x_new, P_new, accepted = _linear_update(
                    x=x,
                    P=P,
                    z=z_v,
                    H=H_vel,
                    R=R_v,
                    gate_chi2=cfg.gate_vel_chi2,
                )
                if accepted:
                    x, P = x_new, P_new
                    modality_usage[m]["vel_updates"] += 1
                else:
                    modality_usage[m]["gated"] += 1

            if cfg.use_speed_measurement and _valid_speed_row(row):
                z_s = np.array([_safe_float(row.get("speed"), 0.0)], dtype=float)
                h_s = _h_speed(x)
                H_s = _H_speed(x)
                R_s = np.array([[float(_speed_meas_var(m, row, time_diff, cfg))]], dtype=float)
                x_new, P_new, accepted = _ekf_update(
                    x=x,
                    P=P,
                    z=z_s,
                    h_x=h_s,
                    H=H_s,
                    R=R_s,
                    gate_chi2=cfg.gate_speed_chi2,
                )
                if accepted:
                    x, P = x_new, P_new
                    modality_usage[m]["speed_updates"] += 1
                else:
                    modality_usage[m]["gated"] += 1

        pred_enu[k] = x[0:3].astype(np.float32)
        pred_cov_trace[k] = float(np.trace(P[0:3, 0:3]))

    pred_lat, pred_lon, pred_alt = enu_to_llh(
        pred_enu[:, 0],
        pred_enu[:, 1],
        pred_enu[:, 2],
        lat0,
        lon0,
        alt0,
    )
    out_df = pd.DataFrame(
        {
            "timestamp": df_truth_u["timestamp"].values,
            "uav_id": [uav_value] * t_total if uav_value is not None else [None] * t_total,
            "pred_e": pred_enu[:, 0],
            "pred_n": pred_enu[:, 1],
            "pred_u": pred_enu[:, 2],
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "pred_alt": pred_alt,
            "gt_e": truth_enu[:, 0],
            "gt_n": truth_enu[:, 1],
            "gt_u": truth_enu[:, 2],
            "cov_trace_pos": pred_cov_trace,
            "q_scale": q_scale_trace,
        }
    )
    return {
        "pred_enu": pred_enu,
        "truth_enu": truth_enu,
        "lat0": float(lat0),
        "lon0": float(lon0),
        "alt0": float(alt0),
        "pred_df": out_df,
        "modality_usage": modality_usage,
    }


def evaluate_batch(
    batch_dir: str,
    cfg: EkfConfig,
    max_uavs: int = 0,
    save_csv: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    truth, mod_frames = _load_batch_frames(batch_dir)
    id_col, uavs = _uav_list(truth)
    if max_uavs and max_uavs > 0:
        uavs = uavs[: int(max_uavs)]

    print(
        f"[EKF] batch={os.path.basename(batch_dir)} | uavs={len(uavs)} | "
        f"align_tol={cfg.align_tolerance_s:.2f}s | vel={cfg.use_velocity_measurement} | "
        f"speed={cfg.use_speed_measurement} | modality_q_hint={cfg.use_modality_q_hint}"
    )
    per_uav_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for idx, uav in enumerate(uavs, start=1):
        df_t = truth[truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
        res = ekf_fuse_uav(df_t, mod_frames, cfg)
        metrics = calc_err(res["pred_enu"], res["truth_enu"])
        usage = res["modality_usage"]
        q_hint_total = int(sum(int(usage[m]["q_hint_hits"]) for m in ALL_MODALITIES))
        usage_summary = ", ".join(
            f"{m}:a{usage[m]['aligned']}/p{usage[m]['pos_updates']}/v{usage[m]['vel_updates']}/s{usage[m]['speed_updates']}/q{usage[m]['q_hint_hits']}/g{usage[m]['gated']}"
            for m in ALL_MODALITIES
        )
        print(
            f"[{idx:03d}/{len(uavs):03d}] {uav} | "
            f"RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} P95={metrics['P95']:.3f} MAX={metrics['MAX']:.3f} | "
            f"q_hint_total={q_hint_total} | "
            f"{usage_summary}"
        )
        row = {
            "uav_id": uav,
            **metrics,
            "q_hint_hits_total": q_hint_total,
        }
        per_uav_rows.append(row)
        pred_df = res["pred_df"]
        pred_df.insert(0, "batch", os.path.basename(batch_dir))
        pred_frames.append(pred_df)

    summary = _mean_metrics(
        [
            {
                "RMSE": r["RMSE"],
                "MAE": r["MAE"],
                "MEDAE": r["MEDAE"],
                "P95": r["P95"],
                "MAX": r["MAX"],
            }
            for r in per_uav_rows
        ]
    )
    print("[EKF][Summary] " + " | ".join(f"{k}={v:.3f}" for k, v in summary.items()))

    out = {
        "batch_dir": batch_dir,
        "summary": summary,
        "per_uav": pd.DataFrame(per_uav_rows),
        "predictions": pd.concat(pred_frames, ignore_index=True) if len(pred_frames) > 0 else pd.DataFrame(),
    }

    if save_csv:
        out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        batch_name = os.path.basename(batch_dir.rstrip("\\/"))
        per_uav_path = os.path.join(out_dir, f"{batch_name}_ekf_metrics.csv")
        pred_path = os.path.join(out_dir, f"{batch_name}_ekf_predictions.csv")
        out["per_uav"].to_csv(per_uav_path, index=False)
        out["predictions"].to_csv(pred_path, index=False)
        print(f"[EKF] saved metrics -> {per_uav_path}")
        print(f"[EKF] saved predictions -> {pred_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="EKF multi-source fusion baseline for current processed dataset batch format.")
    p.add_argument("--batch-dir", type=str, default=_default_batch_dir(), help="Path to batch directory with truth.csv and modality csvs.")
    p.add_argument("--max-uavs", type=int, default=1, help="Only run first N UAVs (0 for all).")
    p.add_argument("--align-tolerance-s", type=float, default=0.55, help="Nearest-neighbor alignment tolerance to truth timeline.")
    p.add_argument("--process-accel-sigma", type=float, default=6.0, help="Base process acceleration sigma for CV model.")
    p.add_argument("--use-vel-update", action="store_true", help="Enable velocity component measurement updates (disabled by default because it can degrade accuracy on ultra-precision scenario).")
    p.add_argument("--no-vel-update", action="store_true", help="Alias for keeping velocity updates disabled.")
    p.add_argument("--use-speed-update", action="store_true", help="Enable EKF nonlinear speed updates (may double-count with vx/vy/vz if both enabled).")
    p.add_argument("--no-confidence-scale", action="store_true", help="Disable confidence-based measurement noise scaling.")
    p.add_argument("--no-quality-cols", action="store_true", help="Ignore rt_m/st_m when estimating measurement noise.")
    p.add_argument(
        "--no-acoustic-q-hint",
        "--no-modality-q-hint",
        dest="no_modality_q_hint",
        action="store_true",
        help="Disable multi-modality process-noise hint (acoustic included when present).",
    )
    p.add_argument(
        "--acoustic-q-strength",
        "--modality-q-strength",
        dest="modality_q_strength",
        type=float,
        default=1.0,
        help="Strength of multi-modality process-noise hint (0~1).",
    )
    p.add_argument("--gate-pos-chi2", type=float, default=None, help="Optional Mahalanobis gate for LLH position updates.")
    p.add_argument("--gate-vel-chi2", type=float, default=None, help="Optional Mahalanobis gate for velocity updates.")
    p.add_argument("--gate-speed-chi2", type=float, default=None, help="Optional Mahalanobis gate for speed updates.")
    p.add_argument("--save-csv", action="store_true", help="Save per-UAV metrics and fused trajectory CSV.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for CSVs (used with --save-csv).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EkfConfig(
        align_tolerance_s=float(args.align_tolerance_s),
        process_accel_sigma_mps2=float(args.process_accel_sigma),
        use_velocity_measurement=bool(args.use_vel_update) and (not bool(args.no_vel_update)),
        use_speed_measurement=bool(args.use_speed_update),
        use_confidence_scaling=not bool(args.no_confidence_scale),
        use_quality_columns=not bool(args.no_quality_cols),
        use_modality_q_hint=not bool(args.no_modality_q_hint),
        modality_q_hint_strength=float(args.modality_q_strength),
        gate_pos_chi2=None if args.gate_pos_chi2 is None else float(args.gate_pos_chi2),
        gate_vel_chi2=None if args.gate_vel_chi2 is None else float(args.gate_vel_chi2),
        gate_speed_chi2=None if args.gate_speed_chi2 is None else float(args.gate_speed_chi2),
    )
    evaluate_batch(
        batch_dir=os.path.normpath(args.batch_dir),
        cfg=cfg,
        max_uavs=int(args.max_uavs),
        save_csv=bool(args.save_csv),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
