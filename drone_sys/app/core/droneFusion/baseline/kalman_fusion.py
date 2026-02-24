import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


R_EARTH = 6378137.0
POS_MODALITIES = ("gps", "radar", "5g_a", "tdoa")
ALL_MODALITIES = ("gps", "radar", "5g_a", "tdoa", "acoustic")

BASE_POS_SIGMA_M = {
    "gps": 8.0,
    "radar": 35.0,
    "5g_a": 55.0,
    "tdoa": 70.0,
}
BASE_VEL_SIGMA_MPS = {
    "gps": 2.0,
    "radar": 5.0,
    "5g_a": 8.0,
    "tdoa": 10.0,
}
GATE_POS_CHI2 = None
GATE_VEL_CHI2 = None


@dataclass
class KalmanConfig:
    align_tolerance_s: float = 0.55
    process_accel_sigma_mps2: float = 6.0
    init_pos_sigma_m: float = 50.0
    init_vel_sigma_mps: float = 30.0
    use_velocity_measurement: bool = True
    use_confidence_scaling: bool = True
    use_quality_columns: bool = True
    use_modality_q_hint: bool = True
    modality_q_hint_strength: float = 1.0


def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    dlat = np.radians(lat - float(lat0))
    dlon = np.radians(lon - float(lon0))
    east = dlon * np.cos(np.radians(float(lat0))) * R_EARTH
    north = dlat * R_EARTH
    up = alt - float(alt0)
    return east, north, up


def enu_to_llh(east, north, up, lat0, lon0, alt0):
    lat0_r = np.radians(float(lat0))
    lon0_r = np.radians(float(lon0))
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    up = np.asarray(up, dtype=float)
    lat = north / R_EARTH + lat0_r
    lon = east / (R_EARTH * np.cos(lat0_r)) + lon0_r
    alt = up + float(alt0)
    return np.degrees(lat), np.degrees(lon), alt


def _detect_id_col(df: pd.DataFrame) -> Optional[str]:
    if "uav_id" in df.columns:
        return "uav_id"
    if "id" in df.columns:
        return "id"
    return None


def _safe_float(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _nearest_truth_index(obs_ts: np.ndarray, truth_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_hi = np.searchsorted(truth_ts, obs_ts, side="left")
    idx_lo = np.clip(idx_hi - 1, 0, len(truth_ts) - 1)
    idx_hi = np.clip(idx_hi, 0, len(truth_ts) - 1)
    d_lo = np.abs(obs_ts - truth_ts[idx_lo])
    d_hi = np.abs(obs_ts - truth_ts[idx_hi])
    choose_hi = d_hi < d_lo
    best_idx = np.where(choose_hi, idx_hi, idx_lo)
    best_diff = np.where(choose_hi, d_hi, d_lo)
    return best_idx.astype(np.int64), best_diff.astype(float)


def calc_err(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if pred is None or gt is None or len(pred) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MEDAE": np.nan, "P95": np.nan, "MAX": np.nan}
    diff = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    dist = np.linalg.norm(diff, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(dist**2))),
        "MAE": float(np.mean(dist)),
        "MEDAE": float(np.median(dist)),
        "P95": float(np.percentile(dist, 95)),
        "MAX": float(np.max(dist)),
    }


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


def _kalman_update(
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


def _valid_position_row(row: pd.Series) -> bool:
    if row is None:
        return False
    if int(_safe_float(row.get("missing_flag", 0), 0)) > 0:
        return False
    lat = _safe_float(row.get("lat", np.nan), np.nan)
    lon = _safe_float(row.get("lon", np.nan), np.nan)
    alt = _safe_float(row.get("alt", np.nan), np.nan)
    return np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)


def _valid_velocity_row(row: pd.Series) -> bool:
    if row is None:
        return False
    if int(_safe_float(row.get("missing_flag", 0), 0)) > 0:
        return False
    vx = _safe_float(row.get("vx", np.nan), np.nan)
    vy = _safe_float(row.get("vy", np.nan), np.nan)
    vz = _safe_float(row.get("vz", np.nan), np.nan)
    return np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)


def _confidence_scale(row: pd.Series, cfg: KalmanConfig) -> float:
    if not cfg.use_confidence_scaling:
        return 1.0
    conf = _safe_float(row.get("confidence", 0.5), 0.5)
    conf = float(np.clip(conf, 0.0, 1.0))
    # conf=1 -> smaller noise, conf=0 -> larger noise
    return float(2.2 - 1.7 * conf)


def _row_quality_sigma_hint(row: pd.Series) -> Optional[float]:
    vals = []
    for c in ("rt_m", "st_m"):
        v = _safe_float(row.get(c, np.nan), np.nan)
        if np.isfinite(v) and v > 0:
            vals.append(float(v))
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _pos_meas_cov(modality: str, row: pd.Series, time_diff_s: float, cfg: KalmanConfig) -> np.ndarray:
    sigma = float(BASE_POS_SIGMA_M.get(modality, 80.0))
    sigma *= _confidence_scale(row, cfg)
    if cfg.use_quality_columns:
        hint = _row_quality_sigma_hint(row)
        if hint is not None and np.isfinite(hint):
            sigma = max(sigma, 0.6 * sigma + 0.4 * hint)
    sigma *= 1.0 + min(float(abs(time_diff_s)) / max(cfg.align_tolerance_s, 1e-3), 1.0)
    sigma = max(sigma, 1.0)
    return np.diag([sigma**2, sigma**2, (1.2 * sigma) ** 2]).astype(float)


def _vel_meas_cov(modality: str, row: pd.Series, time_diff_s: float, cfg: KalmanConfig) -> np.ndarray:
    sigma = float(BASE_VEL_SIGMA_MPS.get(modality, 12.0))
    sigma *= _confidence_scale(row, cfg)
    sigma *= 1.0 + 0.5 * min(float(abs(time_diff_s)) / max(cfg.align_tolerance_s, 1e-3), 1.0)
    sigma = max(sigma, 0.3)
    return np.diag([sigma**2, sigma**2, sigma**2]).astype(float)


def _cfg_use_modality_q_hint(cfg) -> bool:
    return bool(getattr(cfg, "use_modality_q_hint", getattr(cfg, "use_acoustic_q_hint", False)))


def _cfg_modality_q_hint_strength(cfg) -> float:
    return float(getattr(cfg, "modality_q_hint_strength", getattr(cfg, "acoustic_q_hint_strength", 0.0)))


def _row_process_q_scale(modality: str, row: Optional[pd.Series], cfg) -> Tuple[float, bool]:
    if (not _cfg_use_modality_q_hint(cfg)) or row is None:
        return 1.0, False

    strength = float(np.clip(_cfg_modality_q_hint_strength(cfg), 0.0, 1.0))
    if strength <= 0.0:
        return 1.0, False

    missing_flag = int(_safe_float(row.get("missing_flag", 0), 0))
    conf = _safe_float(row.get("confidence", np.nan), np.nan)
    conf_valid = bool(np.isfinite(conf))
    conf_val = float(np.clip(conf, 0.0, 1.0)) if conf_valid else 0.0

    has_pos = _valid_position_row(row)
    has_vel = _valid_velocity_row(row)
    speed_raw = _safe_float(row.get("speed", np.nan), np.nan)
    has_speed = (missing_flag <= 0) and bool(np.isfinite(speed_raw))
    detected = float(np.clip(_safe_float(row.get("detected_flag", 0.0), 0.0), 0.0, 1.0))
    energy = _safe_float(row.get("acoustic_energy", np.nan), np.nan)
    energy_val = float(np.clip(energy, 0.0, 1.0)) if np.isfinite(energy) else 0.0

    info_score = 0.0
    info_flag = False
    if conf_valid:
        info_score += 0.45 * conf_val
        info_flag = True
    if has_pos:
        info_score += 0.30
        info_flag = True
    if has_vel:
        info_score += 0.20
        info_flag = True
    elif has_speed:
        info_score += 0.10
        info_flag = True
    if modality == "acoustic":
        if detected > 0.0:
            info_score += 0.18 * detected
            info_flag = True
        if np.isfinite(energy):
            info_score += 0.12 * energy_val
            info_flag = True

    if missing_flag > 0:
        # Explicit misses still carry information: increase uncertainty mildly.
        miss_boost = 1.0 + (0.12 if modality == "acoustic" else 0.08) * strength
        return float(miss_boost), True

    if not info_flag:
        return 1.0, False

    info_score = float(np.clip(info_score, 0.0, 1.0))
    scale = 1.12 - 0.32 * strength * info_score
    return float(np.clip(scale, 0.72, 1.25)), True


def _multi_modal_q_scale(rows_by_modality: Dict[str, Optional[pd.Series]], cfg) -> Tuple[float, List[str]]:
    if not _cfg_use_modality_q_hint(cfg):
        return 1.0, []

    mod_w = {"gps": 1.0, "radar": 0.85, "5g_a": 0.75, "tdoa": 0.7, "acoustic": 0.45}
    w_sum = 0.0
    accum = 0.0
    hit_modalities: List[str] = []
    for m in ALL_MODALITIES:
        row = rows_by_modality.get(m)
        scale, hit = _row_process_q_scale(m, row, cfg)
        if not hit:
            continue
        w = float(mod_w.get(m, 0.5))
        accum += w * float(scale)
        w_sum += w
        hit_modalities.append(m)

    if w_sum <= 0.0:
        return 1.0, []
    q_scale = float(accum / w_sum)
    return float(np.clip(q_scale, 0.70, 1.35)), hit_modalities


def _align_modality_rows(
    df_truth_u: pd.DataFrame,
    df_mod_u: pd.DataFrame,
    align_tolerance_s: float,
) -> Tuple[List[Optional[pd.Series]], np.ndarray]:
    t_total = len(df_truth_u)
    rows: List[Optional[pd.Series]] = [None] * t_total
    deltas = np.full((t_total,), np.inf, dtype=float)
    if t_total == 0 or df_mod_u is None or len(df_mod_u) == 0 or "timestamp" not in df_mod_u.columns:
        return rows, deltas

    truth_ts = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)
    mod = df_mod_u.copy()
    mod["timestamp"] = pd.to_numeric(mod["timestamp"], errors="coerce")
    mod = mod[np.isfinite(mod["timestamp"].to_numpy(dtype=float))].sort_values("timestamp").reset_index(drop=True)
    if len(mod) == 0:
        return rows, deltas

    obs_ts = mod["timestamp"].to_numpy(dtype=float)
    t_idx, t_diff = _nearest_truth_index(obs_ts, truth_ts)
    best_for_t: Dict[int, Tuple[float, int]] = {}
    for obs_i, (ti, diff_s) in enumerate(zip(t_idx, t_diff)):
        if float(diff_s) > float(align_tolerance_s):
            continue
        prev = best_for_t.get(int(ti))
        if prev is None or float(diff_s) < prev[0]:
            best_for_t[int(ti)] = (float(diff_s), int(obs_i))

    for ti, (diff_s, obs_i) in best_for_t.items():
        rows[ti] = mod.iloc[int(obs_i)]
        deltas[ti] = float(diff_s)
    return rows, deltas


def _load_batch_frames(batch_dir: str, modalities: Tuple[str, ...] = ALL_MODALITIES):
    truth_path = os.path.join(batch_dir, "truth.csv")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.csv not found: {truth_path}")
    truth = pd.read_csv(truth_path)
    mod_frames: Dict[str, pd.DataFrame] = {}
    for m in modalities:
        fname = "5g_a.csv" if m == "5g_a" else f"{m}.csv"
        p = os.path.join(batch_dir, fname)
        mod_frames[m] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return truth, mod_frames


def _uav_list(df_truth: pd.DataFrame) -> Tuple[str, List[str]]:
    id_col = _detect_id_col(df_truth)
    if id_col is None:
        raise RuntimeError("truth.csv must contain `uav_id` or `id`")
    uavs = list(pd.Series(df_truth[id_col]).dropna().unique())
    if len(uavs) == 0:
        raise RuntimeError("no UAV found in truth.csv")
    return id_col, uavs


def _meas_weighted_state_init(
    aligned_rows: Dict[str, List[Optional[pd.Series]]],
    aligned_deltas: Dict[str, np.ndarray],
    lat0: float,
    lon0: float,
    alt0: float,
    cfg: KalmanConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros((6,), dtype=float)
    P0 = np.diag(
        [
            cfg.init_pos_sigma_m**2,
            cfg.init_pos_sigma_m**2,
            (1.5 * cfg.init_pos_sigma_m) ** 2,
            cfg.init_vel_sigma_mps**2,
            cfg.init_vel_sigma_mps**2,
            cfg.init_vel_sigma_mps**2,
        ]
    ).astype(float)

    pos_accum = []
    pos_w = []
    vel_accum = []
    vel_w = []
    for m in ALL_MODALITIES:
        rows = aligned_rows.get(m)
        deltas = aligned_deltas.get(m)
        if rows is None or deltas is None or len(rows) == 0:
            continue
        row = rows[0]
        if row is None:
            continue
        dt_diff = float(deltas[0]) if len(deltas) > 0 else 0.0
        if _valid_position_row(row):
            e, n, u = latlon_to_enu(
                _safe_float(row.get("lat"), np.nan),
                _safe_float(row.get("lon"), np.nan),
                _safe_float(row.get("alt"), np.nan),
                lat0,
                lon0,
                alt0,
            )
            base_sigma = np.sqrt(np.diag(_pos_meas_cov(m, row, dt_diff, cfg))[0])
            w = 1.0 / max(float(base_sigma) ** 2, 1e-6)
            pos_accum.append(np.array([float(e), float(n), float(u)], dtype=float))
            pos_w.append(w)
        if cfg.use_velocity_measurement and _valid_velocity_row(row):
            vv = np.array(
                [
                    _safe_float(row.get("vx"), 0.0),
                    _safe_float(row.get("vy"), 0.0),
                    _safe_float(row.get("vz"), 0.0),
                ],
                dtype=float,
            )
            base_sigma_v = np.sqrt(np.diag(_vel_meas_cov(m, row, dt_diff, cfg))[0])
            wv = 1.0 / max(float(base_sigma_v) ** 2, 1e-6)
            vel_accum.append(vv)
            vel_w.append(wv)

    if len(pos_accum) > 0:
        W = np.asarray(pos_w, dtype=float)
        x0[0:3] = (np.asarray(pos_accum, dtype=float) * W[:, None]).sum(axis=0) / np.sum(W)
    if len(vel_accum) > 0:
        W = np.asarray(vel_w, dtype=float)
        x0[3:6] = (np.asarray(vel_accum, dtype=float) * W[:, None]).sum(axis=0) / np.sum(W)
    return x0, P0


def kalman_fuse_uav(
    df_truth_u: pd.DataFrame,
    mod_frames: Dict[str, pd.DataFrame],
    cfg: KalmanConfig,
) -> Dict[str, object]:
    if len(df_truth_u) == 0:
        return {"pred_enu": np.zeros((0, 3), dtype=np.float32), "truth_enu": np.zeros((0, 3), dtype=np.float32)}

    df_truth_u = df_truth_u.sort_values("timestamp").reset_index(drop=True)
    lat0, lon0, alt0 = df_truth_u.iloc[0][["lat", "lon", "alt"]]
    t_total = len(df_truth_u)
    truth_ts = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)

    e_gt, n_gt, u_gt = latlon_to_enu(df_truth_u["lat"].values, df_truth_u["lon"].values, df_truth_u["alt"].values, lat0, lon0, alt0)
    truth_enu = np.stack([e_gt, n_gt, u_gt], axis=-1).astype(np.float32)

    id_col_truth = _detect_id_col(df_truth_u)
    uav_value = df_truth_u.iloc[0][id_col_truth] if id_col_truth else None

    aligned_rows: Dict[str, List[Optional[pd.Series]]] = {}
    aligned_deltas: Dict[str, np.ndarray] = {}
    modality_usage = {
        m: {"pos_updates": 0, "vel_updates": 0, "gated": 0, "aligned": 0, "q_hint_hits": 0}
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

    H_pos = np.zeros((3, 6), dtype=float)
    H_pos[:, 0:3] = np.eye(3, dtype=float)
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
        q_scale_trace[k] = float(q_scale)
        for m in q_hit_mods:
            modality_usage[m]["q_hint_hits"] += 1

        F, Q = _cv_F_Q(dt, float(cfg.process_accel_sigma_mps2) * np.sqrt(max(float(q_scale), 1e-6)))
        x = F @ x
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        for m in ALL_MODALITIES:
            row = aligned_rows[m][k]
            if row is None:
                continue
            time_diff = float(aligned_deltas[m][k]) if np.isfinite(aligned_deltas[m][k]) else np.inf

            if _valid_position_row(row):
                e, n, u = latlon_to_enu(
                    _safe_float(row.get("lat"), np.nan),
                    _safe_float(row.get("lon"), np.nan),
                    _safe_float(row.get("alt"), np.nan),
                    lat0,
                    lon0,
                    alt0,
                )
                z = np.array([float(e), float(n), float(u)], dtype=float)
                R = _pos_meas_cov(m, row, time_diff, cfg)
                x_new, P_new, accepted = _kalman_update(x, P, z, H_pos, R, gate_chi2=GATE_POS_CHI2)
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
                x_new, P_new, accepted = _kalman_update(x, P, z_v, H_vel, R_v, gate_chi2=GATE_VEL_CHI2)
                if accepted:
                    x, P = x_new, P_new
                    modality_usage[m]["vel_updates"] += 1
                else:
                    modality_usage[m]["gated"] += 1

        pred_enu[k] = x[0:3].astype(np.float32)
        pred_cov_trace[k] = float(np.trace(P[0:3, 0:3]))

    pred_lat, pred_lon, pred_alt = enu_to_llh(pred_enu[:, 0], pred_enu[:, 1], pred_enu[:, 2], lat0, lon0, alt0)
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


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if len(rows) == 0:
        return {}
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        vals = np.array([_safe_float(r.get(k, np.nan), np.nan) for r in rows], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    return out


def evaluate_batch(
    batch_dir: str,
    cfg: KalmanConfig,
    max_uavs: int = 0,
    save_csv: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    truth, mod_frames = _load_batch_frames(batch_dir)
    id_col, uavs = _uav_list(truth)
    if max_uavs and max_uavs > 0:
        uavs = uavs[: int(max_uavs)]

    print(f"[Kalman] batch={os.path.basename(batch_dir)} | uavs={len(uavs)} | align_tol={cfg.align_tolerance_s:.2f}s")
    per_uav_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for idx, uav in enumerate(uavs, start=1):
        df_t = truth[truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
        res = kalman_fuse_uav(df_t, mod_frames, cfg)
        metrics = calc_err(res["pred_enu"], res["truth_enu"])
        usage = res["modality_usage"]
        usage_summary = ", ".join(
            f"{m}:a{usage[m]['aligned']}/p{usage[m]['pos_updates']}/v{usage[m]['vel_updates']}/q{usage[m]['q_hint_hits']}/g{usage[m]['gated']}"
            for m in ALL_MODALITIES
        )
        print(
            f"[{idx:03d}/{len(uavs):03d}] {uav} | "
            f"RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} P95={metrics['P95']:.3f} MAX={metrics['MAX']:.3f} | "
            f"{usage_summary}"
        )
        row = {"uav_id": uav, **metrics}
        per_uav_rows.append(row)
        pred_df = res["pred_df"]
        pred_df.insert(0, "batch", os.path.basename(batch_dir))
        pred_frames.append(pred_df)

    summary = _mean_metrics([{"RMSE": r["RMSE"], "MAE": r["MAE"], "MEDAE": r["MEDAE"], "P95": r["P95"], "MAX": r["MAX"]} for r in per_uav_rows])
    print(
        "[Kalman][Summary] "
        + " | ".join(f"{k}={v:.3f}" for k, v in summary.items())
    )

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
        per_uav_path = os.path.join(out_dir, f"{batch_name}_kalman_metrics.csv")
        pred_path = os.path.join(out_dir, f"{batch_name}_kalman_predictions.csv")
        out["per_uav"].to_csv(per_uav_path, index=False)
        out["predictions"].to_csv(pred_path, index=False)
        print(f"[Kalman] saved metrics -> {per_uav_path}")
        print(f"[Kalman] saved predictions -> {pred_path}")
    return out


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


def parse_args():
    p = argparse.ArgumentParser(description="Traditional constant-velocity Kalman fusion baseline for current dataset format.")
    p.add_argument("--batch-dir", type=str, default=_default_batch_dir(), help="Path to one processed batch directory containing truth.csv and modality csv files.")
    p.add_argument("--max-uavs", type=int, default=1, help="Only run first N UAVs (0 for all).")
    p.add_argument("--align-tolerance-s", type=float, default=0.55, help="Nearest-neighbor alignment tolerance to truth timeline.")
    p.add_argument("--process-accel-sigma", type=float, default=6.0, help="Process acceleration sigma for constant-velocity model.")
    p.add_argument("--no-vel-update", action="store_true", help="Disable velocity measurement updates.")
    p.add_argument("--no-confidence-scale", action="store_true", help="Disable confidence-based measurement noise scaling.")
    p.add_argument("--no-quality-cols", action="store_true", help="Ignore rt_m/st_m when estimating measurement noise.")
    p.add_argument("--no-modality-q-hint", action="store_true", help="Disable multi-modality process-noise hint (uses all five modalities when available).")
    p.add_argument("--modality-q-strength", type=float, default=1.0, help="Strength of multi-modality process-noise hint (0~1).")
    p.add_argument("--save-csv", action="store_true", help="Save per-UAV metrics and fused trajectory CSV.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for CSVs (used with --save-csv).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = KalmanConfig(
        align_tolerance_s=float(args.align_tolerance_s),
        process_accel_sigma_mps2=float(args.process_accel_sigma),
        use_velocity_measurement=not bool(args.no_vel_update),
        use_confidence_scaling=not bool(args.no_confidence_scale),
        use_quality_columns=not bool(args.no_quality_cols),
        use_modality_q_hint=not bool(args.no_modality_q_hint),
        modality_q_hint_strength=float(args.modality_q_strength),
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
