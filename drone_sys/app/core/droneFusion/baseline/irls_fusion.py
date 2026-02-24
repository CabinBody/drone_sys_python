import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        _align_modality_rows,
        _detect_id_col,
        _load_batch_frames,
        _mean_metrics,
        _multi_modal_q_scale,
        _pos_meas_cov,
        _safe_float,
        _uav_list,
        _valid_position_row,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )
except ImportError:
    from kalman_fusion import (  # type: ignore
        ALL_MODALITIES,
        _align_modality_rows,
        _detect_id_col,
        _load_batch_frames,
        _mean_metrics,
        _multi_modal_q_scale,
        _pos_meas_cov,
        _safe_float,
        _uav_list,
        _valid_position_row,
        calc_err,
        enu_to_llh,
        latlon_to_enu,
    )


@dataclass
class IrlsConfig:
    align_tolerance_s: float = 0.55
    use_confidence_scaling: bool = True
    use_quality_columns: bool = True
    use_modality_q_hint: bool = True
    modality_q_hint_strength: float = 1.0
    irls_max_iter: int = 8
    irls_huber_k: float = 2.5
    irls_tol_m: float = 1e-3
    use_temporal_prior: bool = True
    use_velocity_prediction: bool = True
    temporal_prior_sigma_m: float = 120.0
    temporal_prior_sigma_scale_by_obs: bool = True
    velocity_blend: float = 0.7
    max_step_mps: float = 120.0


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


def _regularize_inv_diag_cov(R: np.ndarray) -> np.ndarray:
    d = np.diag(np.asarray(R, dtype=float)).copy()
    d = np.where(np.isfinite(d) & (d > 1e-9), d, 1e-9)
    return np.diag(1.0 / d)


def _weighted_mean_from_obs(obs_items: List[Dict[str, object]], fallback: Optional[np.ndarray] = None) -> np.ndarray:
    if len(obs_items) == 0:
        if fallback is None:
            return np.zeros((3,), dtype=float)
        return np.asarray(fallback, dtype=float).reshape(3)
    num = np.zeros((3,), dtype=float)
    den = np.zeros((3,), dtype=float)
    for item in obs_items:
        z = np.asarray(item["z"], dtype=float).reshape(3)
        Rinv = np.asarray(item["Rinv"], dtype=float)
        wdiag = np.diag(Rinv)
        num += wdiag * z
        den += wdiag
    den = np.where(den > 1e-12, den, 1e-12)
    return (num / den).astype(float)


def _huber_weight(norm_residual: float, k: float) -> float:
    r = float(abs(norm_residual))
    kk = max(float(k), 1e-6)
    if not np.isfinite(r):
        return 0.0
    if r <= kk:
        return 1.0
    return float(kk / max(r, 1e-9))


def _irls_solve_position(
    obs_items: List[Dict[str, object]],
    x_init: np.ndarray,
    cfg: IrlsConfig,
) -> Tuple[np.ndarray, int, Dict[int, float]]:
    x = np.asarray(x_init, dtype=float).reshape(3).copy()
    max_iter = max(int(cfg.irls_max_iter), 1)
    tol = max(float(cfg.irls_tol_m), 1e-9)
    final_weights: Dict[int, float] = {}

    if len(obs_items) == 0:
        return x, 0, final_weights

    for it in range(max_iter):
        A = np.zeros((3, 3), dtype=float)
        b = np.zeros((3,), dtype=float)
        cur_weights = {}
        for j, item in enumerate(obs_items):
            z = np.asarray(item["z"], dtype=float).reshape(3)
            Rinv = np.asarray(item["Rinv"], dtype=float)
            e = x - z
            d2 = float(e.T @ Rinv @ e)
            d = float(np.sqrt(max(d2, 0.0)))
            wr = _huber_weight(d, float(cfg.irls_huber_k))
            cur_weights[j] = wr
            A += wr * Rinv
            b += wr * (Rinv @ z)

        # diagonal loading for numerical safety
        A += 1e-9 * np.eye(3, dtype=float)
        try:
            x_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_new = np.linalg.pinv(A) @ b
        if float(np.linalg.norm(x_new - x)) <= tol:
            x = x_new.astype(float)
            final_weights = cur_weights
            return x, it + 1, final_weights
        x = x_new.astype(float)
        final_weights = cur_weights

    return x, max_iter, final_weights


def irls_fuse_uav(
    df_truth_u: pd.DataFrame,
    mod_frames: Dict[str, pd.DataFrame],
    cfg: IrlsConfig,
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
        m: {"aligned": 0, "pos_obs": 0, "q_hint_hits": 0, "downweighted_obs": 0}
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

    pred_enu = np.zeros((t_total, 3), dtype=np.float32)
    q_scale_trace = np.ones((t_total,), dtype=np.float32)
    obs_count_trace = np.zeros((t_total,), dtype=np.int32)
    irls_iter_trace = np.zeros((t_total,), dtype=np.int32)
    prior_used_trace = np.zeros((t_total,), dtype=np.int32)

    x_prev: Optional[np.ndarray] = None
    v_prev = np.zeros((3,), dtype=float)

    for k in range(t_total):
        if k == 0:
            dt = 1.0
            if t_total > 1 and np.isfinite(truth_ts[1]) and np.isfinite(truth_ts[0]):
                dt = max(float(truth_ts[1] - truth_ts[0]), 1e-3)
        else:
            dt = max(float(truth_ts[k] - truth_ts[k - 1]), 1e-3)
        dt = max(dt, 1e-3)

        q_scale, q_hit_mods = _multi_modal_q_scale({m: aligned_rows[m][k] for m in ALL_MODALITIES}, cfg)
        q_scale_trace[k] = float(q_scale)
        for m in q_hit_mods:
            modality_usage[m]["q_hint_hits"] += 1

        obs_items: List[Dict[str, object]] = []
        for m in ALL_MODALITIES:
            row = aligned_rows[m][k]
            if row is None or (not _valid_position_row(row)):
                continue

            time_diff = float(aligned_deltas[m][k]) if np.isfinite(aligned_deltas[m][k]) else np.inf
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
            Rinv = _regularize_inv_diag_cov(R)
            obs_items.append({"modality": m, "z": z, "Rinv": Rinv})
            modality_usage[m]["pos_obs"] += 1

        obs_count = len(obs_items)
        obs_count_trace[k] = int(obs_count)

        x_pred = None
        if x_prev is not None:
            if bool(cfg.use_velocity_prediction):
                step = v_prev * dt
                max_step = float(max(cfg.max_step_mps, 1e-3)) * dt
                step_norm = float(np.linalg.norm(step))
                if step_norm > max_step:
                    step = step * (max_step / max(step_norm, 1e-9))
                x_pred = x_prev + step
            else:
                x_pred = x_prev.copy()

        if bool(cfg.use_temporal_prior) and x_pred is not None:
            sigma_prior = float(max(cfg.temporal_prior_sigma_m, 1.0))
            sigma_prior *= float(np.clip(q_scale, 0.7, 1.35))
            if bool(cfg.temporal_prior_sigma_scale_by_obs):
                # More observations -> weaker temporal prior; fewer observations -> stronger prior.
                sigma_prior *= (1.0 + 0.35 * max(0, obs_count - 1))
                if obs_count == 0:
                    sigma_prior *= 0.65
            R_prior_inv = np.diag(np.full((3,), 1.0 / max(sigma_prior**2, 1e-9), dtype=float))
            obs_items.append({"modality": "__prior__", "z": x_pred.astype(float), "Rinv": R_prior_inv})
            prior_used_trace[k] = 1

        if x_pred is not None:
            x_init = x_pred
        else:
            x_init = _weighted_mean_from_obs(obs_items)

        if len(obs_items) == 0:
            x_est = x_prev.copy() if x_prev is not None else np.zeros((3,), dtype=float)
            n_iter = 0
            final_weights = {}
        else:
            x_est, n_iter, final_weights = _irls_solve_position(obs_items=obs_items, x_init=x_init, cfg=cfg)

        irls_iter_trace[k] = int(n_iter)

        # Count strong downweighting of real modality observations (ignore temporal prior pseudo-observation).
        for j, item in enumerate(obs_items):
            m = str(item.get("modality"))
            if m == "__prior__":
                continue
            w = float(final_weights.get(j, 1.0))
            if w < 0.5:
                modality_usage[m]["downweighted_obs"] += 1

        pred_enu[k] = x_est.astype(np.float32)

        if x_prev is not None and dt > 1e-6:
            v_meas = (x_est - x_prev) / dt
            vb = float(np.clip(cfg.velocity_blend, 0.0, 1.0))
            v_prev = vb * v_prev + (1.0 - vb) * v_meas
        x_prev = x_est.astype(float)

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
            "q_scale": q_scale_trace,
            "obs_count": obs_count_trace,
            "irls_iters": irls_iter_trace,
            "prior_used": prior_used_trace,
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
    cfg: IrlsConfig,
    max_uavs: int = 0,
    save_csv: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    truth, mod_frames = _load_batch_frames(batch_dir)
    id_col, uavs = _uav_list(truth)
    if max_uavs and max_uavs > 0:
        uavs = uavs[: int(max_uavs)]

    print(
        f"[IRLS] batch={os.path.basename(batch_dir)} | uavs={len(uavs)} | "
        f"align_tol={cfg.align_tolerance_s:.2f}s | temporal_prior={cfg.use_temporal_prior} | "
        f"vel_pred={cfg.use_velocity_prediction} | modality_q_hint={cfg.use_modality_q_hint}"
    )
    per_uav_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for idx, uav in enumerate(uavs, start=1):
        df_t = truth[truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
        res = irls_fuse_uav(df_t, mod_frames, cfg)
        metrics = calc_err(res["pred_enu"], res["truth_enu"])
        usage = res["modality_usage"]
        q_hint_total = int(sum(int(usage[m]["q_hint_hits"]) for m in ALL_MODALITIES))
        usage_summary = ", ".join(
            f"{m}:a{usage[m]['aligned']}/p{usage[m]['pos_obs']}/q{usage[m]['q_hint_hits']}/dw{usage[m]['downweighted_obs']}"
            for m in ALL_MODALITIES
        )
        print(
            f"[{idx:03d}/{len(uavs):03d}] {uav} | "
            f"RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} P95={metrics['P95']:.3f} MAX={metrics['MAX']:.3f} | "
            f"q_hint_total={q_hint_total} | {usage_summary}"
        )
        row = {"uav_id": uav, **metrics, "q_hint_hits_total": q_hint_total}
        per_uav_rows.append(row)
        pred_df = res["pred_df"]
        pred_df.insert(0, "batch", os.path.basename(batch_dir))
        pred_frames.append(pred_df)

    summary = _mean_metrics(
        [{"RMSE": r["RMSE"], "MAE": r["MAE"], "MEDAE": r["MEDAE"], "P95": r["P95"], "MAX": r["MAX"]} for r in per_uav_rows]
    )
    print("[IRLS][Summary] " + " | ".join(f"{k}={v:.3f}" for k, v in summary.items()))

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
        per_uav_path = os.path.join(out_dir, f"{batch_name}_irls_metrics.csv")
        pred_path = os.path.join(out_dir, f"{batch_name}_irls_predictions.csv")
        out["per_uav"].to_csv(per_uav_path, index=False)
        out["predictions"].to_csv(pred_path, index=False)
        print(f"[IRLS] saved metrics -> {per_uav_path}")
        print(f"[IRLS] saved predictions -> {pred_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="IRLS (Iteratively Reweighted Least Squares) multi-source fusion baseline for current processed dataset batch format.")
    p.add_argument("--batch-dir", type=str, default=_default_batch_dir(), help="Path to batch directory with truth.csv and modality csvs.")
    p.add_argument("--max-uavs", type=int, default=1, help="Only run first N UAVs (0 for all).")
    p.add_argument("--align-tolerance-s", type=float, default=0.55, help="Nearest-neighbor alignment tolerance to truth timeline.")
    p.add_argument("--no-confidence-scale", action="store_true", help="Disable confidence-based measurement noise scaling.")
    p.add_argument("--no-quality-cols", action="store_true", help="Ignore rt_m/st_m when estimating measurement noise.")
    p.add_argument(
        "--no-acoustic-q-hint",
        "--no-modality-q-hint",
        dest="no_modality_q_hint",
        action="store_true",
        help="Disable multi-modality process/smoothing hint (acoustic included when present).",
    )
    p.add_argument(
        "--acoustic-q-strength",
        "--modality-q-strength",
        dest="modality_q_strength",
        type=float,
        default=1.0,
        help="Strength of multi-modality hint (0~1).",
    )
    p.add_argument("--irls-max-iter", type=int, default=8, help="Maximum IRLS iterations per timestamp.")
    p.add_argument("--irls-huber-k", type=float, default=2.5, help="Huber threshold in whitened residual norm units.")
    p.add_argument("--irls-tol-m", type=float, default=1e-3, help="IRLS convergence tolerance in meters.")
    p.add_argument("--no-temporal-prior", action="store_true", help="Disable temporal prior pseudo-observation.")
    p.add_argument("--no-vel-pred", action="store_true", help="Disable velocity-based prediction for temporal prior.")
    p.add_argument("--temporal-prior-sigma", type=float, default=120.0, help="Base temporal prior sigma in meters.")
    p.add_argument("--velocity-blend", type=float, default=0.7, help="Blend factor for internal velocity estimate (0~1).")
    p.add_argument("--max-step-mps", type=float, default=120.0, help="Clamp predicted step speed for stability.")
    p.add_argument("--save-csv", action="store_true", help="Save per-UAV metrics and fused trajectory CSV.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for CSVs (used with --save-csv).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = IrlsConfig(
        align_tolerance_s=float(args.align_tolerance_s),
        use_confidence_scaling=not bool(args.no_confidence_scale),
        use_quality_columns=not bool(args.no_quality_cols),
        use_modality_q_hint=not bool(args.no_modality_q_hint),
        modality_q_hint_strength=float(args.modality_q_strength),
        irls_max_iter=int(args.irls_max_iter),
        irls_huber_k=float(args.irls_huber_k),
        irls_tol_m=float(args.irls_tol_m),
        use_temporal_prior=not bool(args.no_temporal_prior),
        use_velocity_prediction=not bool(args.no_vel_pred),
        temporal_prior_sigma_m=float(args.temporal_prior_sigma),
        velocity_blend=float(args.velocity_blend),
        max_step_mps=float(args.max_step_mps),
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
