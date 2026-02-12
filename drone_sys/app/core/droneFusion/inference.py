import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import (
    MODALITIES,
    MODALITY_TO_ID,
    NODE_FEAT_DIM,
    NORM_STATS_PATH,
    _build_node_feature,
    _nearest_truth_index,
    enu_to_llh,
    latlon_to_enu,
)
from model import GraphFusionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================================================
# CONFIG
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"../datasetBuilder/dataset-processed/scenario_eval_high_missing_mixed_100x60/"
BATCH_DIR = os.path.join(DATA_ROOT, "batch01")
MODEL_PATH = os.path.join(BASE_DIR, "graph_fusion_model_processed.pt")
NORM_PATH = os.path.join(BASE_DIR, NORM_STATS_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UID = 10

WINDOW_SIZE = 20
STRIDE = 1
ALIGN_TOLERANCE_S = 0.55
MERGE_EDGE_TAPER_MIN = 0.25
WARMUP_POINTS = 20
WARMUP_MIN_COVERAGE = 3.0
TAIL_POINTS = 20


# ===========================================================
# HELPERS
# ===========================================================
def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_int(v, default):
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _to_float(v, default):
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _as_1d_array(x, dtype=None):
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _detect_id_col(df: pd.DataFrame) -> Optional[str]:
    if "uav_id" in df.columns:
        return "uav_id"
    if "id" in df.columns:
        return "id"
    return None


def _extract_state_dict(payload: dict) -> Tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is not a dict")

    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        state_dict = payload["model_state_dict"]
        meta = payload
    elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
        meta = payload
    elif len(payload) > 0 and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state_dict = payload
        meta = {}
    else:
        raise RuntimeError("checkpoint has no model_state_dict/state_dict")

    if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict, meta


def _split_cfg(raw_cfg: dict) -> Tuple[dict, dict, dict]:
    if not isinstance(raw_cfg, dict):
        return {}, {}, {}
    if isinstance(raw_cfg.get("model"), dict) or isinstance(raw_cfg.get("data"), dict):
        return raw_cfg, raw_cfg.get("model", {}), raw_cfg.get("data", {})
    return raw_cfg, raw_cfg, {}


def _infer_num_layers(state_dict: dict, fallback: int = 3) -> int:
    layer_ids = set()
    for k in state_dict.keys():
        parts = k.split(".")
        if len(parts) > 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_ids.add(int(parts[1]))
    if len(layer_ids) == 0:
        return int(fallback)
    return int(max(layer_ids) + 1)


def _load_norm_stats(meta: dict, norm_path: str, device: str):
    keys = ("x_mean", "x_std", "y_mean", "y_std")
    if all(k in meta for k in keys):
        return tuple(torch.as_tensor(meta[k]).to(device).float() for k in keys)

    if not os.path.exists(norm_path):
        raise RuntimeError(
            f"checkpoint has no norm stats and norm file is missing: {norm_path}"
        )

    stats = _safe_torch_load(norm_path)
    if not isinstance(stats, dict) or not all(k in stats for k in keys):
        raise RuntimeError(f"invalid norm stats file: {norm_path}")
    return tuple(torch.as_tensor(stats[k]).to(device).float() for k in keys)


def load_model_and_runtime(model_path: str, norm_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")

    payload = _safe_torch_load(model_path)
    state_dict, meta = _extract_state_dict(payload)
    raw_cfg, model_cfg, data_cfg = _split_cfg(meta.get("config", {}))
    x_mean, x_std, y_mean, y_std = _load_norm_stats(meta, norm_path, device)

    node_encoder_w = state_dict.get("node_encoder.weight")
    in_dim = _to_int(
        raw_cfg.get("in_dim", model_cfg.get("in_dim", None)),
        node_encoder_w.shape[1] if node_encoder_w is not None else int(x_mean.numel()),
    )
    d_model = _to_int(
        model_cfg.get("d_model", raw_cfg.get("d_model", None)),
        node_encoder_w.shape[0] if node_encoder_w is not None else 128,
    )
    num_heads = _to_int(
        model_cfg.get("num_heads", raw_cfg.get("num_heads", None)),
        state_dict.get("edge_mlp.2.bias", torch.zeros(4)).numel(),
    )
    num_layers = _to_int(
        model_cfg.get("num_layers", raw_cfg.get("num_layers", None)),
        _infer_num_layers(state_dict, fallback=3),
    )
    dim_ff = _to_int(
        model_cfg.get("dim_ff", raw_cfg.get("dim_ff", None)),
        state_dict.get("layers.0.ffn.0.bias", torch.zeros(256)).numel(),
    )
    dropout = _to_float(model_cfg.get("dropout", raw_cfg.get("dropout", 0.1)), 0.1)
    window_size = _to_int(raw_cfg.get("window_size", data_cfg.get("window_size", WINDOW_SIZE)), WINDOW_SIZE)

    modalities_cfg = data_cfg.get("modalities", MODALITIES)
    if isinstance(modalities_cfg, (list, tuple)):
        modalities = [m for m in modalities_cfg if m in MODALITY_TO_ID]
    else:
        modalities = list(MODALITIES)
    if len(modalities) == 0:
        modalities = list(MODALITIES)

    num_modalities = _to_int(raw_cfg.get("num_modalities", len(modalities)), len(modalities))
    knn_k = _to_int(model_cfg.get("knn_k", raw_cfg.get("knn_k", 8)), 8)
    stride = _to_int(data_cfg.get("stride", STRIDE), STRIDE)
    align_tolerance_s = _to_float(data_cfg.get("align_tolerance_s", ALIGN_TOLERANCE_S), ALIGN_TOLERANCE_S)

    model = GraphFusionModel(
        in_dim=in_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        window_size=window_size,
        num_modalities=num_modalities,
        knn_k=knn_k,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    runtime = {
        "in_dim": in_dim,
        "window_size": window_size,
        "stride": stride,
        "align_tolerance_s": align_tolerance_s,
        "modalities": modalities,
    }
    return model, x_mean, x_std, y_mean, y_std, runtime


# ===========================================================
# ERROR
# ===========================================================
def calc_err(pred, gt):
    if len(pred) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAX": np.nan}
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(dist ** 2))),
        "MAE": float(np.mean(np.abs(dist))),
        "MAX": float(np.max(dist)),
    }


def estimate_window_quality(node_feat: np.ndarray):
    if node_feat is None or node_feat.size == 0:
        return 1e-3

    n_nodes = float(node_feat.shape[0])
    conf_mean = float(np.clip(np.nanmean(node_feat[:, 7]) if node_feat.shape[1] > 7 else 0.5, 0.0, 1.0))
    pos_valid_ratio = float(np.clip(np.nanmean(node_feat[:, 9]) if node_feat.shape[1] > 9 else 1.0, 0.0, 1.0))
    obs_valid_ratio = float(np.clip(np.nanmean(node_feat[:, 10]) if node_feat.shape[1] > 10 else 1.0, 0.0, 1.0))

    score = np.log1p(n_nodes)
    score *= (0.3 + 0.7 * conf_mean)
    score *= (0.5 + 0.5 * pos_valid_ratio)
    score *= (0.5 + 0.5 * obs_valid_ratio)
    return float(max(score, 1e-3))


def _edge_taper_weights(valid_len: int, min_edge_weight: float):
    if valid_len <= 1:
        return np.ones((max(valid_len, 1),), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, valid_len, dtype=np.float32)
    tri = 1.0 - np.abs(x)
    return np.asarray(min_edge_weight + (1.0 - min_edge_weight) * tri, dtype=np.float32)


def merge_windows(preds, starts, t_total, window, window_weights=None, edge_taper_min=0.35):
    fusion_sum = np.zeros((t_total, 3), dtype=np.float32)
    weight_sum = np.zeros(t_total, dtype=np.float32)
    cover_count = np.zeros(t_total, dtype=np.float32)

    if window_weights is None:
        window_weights = [1.0] * len(starts)

    for p, s, w in zip(preds, starts, window_weights):
        valid = min(window, t_total - s)
        if valid <= 0:
            continue
        local_w = _edge_taper_weights(valid, min_edge_weight=float(edge_taper_min)) * float(max(w, 1e-6))
        fusion_sum[s : s + valid] += p[:valid] * local_w[:, None]
        weight_sum[s : s + valid] += local_w
        cover_count[s : s + valid] += 1.0

    fusion = fusion_sum / (weight_sum[:, None] + 1e-6)
    return fusion.astype(np.float32), weight_sum.astype(np.float32), cover_count.astype(np.float32)


def build_obs_fallback_series(
    df_truth_u: pd.DataFrame,
    mod_frames: Dict[str, pd.DataFrame],
    modalities: List[str],
    lat0: float,
    lon0: float,
    alt0: float,
    align_tolerance_s: float,
):
    t_total = len(df_truth_u)
    if t_total == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    ts_truth = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)
    id_col_t = _detect_id_col(df_truth_u)
    uav_value = df_truth_u.iloc[0][id_col_t] if id_col_t is not None else None

    # Conservative priors to avoid noisy modalities dominating warmup.
    mod_prior = {"gps": 1.0, "radar": 0.7, "5g_a": 0.6, "tdoa": 0.45, "acoustic": 0.2}

    sum_pos = np.zeros((t_total, 3), dtype=np.float64)
    sum_w = np.zeros((t_total,), dtype=np.float64)

    for m in modalities:
        df_m = mod_frames.get(m, pd.DataFrame())
        if len(df_m) == 0 or "timestamp" not in df_m.columns:
            continue
        if not all(c in df_m.columns for c in ["lat", "lon", "alt"]):
            continue

        id_col_m = _detect_id_col(df_m)
        if id_col_m is not None and uav_value is not None:
            df_m = df_m[df_m[id_col_m] == uav_value]
        if len(df_m) == 0:
            continue

        ts_obs = _as_1d_array(pd.to_numeric(df_m["timestamp"], errors="coerce").to_numpy(dtype=float), dtype=float)
        valid_ts = np.isfinite(ts_obs).reshape(-1)
        valid_idx = np.flatnonzero(valid_ts)
        if valid_idx.size == 0:
            continue
        df_m = df_m.iloc[valid_idx].reset_index(drop=True)
        ts_obs = ts_obs[valid_idx]

        tidx, tdiff = _nearest_truth_index(ts_obs, ts_truth)
        tidx = _as_1d_array(tidx, dtype=np.int64)
        tdiff = _as_1d_array(tdiff, dtype=float)
        keep_idx = np.flatnonzero(tdiff <= float(align_tolerance_s))
        if keep_idx.size == 0:
            continue

        rows = df_m.iloc[keep_idx].reset_index(drop=True)
        idx = tidx[keep_idx].astype(np.int64)

        lat_v = pd.to_numeric(rows["lat"], errors="coerce").to_numpy(dtype=float)
        lon_v = pd.to_numeric(rows["lon"], errors="coerce").to_numpy(dtype=float)
        alt_v = pd.to_numeric(rows["alt"], errors="coerce").to_numpy(dtype=float)
        valid_pos = np.isfinite(lat_v) & np.isfinite(lon_v) & np.isfinite(alt_v)
        pos_idx = np.flatnonzero(valid_pos)
        if pos_idx.size == 0:
            continue

        rows = rows.iloc[pos_idx].reset_index(drop=True)
        idx = idx[pos_idx]

        e, n, u = latlon_to_enu(
            rows["lat"].to_numpy(dtype=float),
            rows["lon"].to_numpy(dtype=float),
            rows["alt"].to_numpy(dtype=float),
            lat0,
            lon0,
            alt0,
        )
        obs = np.stack([e, n, u], axis=1).astype(np.float64)

        if "confidence" in rows.columns:
            conf = pd.to_numeric(rows["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        elif "source_conf" in rows.columns:
            conf = pd.to_numeric(rows["source_conf"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            conf = np.ones((len(rows),), dtype=float)
        conf = np.clip(conf, 0.0, 1.0)

        w = (0.3 + 0.7 * conf) * float(mod_prior.get(m, 0.5))
        for k, ti in enumerate(idx.tolist()):
            sum_pos[ti] += obs[k] * w[k]
            sum_w[ti] += w[k]

    fallback = np.zeros((t_total, 3), dtype=np.float32)
    ok = sum_w > 1e-9
    fallback[ok] = (sum_pos[ok] / sum_w[ok, None]).astype(np.float32)
    return fallback, sum_w.astype(np.float32)


def apply_warmup_blend(fusion, cover_count, obs_fallback, obs_w, warmup_points, min_coverage):
    out = fusion.copy()
    n = min(int(warmup_points), out.shape[0])
    replaced = 0
    for t in range(n):
        if obs_w[t] <= 1e-9:
            continue
        cov = float(np.clip(cover_count[t] / max(float(min_coverage), 1.0), 0.0, 1.0))
        if cov >= 0.999:
            continue
        out[t] = cov * out[t] + (1.0 - cov) * obs_fallback[t]
        replaced += 1
    return out, replaced


def apply_tail_blend(fusion, cover_count, obs_fallback, obs_w, tail_points, min_coverage):
    out = fusion.copy()
    total = out.shape[0]
    n = min(int(tail_points), total)
    replaced = 0
    start = max(0, total - n)
    for t in range(start, total):
        if obs_w[t] <= 1e-9:
            continue
        cov = float(np.clip(cover_count[t] / max(float(min_coverage), 1.0), 0.0, 1.0))
        if cov >= 0.999:
            continue
        out[t] = cov * out[t] + (1.0 - cov) * obs_fallback[t]
        replaced += 1
    return out, replaced


def modality_series_enu(df_truth_u, df_mod_u, lat0, lon0, alt0, align_tolerance_s):
    if len(df_mod_u) == 0 or "timestamp" not in df_mod_u.columns:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if not all(c in df_mod_u.columns for c in ["lat", "lon", "alt"]):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    ts_truth = _as_1d_array(pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float), dtype=float)
    ts_obs = _as_1d_array(pd.to_numeric(df_mod_u["timestamp"], errors="coerce").to_numpy(dtype=float), dtype=float)
    valid_ts = np.isfinite(ts_obs).reshape(-1)
    valid_idx = np.flatnonzero(valid_ts)
    if valid_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rows = df_mod_u.iloc[valid_idx].reset_index(drop=True)
    ts_obs = ts_obs[valid_idx]
    tidx, tdiff = _nearest_truth_index(ts_obs, ts_truth)
    tidx = _as_1d_array(tidx, dtype=np.int64)
    tdiff = _as_1d_array(tdiff, dtype=float)
    keep_idx = np.flatnonzero(tdiff <= float(align_tolerance_s))
    if keep_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rows = rows.iloc[keep_idx].reset_index(drop=True)
    idx = tidx[keep_idx].astype(np.int64)

    lat_o = pd.to_numeric(rows["lat"], errors="coerce").to_numpy(dtype=float)
    lon_o = pd.to_numeric(rows["lon"], errors="coerce").to_numpy(dtype=float)
    alt_o = pd.to_numeric(rows["alt"], errors="coerce").to_numpy(dtype=float)
    valid_pos = (np.isfinite(lat_o) & np.isfinite(lon_o) & np.isfinite(alt_o)).reshape(-1)
    valid_pos_idx = np.flatnonzero(valid_pos)
    if valid_pos_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rows = rows.iloc[valid_pos_idx].reset_index(drop=True)
    idx = idx[valid_pos_idx]
    truth_hit = df_truth_u.iloc[idx].reset_index(drop=True)

    e_t, n_t, u_t = latlon_to_enu(
        truth_hit["lat"].values,
        truth_hit["lon"].values,
        truth_hit["alt"].values,
        lat0,
        lon0,
        alt0,
    )
    e_m, n_m, u_m = latlon_to_enu(
        rows["lat"].values,
        rows["lon"].values,
        rows["alt"].values,
        lat0,
        lon0,
        alt0,
    )
    gt = np.stack([e_t, n_t, u_t], axis=1).astype(np.float32)
    obs = np.stack([e_m, n_m, u_m], axis=1).astype(np.float32)
    return gt, obs


def build_sparse_windows_new(
    df_truth_u: pd.DataFrame,
    mod_frames: Dict[str, pd.DataFrame],
    lat0: float,
    lon0: float,
    alt0: float,
    modalities: List[str],
    window_size: int,
    stride: int,
    align_tolerance_s: float,
):
    ts_truth = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").to_numpy(dtype=float)
    id_col_t = _detect_id_col(df_truth_u)
    uav_value = df_truth_u.iloc[0][id_col_t] if id_col_t is not None else None

    aligned_by_t = {}
    for m in modalities:
        df_m = mod_frames.get(m, pd.DataFrame())
        if len(df_m) == 0 or "timestamp" not in df_m.columns:
            aligned_by_t[m] = {}
            continue

        id_col_m = _detect_id_col(df_m)
        if id_col_m is not None and uav_value is not None:
            df_m = df_m[df_m[id_col_m] == uav_value]
        if len(df_m) == 0:
            aligned_by_t[m] = {}
            continue

        ts_obs = _as_1d_array(pd.to_numeric(df_m["timestamp"], errors="coerce").to_numpy(dtype=float), dtype=float)
        valid = np.isfinite(ts_obs).reshape(-1)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            aligned_by_t[m] = {}
            continue
        df_m = df_m.iloc[valid_idx].copy().reset_index(drop=True)
        ts_obs = ts_obs[valid_idx]
        if len(df_m) == 0:
            aligned_by_t[m] = {}
            continue

        tidx, tdiff = _nearest_truth_index(ts_obs, ts_truth)
        tidx = _as_1d_array(tidx, dtype=np.int64)
        tdiff = _as_1d_array(tdiff, dtype=float)
        keep_idx = np.flatnonzero(tdiff <= float(align_tolerance_s))
        if keep_idx.size == 0:
            aligned_by_t[m] = {}
            continue

        df_m = df_m.iloc[keep_idx].copy().reset_index(drop=True)
        df_m["t_idx"] = tidx[keep_idx].astype(np.int64)
        aligned_by_t[m] = {int(k): g for k, g in df_m.groupby("t_idx")}

    windows = []
    starts = []
    t_total = len(df_truth_u)
    t_denom = max(window_size - 1, 1)
    modality_width = max((MODALITY_TO_ID[m] for m in modalities), default=-1) + 1
    for s in range(0, t_total - window_size + 1, stride):
        feats = []
        t_ids = []
        m_ids = []

        for ti in range(window_size):
            t_global = s + ti
            for m in modalities:
                rows = aligned_by_t.get(m, {}).get(int(t_global))
                if rows is None or len(rows) == 0:
                    continue
                for _, row in rows.iterrows():
                    feat = _build_node_feature(
                        row=row,
                        modality=m,
                        ti_local=ti,
                        t_denom=t_denom,
                        lat0=float(lat0),
                        lon0=float(lon0),
                        alt0=float(alt0),
                        n_modalities=max(modality_width, len(modalities)),
                    )
                    feats.append(feat)
                    t_ids.append(ti)
                    m_ids.append(MODALITY_TO_ID[m])

        if len(feats) == 0:
            continue

        windows.append(
            {
                "node_feat": np.stack(feats).astype(np.float32),
                "node_t": np.array(t_ids, dtype=np.int64),
                "node_m": np.array(m_ids, dtype=np.int64),
            }
        )
        starts.append(s)

    return windows, starts


def build_sparse_windows_legacy(df_truth_u, mod_frames, lat0, lon0, alt0, modalities, window_size, stride, in_dim):
    ts_all = pd.to_numeric(df_truth_u["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    vx_all = pd.to_numeric(df_truth_u["vx"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    vy_all = pd.to_numeric(df_truth_u["vy"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    vz_all = pd.to_numeric(df_truth_u["vz"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    spd_all = pd.to_numeric(df_truth_u["speed"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    id_col_t = _detect_id_col(df_truth_u)
    uav_value = df_truth_u.iloc[0][id_col_t] if id_col_t is not None else None

    mod_maps = {}
    for m in modalities:
        df_m = mod_frames.get(m, pd.DataFrame())
        id_col_m = _detect_id_col(df_m)
        if id_col_m is not None and uav_value is not None:
            df_m = df_m[df_m[id_col_m] == uav_value]
        if len(df_m) == 0:
            mod_maps[m] = {}
            continue
        conf_col = "confidence" if "confidence" in df_m.columns else "source_conf"
        cols = [c for c in ["timestamp", "lat", "lon", "alt", conf_col] if c in df_m.columns]
        if len(cols) < 5:
            mod_maps[m] = {}
            continue
        mod_maps[m] = {
            int(r.timestamp): r for r in df_m[cols].itertuples(index=False)
        }

    windows = []
    starts = []
    t_total = len(df_truth_u)
    denom = max(window_size - 1, 1)
    onehot_offset = 9
    for s in range(0, t_total - window_size + 1, stride):
        ts_win = ts_all[s : s + window_size]
        vx = vx_all[s : s + window_size]
        vy = vy_all[s : s + window_size]
        vz = vz_all[s : s + window_size]
        spd = spd_all[s : s + window_size]
        feats = []
        t_ids = []
        m_ids = []

        for ti in range(window_size):
            ts = int(ts_win[ti])
            for m in modalities:
                row = mod_maps.get(m, {}).get(ts)
                if row is None:
                    continue
                east, north, up = latlon_to_enu(row.lat, row.lon, row.alt, lat0, lon0, alt0)
                feat = np.zeros((in_dim,), dtype=np.float32)
                if in_dim > 2:
                    feat[0:3] = np.array([east, north, up], dtype=np.float32)
                if in_dim > 3:
                    feat[3] = vx[ti]
                if in_dim > 4:
                    feat[4] = vy[ti]
                if in_dim > 5:
                    feat[5] = vz[ti]
                if in_dim > 6:
                    feat[6] = spd[ti]
                if in_dim > 7:
                    conf_v = getattr(row, "confidence", getattr(row, "source_conf", 0.0))
                    feat[7] = float(conf_v)
                if in_dim > 8:
                    feat[8] = float(ti / denom)
                hot_i = onehot_offset + MODALITY_TO_ID[m]
                if 0 <= hot_i < in_dim:
                    feat[hot_i] = 1.0
                feats.append(feat)
                t_ids.append(ti)
                m_ids.append(MODALITY_TO_ID[m])

        if len(feats) == 0:
            continue
        windows.append(
            {
                "node_feat": np.stack(feats).astype(np.float32),
                "node_t": np.array(t_ids, dtype=np.int64),
                "node_m": np.array(m_ids, dtype=np.int64),
            }
        )
        starts.append(s)
    return windows, starts


def fit_feature_dim(node_feat: torch.Tensor, target_dim: int):
    cur_dim = int(node_feat.shape[1])
    if cur_dim == target_dim:
        return node_feat
    if cur_dim > target_dim:
        return node_feat[:, :target_dim]
    pad = torch.zeros((node_feat.shape[0], target_dim - cur_dim), device=node_feat.device, dtype=node_feat.dtype)
    return torch.cat([node_feat, pad], dim=1)


def main():
    model, x_mean, x_std, y_mean, y_std, runtime = load_model_and_runtime(
        model_path=MODEL_PATH,
        norm_path=NORM_PATH,
        device=DEVICE,
    )

    in_dim = int(runtime["in_dim"])
    window_size = int(runtime["window_size"])
    stride = int(runtime["stride"])
    align_tolerance_s = float(runtime["align_tolerance_s"])
    modalities = list(runtime["modalities"])

    truth = pd.read_csv(os.path.join(BATCH_DIR, "truth.csv"))
    id_col = _detect_id_col(truth)
    if id_col is None:
        raise RuntimeError("truth.csv must contain `id` or `uav_id`")

    uavs = list(pd.Series(truth[id_col]).dropna().unique())
    if len(uavs) == 0:
        raise RuntimeError("no UAV found in truth.csv")
    if UID < 0 or UID >= len(uavs):
        raise RuntimeError(f"UID index out of range: {UID}, total={len(uavs)}")

    uav = uavs[UID]
    print("UAV:", uav)
    print(
        f"[Runtime] model={os.path.basename(MODEL_PATH)} | in_dim={in_dim} | "
        f"window={window_size} | stride={stride} | mods={modalities}"
    )

    df_t = truth[truth[id_col] == uav].sort_values("timestamp").reset_index(drop=True)
    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]

    e_gt, n_gt, u_gt = latlon_to_enu(
        df_t["lat"].values,
        df_t["lon"].values,
        df_t["alt"].values,
        lat0,
        lon0,
        alt0,
    )
    truth_full_enu = np.stack([e_gt, n_gt, u_gt], axis=-1).astype(np.float32)

    mod_frames = {}
    for m in modalities:
        fname = "5g_a.csv" if m == "5g_a" else f"{m}.csv"
        p = os.path.join(BATCH_DIR, fname)
        mod_frames[m] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    if in_dim >= NODE_FEAT_DIM:
        windows, starts = build_sparse_windows_new(
            df_truth_u=df_t,
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
        windows, starts = build_sparse_windows_legacy(
            df_truth_u=df_t,
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
        raise RuntimeError("No valid sparse windows for inference")

    obs_fallback_enu, obs_fallback_w = build_obs_fallback_series(
        df_truth_u=df_t,
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
        window_weights.append(estimate_window_quality(w["node_feat"]))
        node_feat = torch.tensor(w["node_feat"], dtype=torch.float32, device=DEVICE)
        node_t = torch.tensor(w["node_t"], dtype=torch.long, device=DEVICE)
        node_m = torch.tensor(w["node_m"], dtype=torch.long, device=DEVICE)
        node_feat = fit_feature_dim(node_feat, int(x_mean.numel()))
        node_feat = (node_feat - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            pred_norm = model(
                node_feat=node_feat.unsqueeze(0),
                node_t=node_t.unsqueeze(0),
                node_m=node_m.unsqueeze(0),
                node_mask=torch.ones((1, node_feat.shape[0]), dtype=torch.float32, device=DEVICE),
                window_size=window_size,
            )[0]
        pred = pred_norm * y_std + y_mean
        preds.append(pred.cpu().numpy())

    fusion_enu, cover_weight, cover_count = merge_windows(
        np.array(preds),
        starts,
        t_total=len(truth_full_enu),
        window=window_size,
        window_weights=window_weights,
        edge_taper_min=MERGE_EDGE_TAPER_MIN,
    )
    fusion_enu, warmup_replaced = apply_warmup_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        warmup_points=WARMUP_POINTS,
        min_coverage=WARMUP_MIN_COVERAGE,
    )
    fusion_enu, tail_replaced = apply_tail_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        tail_points=TAIL_POINTS,
        min_coverage=WARMUP_MIN_COVERAGE,
    )

    fusion_err = calc_err(fusion_enu, truth_full_enu)
    print("\nFusion Error:", fusion_err)
    print(
        f"[Merge] windows={len(windows)} | "
        f"edge_taper_min={MERGE_EDGE_TAPER_MIN:.2f} | "
        f"warmup_points={WARMUP_POINTS} | warmup_blended={warmup_replaced} | "
        f"tail_points={TAIL_POINTS} | tail_blended={tail_replaced}"
    )
    head_n = min(20, len(cover_count))
    tail_n = min(20, len(cover_count))
    print("[Merge] head coverage count:", cover_count[:head_n].astype(int).tolist())
    print("[Merge] head coverage weight:", np.round(cover_weight[:head_n], 3).tolist())
    print("[Merge] tail coverage count:", cover_count[-tail_n:].astype(int).tolist())
    print("[Merge] tail coverage weight:", np.round(cover_weight[-tail_n:], 3).tolist())

    for m in modalities:
        df_mod_u = mod_frames[m]
        id_col_m = _detect_id_col(df_mod_u)
        if id_col_m is not None:
            df_mod_u = df_mod_u[df_mod_u[id_col_m] == uav]
        gt_m, obs_m = modality_series_enu(df_t, df_mod_u, lat0, lon0, alt0, align_tolerance_s)
        print(f"{m} Error:", calc_err(obs_m, gt_m))

    pred_lat, pred_lon, _ = enu_to_llh(
        fusion_enu[:, 0],
        fusion_enu[:, 1],
        fusion_enu[:, 2],
        lat0,
        lon0,
        alt0,
    )
    truth_lat = df_t["lat"].values
    truth_lon = df_t["lon"].values

    plt.figure(figsize=(10, 8))
    plt.scatter(truth_lon, truth_lat, s=15, c="black", label="Truth")
    plt.scatter(pred_lon, pred_lat, s=15, c="red", label="Fusion")

    colors = {"gps": "orange", "radar": "blue", "5g_a": "green", "tdoa": "purple", "acoustic": "brown"}
    for m in modalities:
        d = mod_frames[m]
        id_col_m = _detect_id_col(d)
        if id_col_m is not None:
            d = d[d[id_col_m] == uav]
        if len(d) == 0:
            continue
        if all(c in d.columns for c in ["lon", "lat"]):
            plt.scatter(d["lon"].values, d["lat"].values, s=8, c=colors.get(m, "gray"), label=m, alpha=0.35)

    plt.title("Truth / Fusion / Modalities")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
