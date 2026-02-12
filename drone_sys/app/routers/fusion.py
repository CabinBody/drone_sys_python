from __future__ import annotations

import math
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from drone_sys.app.core.datasetBuilder import transfer_confidence as tc

router = APIRouter(tags=["fusion"])

_REQ_MODALITIES = ["gps", "radar", "fiveg", "tdoa", "acoustic"]
_QUALITY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gps": {"Nsat": 0.0, "DOP": 99.0, "RTK": "NONE"},
    "radar": {"E": 0.0, "Ptrk": 0.0},
    "fiveg": {"SNR": 0.0, "RSSI": -120.0, "d": 300.0, "ploss": 1.0},
    "tdoa": {"e": 50.0, "eps_sync": 200.0},
    "acoustic": {"SNRa": 0.0, "n": 1.0},
}


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _is_finite_number(v: Any) -> bool:
    fv = _to_float(v, float("nan"))
    return math.isfinite(fv)


def _normalize_modality_key(k: str) -> Optional[str]:
    if not isinstance(k, str):
        return None
    t = k.strip().lower()
    if t in ("gps", "radar", "tdoa", "acoustic"):
        return t
    if t in ("fiveg", "5g", "5ga", "5g_a"):
        return "fiveg"
    return None


def _extract_packets(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        packets = payload.get("data", payload.get("packets", payload.get("list")))
        if isinstance(packets, list):
            uav_id = payload.get("uav_id")
            return packets, str(uav_id) if uav_id is not None else None
    raise HTTPException(status_code=400, detail="Request body must be a list or an object with `data` list.")


def _extract_mod_row(packet: Dict[str, Any], modality: str) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {}
    for k, v in packet.items():
        nk = _normalize_modality_key(k)
        if nk == modality and isinstance(v, dict):
            return dict(v)
    return {}


def _extract_timestamp(packet: Dict[str, Any], modality_rows: Dict[str, Dict[str, Any]], fallback_idx: int) -> float:
    if isinstance(packet, dict) and _is_finite_number(packet.get("timestamp")):
        return _to_float(packet.get("timestamp"), float(fallback_idx))
    for m in _REQ_MODALITIES:
        row = modality_rows.get(m, {})
        if _is_finite_number(row.get("timestamp")):
            return _to_float(row.get("timestamp"), float(fallback_idx))
    return float(fallback_idx)


def _extract_uav_id(explicit_uav_id: Optional[str], packet: Dict[str, Any], modality_rows: Dict[str, Dict[str, Any]]) -> str:
    if explicit_uav_id:
        return explicit_uav_id
    if isinstance(packet, dict):
        if packet.get("uav_id") is not None:
            return str(packet.get("uav_id"))
        if packet.get("id") is not None:
            return str(packet.get("id"))
    for m in _REQ_MODALITIES:
        row = modality_rows.get(m, {})
        if row.get("uav_id") is not None:
            return str(row.get("uav_id"))
        if row.get("id") is not None:
            return str(row.get("id"))
    return "UAV_HTTP"


def _fill_base_fields(row: Dict[str, Any], timestamp: float, uav_id: str) -> Dict[str, Any]:
    out = dict(row)
    out["timestamp"] = _to_float(out.get("timestamp", timestamp), timestamp)
    out["uav_id"] = str(out.get("uav_id", out.get("id", uav_id)))
    out["arrival_time"] = _to_float(out.get("arrival_time", out["timestamp"]), out["timestamp"])
    out["scenario_tag"] = str(out.get("scenario_tag", "HTTP"))
    out["missing_flag"] = int(_to_float(out.get("missing_flag", 0.0), 0.0) > 0.0)
    return out


def _has_valid_pos(row: Dict[str, Any]) -> bool:
    return _is_finite_number(row.get("lat")) and _is_finite_number(row.get("lon")) and _is_finite_number(row.get("alt"))


def _build_truth_rows(per_mod_rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    n = len(per_mod_rows["gps"])
    truth_rows: List[Dict[str, Any]] = []
    last_pos: Optional[Tuple[float, float, float]] = None
    for i in range(n):
        candidate: Optional[Dict[str, Any]] = None
        for m in ("gps", "radar", "fiveg", "tdoa"):
            row = per_mod_rows[m][i]
            if int(_to_float(row.get("missing_flag", 0), 0)) > 0:
                continue
            if _has_valid_pos(row):
                candidate = row
                break
        if candidate is None:
            for m in ("gps", "radar", "fiveg", "tdoa"):
                row = per_mod_rows[m][i]
                if _has_valid_pos(row):
                    candidate = row
                    break

        ts = _to_float(per_mod_rows["gps"][i].get("timestamp", i), float(i))
        uav_id = str(per_mod_rows["gps"][i].get("uav_id", "UAV_HTTP"))
        if candidate is not None:
            lat = _to_float(candidate.get("lat"), float("nan"))
            lon = _to_float(candidate.get("lon"), float("nan"))
            alt = _to_float(candidate.get("alt"), float("nan"))
            if math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt):
                last_pos = (lat, lon, alt)
            vx = _to_float(candidate.get("vx", 0.0), 0.0)
            vy = _to_float(candidate.get("vy", 0.0), 0.0)
            vz = _to_float(candidate.get("vz", 0.0), 0.0)
            speed = _to_float(candidate.get("speed", 0.0), 0.0)
        else:
            vx, vy, vz, speed = 0.0, 0.0, 0.0, 0.0

        if last_pos is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of gps/radar/fiveg/tdoa must provide valid lat/lon/alt in the first packet.",
            )
        lat, lon, alt = last_pos
        truth_rows.append(
            {
                "timestamp": ts,
                "uav_id": uav_id,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": speed,
            }
        )
    return truth_rows


def _rows_to_csv(df_rows: List[Dict[str, Any]], path: str) -> None:
    pd.DataFrame(df_rows).to_csv(path, index=False)


def _build_raw_dataset_from_packets(packets: List[Dict[str, Any]], req_uav_id: Optional[str], raw_dir: str) -> None:
    if len(packets) != 20:
        raise HTTPException(status_code=400, detail=f"Expected exactly 20 packets, got {len(packets)}")

    per_mod_rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in _REQ_MODALITIES}
    current_uav_id = req_uav_id
    for i, packet in enumerate(packets):
        if not isinstance(packet, dict):
            raise HTTPException(status_code=400, detail=f"Packet index {i} is not an object.")

        modality_rows = {m: _extract_mod_row(packet, m) for m in _REQ_MODALITIES}
        ts = _extract_timestamp(packet, modality_rows, fallback_idx=i)
        uav_id = _extract_uav_id(current_uav_id, packet, modality_rows)
        if current_uav_id is None:
            current_uav_id = uav_id

        for m in _REQ_MODALITIES:
            row = _fill_base_fields(modality_rows[m], timestamp=ts, uav_id=current_uav_id)
            row.setdefault("lat", float("nan"))
            row.setdefault("lon", float("nan"))
            row.setdefault("alt", float("nan"))
            row.setdefault("vx", 0.0)
            row.setdefault("vy", 0.0)
            row.setdefault("vz", 0.0)
            row.setdefault("speed", 0.0)
            row.update({k: row.get(k, v) for k, v in _QUALITY_DEFAULTS[m].items()})
            if m == "acoustic":
                row.setdefault("detected_flag", 0)
                row.setdefault("spl_db", float("nan"))
                row.setdefault("acoustic_energy", float("nan"))
            per_mod_rows[m].append(row)

    truth_rows = _build_truth_rows(per_mod_rows)

    _rows_to_csv(truth_rows, os.path.join(raw_dir, "truth.csv"))
    _rows_to_csv(per_mod_rows["gps"], os.path.join(raw_dir, "gps.csv"))
    _rows_to_csv(per_mod_rows["radar"], os.path.join(raw_dir, "radar.csv"))
    _rows_to_csv(per_mod_rows["fiveg"], os.path.join(raw_dir, "5g_a.csv"))
    _rows_to_csv(per_mod_rows["tdoa"], os.path.join(raw_dir, "tdoa.csv"))
    _rows_to_csv(per_mod_rows["acoustic"], os.path.join(raw_dir, "acoustic.csv"))


@lru_cache(maxsize=1)
def _load_runtime_bundle():
    # Delay heavy imports until the endpoint is called.
    drone_fusion_dir = Path(__file__).resolve().parents[1] / "core" / "droneFusion"
    if str(drone_fusion_dir) not in sys.path:
        sys.path.insert(0, str(drone_fusion_dir))
    try:
        import torch  # noqa: WPS433
        from drone_sys.app.core.droneFusion import inference as inf  # noqa: WPS433
    except Exception as ex:
        raise RuntimeError(f"Failed to import inference runtime dependencies: {ex}") from ex

    model, x_mean, x_std, y_mean, y_std, runtime = inf.load_model_and_runtime(
        model_path=inf.MODEL_PATH,
        norm_path=inf.NORM_PATH,
        device=inf.DEVICE,
    )
    return torch, inf, model, x_mean, x_std, y_mean, y_std, runtime


def _run_model_inference(processed_dir: str) -> List[Dict[str, float]]:
    torch, inf, model, x_mean, x_std, y_mean, y_std, runtime = _load_runtime_bundle()
    in_dim = int(runtime["in_dim"])
    window_size = int(runtime["window_size"])
    stride = int(runtime["stride"])
    align_tolerance_s = float(runtime["align_tolerance_s"])
    modalities = list(runtime["modalities"])

    truth_path = os.path.join(processed_dir, "truth.csv")
    truth = pd.read_csv(truth_path)
    if truth.empty:
        raise RuntimeError("truth.csv is empty after transfer_confidence conversion.")

    id_col = inf._detect_id_col(truth)
    if id_col is None:
        raise RuntimeError("truth.csv must contain `uav_id` or `id`.")

    uav_value = truth.iloc[0][id_col]
    df_t = truth[truth[id_col] == uav_value].sort_values("timestamp").reset_index(drop=True)
    if len(df_t) < window_size:
        raise RuntimeError(f"Need at least {window_size} truth rows, got {len(df_t)}.")

    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]
    e_gt, n_gt, u_gt = inf.latlon_to_enu(
        df_t["lat"].to_numpy(dtype=float),
        df_t["lon"].to_numpy(dtype=float),
        df_t["alt"].to_numpy(dtype=float),
        float(lat0),
        float(lon0),
        float(alt0),
    )
    truth_full_enu = pd.DataFrame({"e": e_gt, "n": n_gt, "u": u_gt}).to_numpy(dtype="float32")

    mod_frames: Dict[str, pd.DataFrame] = {}
    for m in modalities:
        fname = "5g_a.csv" if m == "5g_a" else f"{m}.csv"
        path = os.path.join(processed_dir, fname)
        mod_frames[m] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    if in_dim >= inf.NODE_FEAT_DIM:
        windows, starts = inf.build_sparse_windows_new(
            df_truth_u=df_t,
            mod_frames=mod_frames,
            lat0=float(lat0),
            lon0=float(lon0),
            alt0=float(alt0),
            modalities=modalities,
            window_size=window_size,
            stride=stride,
            align_tolerance_s=align_tolerance_s,
        )
    else:
        windows, starts = inf.build_sparse_windows_legacy(
            df_truth_u=df_t,
            mod_frames=mod_frames,
            lat0=float(lat0),
            lon0=float(lon0),
            alt0=float(alt0),
            modalities=modalities,
            window_size=window_size,
            stride=stride,
            in_dim=in_dim,
        )
    if len(windows) == 0:
        raise RuntimeError("No valid sparse windows for inference.")

    obs_fallback_enu, obs_fallback_w = inf.build_obs_fallback_series(
        df_truth_u=df_t,
        mod_frames=mod_frames,
        modalities=modalities,
        lat0=float(lat0),
        lon0=float(lon0),
        alt0=float(alt0),
        align_tolerance_s=align_tolerance_s,
    )

    preds: List[Any] = []
    window_weights: List[float] = []
    for w in windows:
        window_weights.append(inf.estimate_window_quality(w["node_feat"]))
        node_feat = torch.tensor(w["node_feat"], dtype=torch.float32, device=inf.DEVICE)
        node_t = torch.tensor(w["node_t"], dtype=torch.long, device=inf.DEVICE)
        node_m = torch.tensor(w["node_m"], dtype=torch.long, device=inf.DEVICE)
        node_feat = inf.fit_feature_dim(node_feat, int(x_mean.numel()))
        node_feat = (node_feat - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
        with torch.no_grad():
            pred_norm = model(
                node_feat=node_feat.unsqueeze(0),
                node_t=node_t.unsqueeze(0),
                node_m=node_m.unsqueeze(0),
                node_mask=torch.ones((1, node_feat.shape[0]), dtype=torch.float32, device=inf.DEVICE),
                window_size=window_size,
            )[0]
        pred = pred_norm * y_std + y_mean
        preds.append(pred.detach().cpu().numpy())

    fusion_enu, _, cover_count = inf.merge_windows(
        preds,
        starts,
        t_total=len(truth_full_enu),
        window=window_size,
        window_weights=window_weights,
        edge_taper_min=inf.MERGE_EDGE_TAPER_MIN,
    )
    fusion_enu, _ = inf.apply_warmup_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        warmup_points=inf.WARMUP_POINTS,
        min_coverage=inf.WARMUP_MIN_COVERAGE,
    )
    fusion_enu, _ = inf.apply_tail_blend(
        fusion=fusion_enu,
        cover_count=cover_count,
        obs_fallback=obs_fallback_enu,
        obs_w=obs_fallback_w,
        tail_points=inf.TAIL_POINTS,
        min_coverage=inf.WARMUP_MIN_COVERAGE,
    )

    pred_lat, pred_lon, pred_alt = inf.enu_to_llh(
        fusion_enu[:, 0],
        fusion_enu[:, 1],
        fusion_enu[:, 2],
        float(lat0),
        float(lon0),
        float(alt0),
    )

    ts_arr = pd.to_numeric(df_t["timestamp"], errors="coerce").to_numpy(dtype=float)
    output: List[Dict[str, float]] = []
    for i in range(len(ts_arr)):
        output.append(
            {
                "timestamp": float(ts_arr[i]),
                "lat": float(pred_lat[i]),
                "lon": float(pred_lon[i]),
                "alt": float(pred_alt[i]),
            }
        )
    return output


@router.post("/fusion/run")
@router.post("/run")
def run_fusion_http(payload: Any = Body(...)):
    """
    Input:
    - list[20] of packets, each packet contains 5 modalities (gps/radar/fiveg/tdoa/acoustic)
    - or object with data: list[20]

    Output:
    - list of {timestamp, lat, lon, alt}
    """
    packets, req_uav_id = _extract_packets(payload)
    with tempfile.TemporaryDirectory(prefix="fusion_http_") as tmp:
        raw_dir = os.path.join(tmp, "raw")
        processed_dir = os.path.join(tmp, "processed")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        _build_raw_dataset_from_packets(packets=packets, req_uav_id=req_uav_id, raw_dir=raw_dir)
        cfg = tc.default_cfg()
        tc.process_one_dir(in_dir=raw_dir, out_dir=processed_dir, cfg=cfg, label="http")
        try:
            return _run_model_inference(processed_dir=processed_dir)
        except HTTPException:
            raise
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Fusion inference failed: {ex}") from ex
