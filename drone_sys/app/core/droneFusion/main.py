import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataset import to_enu_single_point

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_sys.app.routers import fusion as fusion_router  # noqa: E402
from drone_sys.app.core.datasetBuilder import generate_bluesky_dataset as gbd  # noqa: E402

MODALITY_ORDER = ["gps", "radar", "fiveg", "tdoa", "acoustic"]
POS_MODALITIES = ["gps", "radar", "fiveg", "tdoa"]


def _jsonable(v: Any):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def calc_track_error(pred_enu: np.ndarray, truth_enu: np.ndarray):
    n = min(len(pred_enu), len(truth_enu))
    if n <= 0:
        return {
            "n": 0,
            "rmse_3d": float("nan"),
            "mae_3d": float("nan"),
            "max_3d": float("nan"),
            "rmse_xy": float("nan"),
            "fde_xy": float("nan"),
        }
    pred = np.asarray(pred_enu[:n], dtype=float)
    truth = np.asarray(truth_enu[:n], dtype=float)
    diff = pred - truth
    dist3 = np.linalg.norm(diff, axis=1)
    dist2 = np.linalg.norm(diff[:, :2], axis=1)
    return {
        "n": int(n),
        "rmse_3d": float(np.sqrt(np.mean(dist3 ** 2))),
        "mae_3d": float(np.mean(np.abs(dist3))),
        "max_3d": float(np.max(dist3)),
        "rmse_xy": float(np.sqrt(np.mean(dist2 ** 2))),
        "fde_xy": float(np.linalg.norm(diff[-1, :2])),
    }


def calc_track_error_masked(pred_enu: np.ndarray, truth_enu: np.ndarray, valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    n = min(len(pred_enu), len(truth_enu), len(valid))
    if n <= 0:
        return calc_track_error(np.zeros((0, 3)), np.zeros((0, 3)))
    idx = np.flatnonzero(valid[:n])
    if idx.size == 0:
        return calc_track_error(np.zeros((0, 3)), np.zeros((0, 3)))
    return calc_track_error(np.asarray(pred_enu[:n])[idx], np.asarray(truth_enu[:n])[idx])


def print_track_error(tag: str, metrics: dict):
    print(
        f"[Demo][{tag}] n={metrics['n']} | "
        f"RMSE3D={metrics['rmse_3d']:.3f} | "
        f"MAE3D={metrics['mae_3d']:.3f} | "
        f"MAX3D={metrics['max_3d']:.3f} | "
        f"RMSE_XY={metrics['rmse_xy']:.3f} | "
        f"FDE_XY={metrics['fde_xy']:.3f}"
    )


def llh_seq_to_enu(lat, lon, alt, lat0, lon0, alt0):
    arr = []
    for i in range(len(lat)):
        if not (np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(alt[i])):
            arr.append([np.nan, np.nan, np.nan])
        else:
            arr.append(to_enu_single_point(float(lat[i]), float(lon[i]), float(alt[i]), lat0, lon0, alt0))
    return np.asarray(arr, dtype=float)


def simulate_truth(T: int, cfg: dict, rng: np.random.Generator):
    sim = cfg["simulation"]
    lat0 = float(sim["base_lat"])
    lon0 = float(sim["base_lon"])
    alt0 = float(np.mean(sim["alt_range_m"]))

    dt = 1.0
    speed = rng.uniform(sim["speed_range_mps"][0], sim["speed_range_mps"][1], size=T)
    heading = np.empty((T,), dtype=float)
    heading[0] = float(rng.uniform(0.0, 2.0 * np.pi))
    for i in range(1, T):
        heading[i] = heading[i - 1] + rng.normal(0.0, 0.10)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    vz = rng.normal(0.0, 0.35, size=T)

    east = np.cumsum(vx * dt)
    north = np.cumsum(vy * dt)
    up_rel = np.cumsum(vz * dt)
    east -= east[0]
    north -= north[0]
    up_rel -= up_rel[0]

    alt = alt0 + up_rel
    lat = lat0 + north / 111000.0
    lon = lon0 + east / (111000.0 * np.cos(np.radians(lat0)))

    t_rel = np.arange(T, dtype=float)
    t_abs = float(sim.get("start_epoch", 0)) + t_rel

    return {
        "lat0": lat0,
        "lon0": lon0,
        "alt0": alt0,
        "lat": lat.astype(float),
        "lon": lon.astype(float),
        "alt": alt.astype(float),
        "east": east.astype(float),
        "north": north.astype(float),
        "up": up_rel.astype(float),
        "vx": vx.astype(float),
        "vy": vy.astype(float),
        "vz": vz.astype(float),
        "speed": speed.astype(float),
        "t_rel": t_rel,
        "t_abs": t_abs,
    }


def scenario_tags_for_demo(T: int, cfg: dict, rng: np.random.Generator):
    tags = gbd.scenario_series(np.arange(T, dtype=float), cfg["scenario_mix"], cfg["scenario_duration_s"], rng)
    # Ensure the 20-point demo is not a single regime; keep distribution-like labels but force variation if needed.
    if T >= 12 and len(set(tags.tolist())) < 2:
        block = max(T // 4, 1)
        manual = ["A", "B", "C", "B"]
        out = []
        for m in manual:
            out.extend([m] * block)
        out = out[:T]
        if len(out) < T:
            out.extend(["B"] * (T - len(out)))
        tags = np.asarray(out, dtype="<U1")
    return tags


def build_blackout_plan_demo(uid: int, cfg: dict):
    mc = cfg.get("missing_control", {})
    rb = mc.get("random_blackout", {})
    rb = deepcopy(rb)
    rb["enabled"] = True
    rb["event_count"] = [1, 2]
    rb["duration_s"] = [3.0, 6.0]
    # Keep GPS mostly available in the synthetic base request so the GPS/no-GPS comparison is meaningful.
    rb["modalities"] = ["radar", "fiveg", "tdoa", "acoustic"]
    cfg_demo = deepcopy(cfg)
    cfg_demo.setdefault("missing_control", {})["random_blackout"] = rb
    return gbd.build_random_blackout_plan(uid, cfg_demo)


def generate_modality_rows(
    modality: str,
    uid: int,
    truth: dict,
    tags: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    blackout_intervals: List[Tuple[float, float]],
):
    T = len(truth["t_rel"])
    mc = cfg["modalities"][modality]
    prev_t = float(truth["t_rel"][0]) if T > 0 else 0.0
    drift_st = {"vec": np.zeros(3), "bias": np.zeros(3), "remain": 0.0}
    mod_st = {"fault": 0, "trend": 0.0}

    rows: List[Dict[str, Any]] = []
    obs_enu = np.full((T, 3), np.nan, dtype=float)
    valid_mask = np.zeros((T,), dtype=bool)

    for i in range(T):
        tt = float(truth["t_rel"][i])
        ta = float(truth["t_abs"][i])
        tg = str(tags[i])
        lat_t = float(truth["lat"][i])
        lon_t = float(truth["lon"][i])
        alt_t = float(truth["alt"][i])
        vx_t = float(truth["vx"][i])
        vy_t = float(truth["vy"][i])
        vz_t = float(truth["vz"][i])

        pen = float(gbd.env_penalty(modality, lat_t, lon_t, ta, cfg))
        q, pos_s, vel_s, delay_ms, miss_p = gbd.sample_quality(modality, tg, pen, mod_st, cfg, rng)
        miss_p_adj, force_missing, force_available = gbd.apply_missing_control(modality, tg, miss_p, cfg)
        blackout_now = int(gbd.in_blackout(tt, blackout_intervals))
        force_missing = bool(force_missing or blackout_now == 1)

        dt = max(tt - prev_t, 1.0)
        drift = gbd.update_drift(drift_st, tg, dt, cfg, rng)
        prev_t = tt
        arrival = ta + max(float(delay_ms), 0.0) / 1000.0 + float(rng.normal(0.0, float(mc["arrival_jitter_s"])))

        if modality == "acoustic":
            if force_missing:
                detected = 0
                miss = 1
            else:
                p_det = float(gbd.acoustic_detect_prob(tg, float(q["SNRa"]), float(q["n"]), pen, cfg))
                detected = int(rng.random() < p_det)
                if force_available:
                    detected = 1
                    miss = 0
                else:
                    miss = int((detected == 0) or (rng.random() < float(gbd.clamp(miss_p_adj, 0.0, 1.0))))

            if detected == 1 and miss == 0:
                spl_lo, spl_hi = cfg.get("acoustic_detection", {}).get("spl_range_db", [35.0, 110.0])
                spl = float(
                    gbd.clamp(
                        40.0 + 1.35 * float(q["SNRa"]) - 8.0 * float(q["n"]) - 7.0 * pen + rng.normal(0.0, 2.2),
                        float(spl_lo),
                        float(spl_hi),
                    )
                )
                energy = float(gbd.clamp((10.0 ** (spl / 20.0)) / 1e5, 0.0, 1.0))
            else:
                spl = float("nan")
                energy = float("nan")
                q["SNRa"] = 0.0
                q["n"] = 1.0

            row = {
                "timestamp": ta,
                "uav_id": f"UAV{uid:05d}",
                "detected_flag": int(detected),
                "spl_db": spl,
                "acoustic_energy": energy,
                "scenario_tag": tg,
                "missing_flag": int(miss),
                "blackout_flag": int(blackout_now),
                "arrival_time": arrival,
                "SNRa": float(q["SNRa"]),
                "n": float(q["n"]),
            }
            rows.append(row)
            continue

        ne = float(rng.normal(0.0, float(pos_s)))
        nn = float(rng.normal(0.0, float(pos_s)))
        nu = float(rng.normal(0.0, 0.6 * float(pos_s)))
        lat = lat_t + float(gbd.meters_to_deg_lat(nn + float(drift[1])))
        lon = lon_t + float(gbd.meters_to_deg_lon(ne + float(drift[0]), lat_t))
        alt = alt_t + nu + float(drift[2])

        vx = vx_t + float(rng.normal(0.0, float(vel_s)))
        vy = vy_t + float(rng.normal(0.0, float(vel_s)))
        vz = vz_t + float(rng.normal(0.0, 0.8 * float(vel_s)))
        speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))

        if force_missing:
            miss = 1
        elif force_available:
            miss = 0
        else:
            miss = int(rng.random() < float(gbd.clamp(miss_p_adj, 0.0, 1.0)))

        if miss == 1:
            lat = lon = alt = vx = vy = vz = speed = float("nan")
        else:
            obs_enu[i] = np.asarray(to_enu_single_point(lat, lon, alt, truth["lat0"], truth["lon0"], truth["alt0"]), dtype=float)
            valid_mask[i] = True

        row = {
            "timestamp": ta,
            "uav_id": f"UAV{uid:05d}",
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "speed": speed,
            "scenario_tag": tg,
            "missing_flag": int(miss),
            "blackout_flag": int(blackout_now),
            "arrival_time": arrival,
        }
        for k, v in q.items():
            row[k] = _jsonable(v)
        rows.append(row)

    return rows, obs_enu, valid_mask


def ensure_first_packet_observable(rows_by_mod: Dict[str, List[Dict[str, Any]]], truth: dict):
    first_ok = False
    for m in POS_MODALITIES:
        row = rows_by_mod[m][0]
        if int(row.get("missing_flag", 0)) == 0 and row.get("lat") is not None and row.get("lon") is not None and row.get("alt") is not None:
            first_ok = True
            break
    if first_ok:
        return

    # Guarantee at least one positional modality in packet[0] so fusion router can construct pseudo-truth.
    gps = rows_by_mod["gps"][0]
    gps.update(
        {
            "lat": float(truth["lat"][0]),
            "lon": float(truth["lon"][0]),
            "alt": float(truth["alt"][0]),
            "vx": float(truth["vx"][0]),
            "vy": float(truth["vy"][0]),
            "vz": float(truth["vz"][0]),
            "speed": float(truth["speed"][0]),
            "missing_flag": 0,
            "Nsat": 17,
            "DOP": 1.1,
            "RTK": "FIX",
        }
    )


def build_http_packets(rows_by_mod: Dict[str, List[Dict[str, Any]]], include_gps: bool = True):
    T = len(next(iter(rows_by_mod.values()))) if rows_by_mod else 0
    packets: List[Dict[str, Any]] = []
    for i in range(T):
        ts = None
        for m in MODALITY_ORDER:
            if m in rows_by_mod and i < len(rows_by_mod[m]):
                ts = rows_by_mod[m][i].get("timestamp")
                if ts is not None:
                    break
        pkt: Dict[str, Any] = {"timestamp": _jsonable(ts if ts is not None else float(i)), "uav_id": "UAV00001"}
        for m in MODALITY_ORDER:
            if m == "gps" and not include_gps:
                continue
            row = rows_by_mod.get(m, [])[i]
            # Keep row in raw/standard shape (quality fields included, no engineered confidence).
            pkt[m] = {k: _jsonable(v) for k, v in row.items() if k != "uav_id"}
        packets.append(pkt)
    return {"uav_id": "UAV00001", "data": packets}


def run_fusion_payload(payload: Dict[str, Any], lat0: float, lon0: float, alt0: float):
    pred_llh = fusion_router.run_fusion_http(payload)
    pred_enu = np.asarray(
        [to_enu_single_point(float(r["lat"]), float(r["lon"]), float(r["alt"]), lat0, lon0, alt0) for r in pred_llh],
        dtype=float,
    )
    return pred_llh, pred_enu


def print_modality_metrics(rows_meta: Dict[str, dict], truth_enu: np.ndarray):
    for m in POS_MODALITIES:
        obs = rows_meta[m]["obs_enu"]
        valid = rows_meta[m]["valid_mask"]
        print_track_error(f"{m.upper()}(valid)", calc_track_error_masked(obs, truth_enu, valid))

    ac_rows = rows_meta["acoustic"]["rows"]
    det = np.array([int(r.get("detected_flag", 0) or 0) for r in ac_rows], dtype=int)
    miss = np.array([int(r.get("missing_flag", 0) or 0) for r in ac_rows], dtype=int)
    snra = np.array([float(r.get("SNRa", 0.0) or 0.0) for r in ac_rows], dtype=float)
    print(
        f"[Demo][ACOUSTIC] n={len(ac_rows)} | detect_rate={det.mean():.3f} | "
        f"missing_rate={miss.mean():.3f} | SNRa_mean={snra.mean():.3f}"
    )


def print_quality_snapshot(rows_by_mod: Dict[str, List[Dict[str, Any]]], idx: int = 0):
    idx = int(max(idx, 0))
    print(f"[Demo] Packet[{idx}] standard input snapshot (with quality fields):")
    for m in MODALITY_ORDER:
        row = rows_by_mod[m][idx]
        keys = [k for k in row.keys() if k not in ("uav_id",)]
        print(f"  - {m}: keys={keys}")


def main():
    _, _, _, _, _, _, _, runtime = fusion_router._load_runtime_bundle()
    T = int(runtime.get("window_size", 20))
    seed = 42
    rng = np.random.default_rng(seed)

    cfg = gbd.default_config()
    cfg["simulation"]["duration_s"] = float(T)
    cfg["simulation"]["truth_dt_s"] = 1.0
    cfg["simulation"]["start_epoch"] = 1700000000
    # Keep demo closer to the training distribution (mostly A/B, some C/D).
    cfg["scenario_mix"] = {"A": 0.35, "B": 0.35, "C": 0.20, "D": 0.10, "E": 0.0}

    truth = simulate_truth(T=T, cfg=cfg, rng=rng)
    tags = scenario_tags_for_demo(T=T, cfg=cfg, rng=rng)
    blackout_plan = build_blackout_plan_demo(uid=1, cfg=cfg)

    rows_by_mod: Dict[str, List[Dict[str, Any]]] = {}
    rows_meta: Dict[str, dict] = {}
    for modality in MODALITY_ORDER:
        rows, obs_enu, valid_mask = generate_modality_rows(
            modality=modality,
            uid=1,
            truth=truth,
            tags=tags,
            cfg=cfg,
            rng=rng,
            blackout_intervals=blackout_plan.get(modality, []),
        )
        rows_by_mod[modality] = rows
        rows_meta[modality] = {"rows": rows, "obs_enu": obs_enu, "valid_mask": valid_mask}

    ensure_first_packet_observable(rows_by_mod, truth)

    truth_enu = np.column_stack([truth["east"], truth["north"], truth["up"]]).astype(float)
    payload_with_gps = build_http_packets(rows_by_mod, include_gps=True)
    payload_without_gps = build_http_packets(rows_by_mod, include_gps=False)

    print(
        f"[Demo] model={Path(runtime.get('router_model_path', '')).name} | "
        f"norm={Path(runtime.get('router_norm_path', '')).name} | "
        f"window={T} | seed={seed}"
    )
    print(f"[Demo] scenario tags: {''.join(tags.tolist())}")
    print_quality_snapshot(rows_by_mod, idx=0)
    print_modality_metrics(rows_meta, truth_enu)

    _, pred_with_gps = run_fusion_payload(payload_with_gps, truth["lat0"], truth["lon0"], truth["alt0"])
    _, pred_without_gps = run_fusion_payload(payload_without_gps, truth["lat0"], truth["lon0"], truth["alt0"])

    print_track_error("Fusion(with GPS)", calc_track_error(pred_with_gps, truth_enu))
    print_track_error("Fusion(no GPS)", calc_track_error(pred_without_gps, truth_enu))

    gps_obs = rows_meta["gps"]["obs_enu"]
    radar_obs = rows_meta["radar"]["obs_enu"]
    fiveg_obs = rows_meta["fiveg"]["obs_enu"]
    tdoa_obs = rows_meta["tdoa"]["obs_enu"]

    plt.figure(figsize=(10, 8))
    plt.plot(truth_enu[:, 0], truth_enu[:, 1], "k-", label="Truth", linewidth=2.0)
    plt.plot(pred_with_gps[:, 0], pred_with_gps[:, 1], "r-", label="Fusion (with GPS)", linewidth=2.0)
    plt.plot(pred_without_gps[:, 0], pred_without_gps[:, 1], "m--", label="Fusion (no GPS)", linewidth=2.0)

    for name, obs, color in [
        ("GPS", gps_obs, "tab:cyan"),
        ("Radar", radar_obs, "tab:blue"),
        ("5G-A", fiveg_obs, "tab:green"),
        ("TDOA", tdoa_obs, "tab:orange"),
    ]:
        valid = np.isfinite(obs[:, 0]) & np.isfinite(obs[:, 1])
        if np.any(valid):
            plt.scatter(obs[valid, 0], obs[valid, 1], s=18, alpha=0.35, c=color, label=name)

    plt.title("Router Fusion Demo (Standard Packet Input, With/Without GPS)")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
