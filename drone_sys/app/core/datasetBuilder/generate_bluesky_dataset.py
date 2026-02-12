#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-source heterogeneous UAV observation dataset using BlueSky.

Outputs:
1) truth.csv
2) gps.csv, radar.csv, 5g_a.csv, tdoa.csv, acoustic.csv
3) dataset_summary.json
"""

import json
import math
import os
import time
import atexit
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd


MODALITY_FILES = {
    "gps": "gps.csv",
    "radar": "radar.csv",
    "fiveg": "5g_a.csv",
    "tdoa": "tdoa.csv",
    "acoustic": "acoustic.csv",
}

SCENARIOS = ["A", "B", "C", "D", "E"]
DESIRED_PRECISION_ORDER = ["gps", "radar", "fiveg", "tdoa", "acoustic"]
DEFAULT_DATASET_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_config.json")
_BS_INITIALIZED = False
bs = None


def _cleanup_bluesky():
    try:
        if bs is not None and getattr(bs, "sim", None) is not None:
            bs.sim.reset()
    except Exception:
        pass


atexit.register(_cleanup_bluesky)


def ensure_bluesky_runtime():
    global bs
    global _BS_INITIALIZED
    if bs is None:
        import bluesky as bs_mod
        bs = bs_mod
    if not _BS_INITIALIZED:
        bs.init(guimode=False)
        _BS_INITIALIZED = True
    else:
        try:
            if getattr(bs, "sim", None) is not None:
                bs.sim.reset()
        except Exception:
            # Fallback: re-init once if previous runtime got into bad state.
            bs.init(guimode=False)
            _BS_INITIALIZED = True


def default_config():
    return {
        "seed": 20260209,
        "output_dir": "./dataset/scenario_multi_source_5000x120",
        "worker_num": 8,
        "use_multiprocessing": True,
        "batching": {
            "enabled": True,
            "batch_size": 200,
            "folder_prefix": "batch",
        },
        "simulation": {
            "uav_count": 5000,
            "duration_s": 120.0,
            "truth_dt_s": 1.0,
            "start_epoch": int(time.time()),
            "base_lat": 45.0,
            "base_lon": 5.0,
            "spawn_span_deg": 0.35,
            "alt_range_m": [80.0, 160.0],
            "speed_range_mps": [30.0, 55.0],
            "segment_count": [10, 22],
            "segment_step_deg": [0.0008, 0.0028],
            "segment_alt_step_m": 25.0,
        },
        "scenario_mix": {"A": 0.40, "B": 0.30, "C": 0.15, "D": 0.10, "E": 0.05}, # （good/normal/bad/jitter/degraded）
        "scenario_duration_s": {
            "A": [8.0, 24.0],
            "B": [6.0, 18.0],
            "C": [4.0, 14.0],
            "D": [3.0, 8.0],
            "E": [5.0, 20.0],
        },
        "modalities": {
            "gps": {"rate_hz": 5.0, "time_jitter_s": 0.015, "arrival_jitter_s": 0.020, "reorder_window": 6},
            "radar": {"rate_hz": 2.0, "time_jitter_s": 0.030, "arrival_jitter_s": 0.035, "reorder_window": 8},
            "fiveg": {"rate_hz": 10.0, "time_jitter_s": 0.010, "arrival_jitter_s": 0.060, "reorder_window": 12},
            "tdoa": {"rate_hz": 1.0, "time_jitter_s": 0.040, "arrival_jitter_s": 0.040, "reorder_window": 6},
            "acoustic": {"rate_hz": 1.0, "time_jitter_s": 0.040, "arrival_jitter_s": 0.050, "reorder_window": 6},
        },
        "quality_bounds": {
            "gps": {"Nsat": [4, 20], "DOP": [0.5, 8.0], "RTK": ["FIX", "FLOAT", "NONE"]},
            "radar": {"E": [0.0, 1.0], "Ptrk": [0.0, 1.0]},
            "fiveg": {"SNR": [0.0, 30.0], "RSSI": [-110.0, -50.0], "d": [5.0, 300.0], "ploss": [0.0, 0.5]},
            "tdoa": {"eps_sync": [0.0, 200.0], "e": [0.0, 50.0]},
            "acoustic": {"SNRa": [0.0, 30.0], "n": [0.0, 1.0]},
        },
        "environment": {
            "anchors": {
                "gps": [45.000, 5.000],
                "radar": [45.020, 5.010],
                "fiveg": [44.990, 5.030],
                "tdoa": [45.040, 4.980],
                "acoustic": [44.970, 5.000],
            },
            "max_range_m": {"gps": 60000.0, "radar": 45000.0, "fiveg": 22000.0, "tdoa": 35000.0, "acoustic": 18000.0},
            "obstacle_zones": [
                {"lat": 45.060, "lon": 5.060, "radius_m": 12000.0, "strength": 0.60},
                {"lat": 44.930, "lon": 5.020, "radius_m": 9000.0, "strength": 0.45},
                {"lat": 45.010, "lon": 4.940, "radius_m": 10000.0, "strength": 0.50},
            ],
            "interference_events": [
                {"start_s": 20.0, "end_s": 35.0, "modalities": ["fiveg", "tdoa"], "level": 0.60},
                {"start_s": 55.0, "end_s": 68.0, "modalities": ["gps", "radar"], "level": 0.40},
                {"start_s": 85.0, "end_s": 105.0, "modalities": ["acoustic", "fiveg"], "level": 0.70},
            ],
        },
        "drift": {"enabled": True, "start_prob": {"C": 0.06, "E": 0.14}, "duration_s": [5.0, 20.0], "bias_mps": 0.35, "rw_sigma_m": 0.9, "decay": 0.93},
        # Missing-control hook for dataset ablation / stress tests.
        # Missing probability is transformed as:
        #   p' = clamp(p * global_scale * scenario_scale[tag] * modality_scale[modality]
        #              + global_bias + modality_bias[modality], 0, 1)
        # and then optional hard overrides are applied:
        #   force_missing_modalities / force_available_modalities.
        "missing_control": {
            "global_scale": 1.0,
            "global_bias": 0.0,
            "scenario_scale": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0},
            "modality_scale": {"gps": 1.0, "radar": 1.0, "fiveg": 1.0, "tdoa": 1.0, "acoustic": 1.0},
            "modality_bias": {"gps": 0.0, "radar": 0.0, "fiveg": 0.0, "tdoa": 0.0, "acoustic": 0.0},
            "force_missing_modalities": [],
            "force_available_modalities": [],
            "random_blackout": {
                "enabled": False,
                "event_count": [0, 0],
                "modality_count": [2, 5],
                "duration_s": [5.0, 10.0],
                "modalities": ["gps", "radar", "fiveg", "tdoa", "acoustic"],
            },
        },
    }


def deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def meters_to_deg_lat(m):
    return m / 111000.0


def meters_to_deg_lon(m, lat_deg):
    c = max(math.cos(math.radians(lat_deg)), 1e-6)
    return m / (111000.0 * c)


def geodist_m(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    x = dlon * math.cos((lat1 + lat2) * 0.5)
    y = dlat
    return 6371000.0 * math.sqrt(x * x + y * y)


def weighted_pick(mix, rng):
    ks = list(mix.keys())
    ps = np.array([mix[k] for k in ks], dtype=float)
    ps = ps / ps.sum()
    return rng.choice(ks, p=ps)


def scenario_series(times_rel, mix, dur_cfg, rng):
    out = np.empty(times_rel.shape[0], dtype="<U1")
    cur = weighted_pick(mix, rng)
    sw = times_rel[0] if times_rel.size else 0.0
    for i, t in enumerate(times_rel):
        if t >= sw:
            cur = weighted_pick(mix, rng)
            lo, hi = dur_cfg[cur]
            sw = t + rng.uniform(lo, hi)
        out[i] = cur
    return out


def extract_vel():
    gs = np.asarray(bs.traf.gs, dtype=float).copy()
    trk = np.asarray(bs.traf.trk, dtype=float).copy()
    vs = np.asarray(bs.traf.vs, dtype=float).copy()
    vx = gs * np.cos(np.deg2rad(trk))
    vy = gs * np.sin(np.deg2rad(trk))
    return vx, vy, vs


def run_bluesky_truth(cfg, rng):
    sim_cfg = cfg["simulation"]
    ensure_bluesky_runtime()
    sim = bs.sim

    n = int(sim_cfg["uav_count"])
    base_lat = sim_cfg["base_lat"]
    base_lon = sim_cfg["base_lon"]
    span = sim_cfg["spawn_span_deg"]

    for i in range(n):
        uid = f"UAV{i + 1:05d}"
        lat = base_lat + rng.uniform(-span, span)
        lon = base_lon + rng.uniform(-span, span)
        alt = rng.uniform(sim_cfg["alt_range_m"][0], sim_cfg["alt_range_m"][1])
        hdg = rng.uniform(0.0, 360.0)
        spd = rng.uniform(sim_cfg["speed_range_mps"][0], sim_cfg["speed_range_mps"][1])
        bs.traf.cre(uid, "UAV", lat, lon, hdg, alt, spd)

        segn = int(rng.integers(sim_cfg["segment_count"][0], sim_cfg["segment_count"][1] + 1))
        wp_lat, wp_lon, wp_alt = lat, lon, alt
        for _ in range(segn):
            hdg = (hdg + rng.uniform(-120.0, 120.0)) % 360.0
            d = rng.uniform(sim_cfg["segment_step_deg"][0], sim_cfg["segment_step_deg"][1])
            wp_lat += d * math.cos(math.radians(hdg))
            wp_lon += d * math.sin(math.radians(hdg))
            wp_alt += rng.uniform(-sim_cfg["segment_alt_step_m"], sim_cfg["segment_alt_step_m"])
            bs.stack.stack(f"ADDWPT {uid} N{wp_lat:.5f} E{wp_lon:.5f} {wp_alt:.1f}")

    dt = float(sim_cfg["truth_dt_s"])
    duration = float(sim_cfg["duration_s"])
    steps = int(round(duration / dt)) + 1
    t0 = float(sim_cfg["start_epoch"])

    rec = []
    for s in range(steps):
        if s > 0:
            sim.step(dt)
            bs.traf.update()
        ts = t0 + s * dt
        vx, vy, vz = extract_vel()
        for i in range(int(bs.traf.ntraf)):
            sp = float(math.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2))
            rec.append({
                "timestamp": ts,
                "uav_id": str(bs.traf.id[i]),
                "lat": float(bs.traf.lat[i]),
                "lon": float(bs.traf.lon[i]),
                "alt": float(bs.traf.alt[i]),
                "vx": float(vx[i]),
                "vy": float(vy[i]),
                "vz": float(vz[i]),
                "speed": sp,
            })
    df = pd.DataFrame(rec).sort_values(["timestamp", "uav_id"]).reset_index(drop=True)
    return df


def truth_cache(truth_df):
    out = {}
    for uid, g in truth_df.groupby("uav_id", sort=False):
        gg = g.sort_values("timestamp")
        out[uid] = {c: gg[c].to_numpy(dtype=float) for c in ["timestamp", "lat", "lon", "alt", "vx", "vy", "vz", "speed"]}
    return out


def interp_track(tk, ts):
    tt = tk["timestamp"]
    return {k: np.interp(ts, tt, tk[k]) for k in ["lat", "lon", "alt", "vx", "vy", "vz", "speed"]}


def env_penalty(modality, lat, lon, t_abs, cfg):
    env = cfg["environment"]
    sim_cfg = cfg["simulation"]
    a_lat, a_lon = env["anchors"][modality]
    d = geodist_m(lat, lon, a_lat, a_lon)
    pen_d = clamp(d / float(env["max_range_m"][modality]), 0.0, 1.0)

    pen_o = 0.0
    for z in env["obstacle_zones"]:
        if geodist_m(lat, lon, z["lat"], z["lon"]) <= z["radius_m"]:
            pen_o = max(pen_o, float(z["strength"]))

    t_rel = float(t_abs - sim_cfg["start_epoch"])
    pen_i = 0.0
    for ev in env["interference_events"]:
        if modality in ev["modalities"] and ev["start_s"] <= t_rel <= ev["end_s"]:
            pen_i = max(pen_i, float(ev["level"]))
    return clamp(0.55 * pen_d + 0.30 * pen_o + 0.30 * pen_i, 0.0, 1.0)


def update_drift(state, tag, dt, cfg, rng):
    dc = cfg["drift"]
    if not dc["enabled"]:
        return state["vec"]
    if tag in ("C", "E") and state["remain"] <= 0 and rng.random() < dc["start_prob"][tag]:
        state["remain"] = rng.uniform(dc["duration_s"][0], dc["duration_s"][1])
        state["bias"] = rng.normal(0.0, dc["bias_mps"], size=3)
    if state["remain"] > 0:
        rw = rng.normal(0.0, dc["rw_sigma_m"] * math.sqrt(max(dt, 1e-3)), size=3)
        state["vec"] = state["vec"] + state["bias"] * dt + rw
        state["remain"] -= dt
    else:
        state["vec"] = state["vec"] * float(dc["decay"])
    return state["vec"]


def apply_error_scale(modality, pos_s, vel_s, cfg):
    default_scales = {"gps": 1.00, "radar": 1.22, "fiveg": 1.36, "tdoa": 1.72, "acoustic": 2.80}
    scale = float(cfg.get("modality_error_scale", {}).get(modality, default_scales.get(modality, 1.0)))
    return pos_s * scale, vel_s * scale


def acoustic_detect_prob(tag, snra, noise_strength, env_pen, cfg):
    ac = cfg.get("acoustic_detection", {})
    base_map = ac.get("base_detect_prob", {"A": 0.98, "B": 0.90, "C": 0.62, "D": 0.78, "E": 0.45})
    base = float(base_map.get(tag, 0.6))
    p = (
        base
        - float(ac.get("distance_weight", 0.22)) * env_pen
        + float(ac.get("snr_weight", 0.12)) * ((snra - 10.0) / 20.0)
        - float(ac.get("noise_weight", 0.18)) * noise_strength
    )
    return clamp(p, 0.01, 0.999)


def apply_missing_control(modality, tag, miss_p, cfg):
    mc = cfg.get("missing_control", {}) if isinstance(cfg, dict) else {}
    force_missing = set(mc.get("force_missing_modalities", []))
    force_available = set(mc.get("force_available_modalities", []))

    if modality in force_missing:
        return 1.0, True, False
    if modality in force_available:
        return 0.0, False, True

    g_scale = float(mc.get("global_scale", 1.0))
    g_bias = float(mc.get("global_bias", 0.0))
    s_scale = float(mc.get("scenario_scale", {}).get(tag, 1.0))
    m_scale = float(mc.get("modality_scale", {}).get(modality, 1.0))
    m_bias = float(mc.get("modality_bias", {}).get(modality, 0.0))
    p = float(miss_p) * g_scale * s_scale * m_scale + g_bias + m_bias
    return clamp(p, 0.0, 1.0), False, False


def stable_u32(*parts):
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def resolve_worker_num(cfg):
    wn = int(cfg.get("worker_num", 8))
    return max(1, wn)


def _modality_uid_seed(cfg, batch_label, modality, uid):
    seed_base = int(cfg.get("seed", 0))
    return stable_u32(seed_base, batch_label, modality, uid)


def _modality_worker(payload):
    modality = payload["modality"]
    uid = payload["uid"]
    tk = payload["tk"]
    cfg = payload["cfg"]
    seed = int(payload["seed"])
    rng = np.random.default_rng(seed)
    df = generate_modality_for_uav(modality, uid, tk, cfg, rng)
    return uid, df


def iter_modality_results(modality, uids, cache, cfg, batch_label, log_prefix=""):
    worker_num = resolve_worker_num(cfg)
    use_mp = bool(cfg.get("use_multiprocessing", True)) and worker_num > 1
    payloads = [
        {
            "modality": modality,
            "uid": uid,
            "tk": cache[uid],
            "cfg": cfg,
            "seed": _modality_uid_seed(cfg, batch_label, modality, uid),
        }
        for uid in uids
    ]

    if not use_mp:
        for p in payloads:
            yield _modality_worker(p)
        return

    mp_ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_num, mp_context=mp_ctx) as ex:
        fut_map = {ex.submit(_modality_worker, p): p["uid"] for p in payloads}
        done = 0
        for fut in as_completed(fut_map):
            uid = fut_map[fut]
            try:
                out_uid, df = fut.result()
            except Exception as ex_err:
                raise RuntimeError(f"worker failed: modality={modality}, uid={uid}, err={ex_err}") from ex_err
            done += 1
            if done % 50 == 0 or done == len(fut_map):
                if log_prefix:
                    print(f"    [{log_prefix}:{modality}] workers done {done}/{len(fut_map)}")
                else:
                    print(f"    [{modality}] workers done {done}/{len(fut_map)}")
            yield out_uid, df


def _merge_intervals(intervals):
    if not intervals:
        return []
    arr = sorted([(float(a), float(b)) for a, b in intervals], key=lambda x: x[0])
    out = [list(arr[0])]
    for s, e in arr[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def build_random_blackout_plan(uid, cfg):
    mc = cfg.get("missing_control", {}) if isinstance(cfg, dict) else {}
    rb = mc.get("random_blackout", {}) if isinstance(mc, dict) else {}
    if not bool(rb.get("enabled", False)):
        return {}

    sim = cfg.get("simulation", {})
    duration_total = float(sim.get("duration_s", 0.0))
    if duration_total <= 0.0:
        return {}

    event_cfg = rb.get("event_count", [0, 0])
    if isinstance(event_cfg, (int, float)):
        event_lo = event_hi = int(event_cfg)
    else:
        event_lo = int(event_cfg[0]) if len(event_cfg) > 0 else 0
        event_hi = int(event_cfg[1]) if len(event_cfg) > 1 else event_lo
    event_lo = max(0, event_lo)
    event_hi = max(event_lo, event_hi)
    if event_hi == 0:
        return {}

    mod_pool = [m for m in rb.get("modalities", list(MODALITY_FILES.keys())) if m in MODALITY_FILES]
    if len(mod_pool) == 0:
        return {}

    mod_count_cfg = rb.get("modality_count", [2, 5])
    if isinstance(mod_count_cfg, (int, float)):
        mod_lo = mod_hi = int(mod_count_cfg)
    else:
        mod_lo = int(mod_count_cfg[0]) if len(mod_count_cfg) > 0 else 2
        mod_hi = int(mod_count_cfg[1]) if len(mod_count_cfg) > 1 else mod_lo
    mod_lo = max(1, min(mod_lo, len(mod_pool)))
    mod_hi = max(mod_lo, min(mod_hi, len(mod_pool)))

    dur_cfg = rb.get("duration_s", [5.0, 10.0])
    if isinstance(dur_cfg, (int, float)):
        dur_lo = dur_hi = float(dur_cfg)
    else:
        dur_lo = float(dur_cfg[0]) if len(dur_cfg) > 0 else 5.0
        dur_hi = float(dur_cfg[1]) if len(dur_cfg) > 1 else dur_lo
    dur_lo = max(0.1, dur_lo)
    dur_hi = max(dur_lo, dur_hi)

    seed = stable_u32(cfg.get("seed", 0), "blackout", uid)
    rng = np.random.default_rng(seed)
    n_events = int(rng.integers(event_lo, event_hi + 1)) if event_hi > event_lo else event_lo

    plan = {m: [] for m in mod_pool}
    for _ in range(n_events):
        n_mod = int(rng.integers(mod_lo, mod_hi + 1)) if mod_hi > mod_lo else mod_lo
        mods = list(rng.choice(mod_pool, size=n_mod, replace=False))
        dur = float(rng.uniform(dur_lo, dur_hi))
        start_max = max(duration_total - dur, 0.0)
        start = float(rng.uniform(0.0, start_max)) if start_max > 0 else 0.0
        end = start + dur
        for m in mods:
            plan[m].append((start, end))

    return {m: _merge_intervals(v) for m, v in plan.items() if len(v) > 0}


def in_blackout(t_rel, intervals):
    for s, e in intervals:
        if s <= t_rel <= e:
            return True
    return False


def sample_quality(modality, tag, pen, st, cfg, rng):
    qb = cfg["quality_bounds"][modality]
    jit = 1.0

    if modality == "gps":
        nsat = int(round(rng.normal({"A": 16, "B": 12, "C": 7, "D": 12, "E": 8}[tag] - 4 * pen, {"A": 2, "B": 2.5, "C": 2, "D": 4, "E": 3}[tag])))
        nsat = int(np.clip(nsat, qb["Nsat"][0], qb["Nsat"][1]))
        dop = clamp(rng.normal({"A": 1.0, "B": 2.0, "C": 5.2, "D": 2.2, "E": 6.0}[tag] + 1.8 * pen, {"A": 0.25, "B": 0.6, "C": 1.0, "D": 1.5, "E": 1.2}[tag]), qb["DOP"][0], qb["DOP"][1])
        if tag == "E" and st["fault"] <= 0 and rng.random() < 0.12:
            st["fault"] = int(rng.integers(6, 22))
        if st["fault"] > 0:
            st["fault"] -= 1
            nsat = min(nsat, 7)
            dop = max(dop, rng.uniform(5.5, qb["DOP"][1]))
            rtk = "NONE"
        else:
            rtk = rng.choice(["FIX", "FLOAT", "NONE"], p={"A": [0.85, 0.13, 0.02], "B": [0.60, 0.30, 0.10], "C": [0.20, 0.35, 0.45], "D": [0.50, 0.25, 0.25], "E": [0.10, 0.20, 0.70]}[tag])
        if tag == "D":
            jit = rng.uniform(1.5, 2.8)
        pos_s = (0.8 + 0.9 * (20 - nsat) + 1.4 * dop + 10 * pen) * jit
        vel_s = (0.10 + 0.12 * dop + 0.03 * (20 - nsat)) * jit
        delay = max(5.0, rng.normal(25.0, 6.0) + (12.0 if tag == "D" else 0.0))
        miss = 0.002 + (0.03 if tag in ("C", "E") else 0.0)
        pos_s, vel_s = apply_error_scale(modality, pos_s, vel_s, cfg)
        return {"Nsat": nsat, "DOP": float(dop), "RTK": rtk}, pos_s, vel_s, delay, miss

    if modality == "radar":
        bp = {"A": ((12, 2), (10, 2)), "B": ((7, 4), (6, 4)), "C": ((2, 7), (2, 8)), "D": ((5, 4), (5, 5)), "E": ((2, 8), (1.5, 10))}[tag]
        e = clamp(rng.beta(bp[0][0], bp[0][1]) - 0.35 * pen, qb["E"][0], qb["E"][1])
        p = clamp(rng.beta(bp[1][0], bp[1][1]) - 0.45 * pen, qb["Ptrk"][0], qb["Ptrk"][1])
        if tag == "E" and st["fault"] <= 0 and rng.random() < 0.16:
            st["fault"] = int(rng.integers(3, 12))
        if st["fault"] > 0:
            st["fault"] -= 1
            p = 0.0
            e = min(e, 0.2)
        if tag == "D":
            jit = rng.uniform(1.8, 3.0)
        pos_s = (4 + (1 - p) * 45 + (1 - e) * 20 + 12 * pen) * jit
        vel_s = (0.25 + (1 - p) + 0.3 * (1 - e)) * jit
        delay = max(8.0, rng.normal(60, 20) + (25 if tag == "D" else 0))
        miss = 0.01 + (0.15 if p < 0.15 else 0.0)
        pos_s, vel_s = apply_error_scale(modality, pos_s, vel_s, cfg)
        return {"E": float(e), "Ptrk": float(p)}, pos_s, vel_s, delay, miss

    if modality == "fiveg":
        snr = clamp(rng.normal({"A": 24, "B": 18, "C": 9, "D": 17, "E": 7}[tag] - 10 * pen, {"A": 2, "B": 3, "C": 4, "D": 7, "E": 5}[tag]), qb["SNR"][0], qb["SNR"][1])
        rssi = clamp(rng.normal({"A": -60, "B": -74, "C": -95, "D": -78, "E": -98}[tag] - 15 * pen, {"A": 3, "B": 4, "C": 6, "D": 8, "E": 7}[tag]), qb["RSSI"][0], qb["RSSI"][1])
        dms = clamp(abs(rng.normal({"A": 15, "B": 45, "C": 160, "D": 70, "E": 180}[tag] + 80 * pen, {"A": 8, "B": 18, "C": 50, "D": 80, "E": 70}[tag])), qb["d"][0], qb["d"][1])
        plp = {"A": (1.2, 30.0), "B": (2.0, 14.0), "C": (6.0, 6.0), "D": (3.5, 10.0), "E": (7.0, 4.0)}[tag]
        pl = clamp(0.5 * rng.beta(plp[0], plp[1]) + 0.15 * pen, qb["ploss"][0], qb["ploss"][1])
        if tag == "E" and st["fault"] <= 0 and rng.random() < 0.18:
            st["fault"] = int(rng.integers(5, 20))
        if st["fault"] > 0:
            st["fault"] -= 1
            pl = max(pl, rng.uniform(0.25, 0.5))
            dms = max(dms, rng.uniform(120, 300))
            snr = min(snr, rng.uniform(0, 8))
            rssi = min(rssi, rng.uniform(-110, -95))
        if tag == "D":
            dms = clamp(dms + abs(rng.normal(0, 60)), qb["d"][0], qb["d"][1])
            pl = clamp(pl + rng.uniform(0, 0.10), qb["ploss"][0], qb["ploss"][1])
            jit = rng.uniform(1.8, 3.2)
        pos_s = (7 + (30 - snr) * 1.2 + dms / 20 + pl * 100 + 10 * pen) * jit
        vel_s = (0.25 + (30 - snr) / 40 + pl * 2 + dms / 500) * jit
        pos_s, vel_s = apply_error_scale(modality, pos_s, vel_s, cfg)
        return {"SNR": float(snr), "RSSI": float(rssi), "d": float(dms), "ploss": float(pl)}, pos_s, vel_s, dms, pl

    if modality == "tdoa":
        eps = clamp(rng.normal({"A": 20, "B": 50, "C": 120, "D": 70, "E": 150}[tag] + 45 * pen, {"A": 8, "B": 15, "C": 25, "D": 40, "E": 25}[tag]), qb["eps_sync"][0], qb["eps_sync"][1])
        err = clamp(rng.normal({"A": 3, "B": 10, "C": 28, "D": 15, "E": 35}[tag] + 10 * pen, {"A": 1.5, "B": 3.5, "C": 7, "D": 10, "E": 6}[tag]), qb["e"][0], qb["e"][1])
        if tag == "E" and st["fault"] <= 0 and rng.random() < 0.20:
            st["fault"] = int(rng.integers(4, 16))
        if st["fault"] > 0:
            st["fault"] -= 1
            eps = max(eps, rng.uniform(150, 200))
            err = max(err, rng.uniform(25, 50))
        if tag == "D":
            jit = rng.uniform(1.8, 3.0)
        pos_s = (10 + err * 1.6 + eps / 5 + 8 * pen) * jit
        vel_s = (0.60 + err * 0.03 + eps / 130) * jit
        delay = max(8.0, rng.normal(40 + eps * 0.03, 15))
        miss = 0.01 + (0.04 if tag in ("C", "E") else 0.0)
        pos_s, vel_s = apply_error_scale(modality, pos_s, vel_s, cfg)
        return {"eps_sync": float(eps), "e": float(err)}, pos_s, vel_s, delay, miss

    if modality == "acoustic":
        snra = clamp(rng.normal({"A": 24, "B": 17, "C": 8, "D": 14, "E": 6}[tag] - 10 * pen, {"A": 2, "B": 3, "C": 4, "D": 6, "E": 5}[tag]), qb["SNRa"][0], qb["SNRa"][1])
        n = clamp(rng.normal({"A": 0.12, "B": 0.32, "C": 0.65, "D": 0.45, "E": 0.75}[tag] + 0.20 * pen, {"A": 0.05, "B": 0.10, "C": 0.12, "D": 0.18, "E": 0.10}[tag]), qb["n"][0], qb["n"][1])
        if tag == "E" and st["fault"] <= 0 and rng.random() < 0.22:
            st["fault"] = int(rng.integers(6, 24))
        if st["fault"] > 0:
            st["fault"] -= 1
            st["trend"] = min(1.0, st["trend"] + rng.uniform(0.03, 0.08))
            n = max(n, st["trend"])
            snra = min(snra, 10.0 - 5.0 * st["trend"])
        else:
            st["trend"] *= 0.95
        if tag == "D":
            jit = rng.uniform(1.7, 3.1)
        pos_s = (12 + (30 - snra) * 1.4 + n * 30 + 8 * pen) * jit
        vel_s = (0.80 + n * 1.1 + (30 - snra) / 45) * jit
        delay = max(8.0, rng.normal(80, 25))
        miss = 0.01 + (0.03 if tag in ("C", "E") else 0.0)
        pos_s, vel_s = apply_error_scale(modality, pos_s, vel_s, cfg)
        return {"SNRa": float(snra), "n": float(n)}, pos_s, vel_s, delay, miss

    raise ValueError(f"unsupported modality {modality}")


def event_times(duration_s, rate_hz, jitter_s, rng):
    if rate_hz <= 0:
        return np.array([], dtype=float)
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / rate_hz, dtype=float)
    if jitter_s > 0:
        t += rng.normal(0.0, jitter_s, size=t.shape[0])
    return np.sort(np.clip(t, 0.0, duration_s))


def generate_modality_for_uav(modality, uid, tk, cfg, rng):
    sim = cfg["simulation"]
    mc = cfg["modalities"][modality]
    t_rel = event_times(sim["duration_s"], mc["rate_hz"], mc["time_jitter_s"], rng)
    if t_rel.size == 0:
        return pd.DataFrame()
    t_abs = sim["start_epoch"] + t_rel
    ti = interp_track(tk, t_abs)

    tag = scenario_series(t_rel, cfg["scenario_mix"], cfg["scenario_duration_s"], rng)
    drift_st = {"vec": np.zeros(3), "bias": np.zeros(3), "remain": 0.0}
    mod_st = {"fault": 0, "trend": 0.0}
    blackout_plan = build_random_blackout_plan(uid, cfg)
    blackout_intervals = blackout_plan.get(modality, [])
    prev = t_rel[0]
    rows = []

    for i in range(t_rel.size):
        tt = float(t_rel[i])
        ta = float(t_abs[i])
        tg = str(tag[i])
        pen = env_penalty(modality, float(ti["lat"][i]), float(ti["lon"][i]), ta, cfg)
        q, pos_s, vel_s, delay_ms, miss_p = sample_quality(modality, tg, pen, mod_st, cfg, rng)
        miss_p_adj, force_missing, force_available = apply_missing_control(modality, tg, miss_p, cfg)
        blackout_now = int(in_blackout(tt, blackout_intervals))
        force_missing = bool(force_missing or blackout_now == 1)

        dt = max(tt - prev, 1.0 / max(mc["rate_hz"], 1e-6))
        drift = update_drift(drift_st, tg, dt, cfg, rng)
        prev = tt

        lat0, lon0, alt0 = float(ti["lat"][i]), float(ti["lon"][i]), float(ti["alt"][i])
        vx0, vy0, vz0 = float(ti["vx"][i]), float(ti["vy"][i]), float(ti["vz"][i])
        arrival = ta + max(delay_ms, 0.0) / 1000.0 + rng.normal(0.0, mc["arrival_jitter_s"])

        if modality == "acoustic":
            if force_missing:
                detected = 0
                miss = 1
            else:
                p_det = acoustic_detect_prob(tg, float(q["SNRa"]), float(q["n"]), pen, cfg)
                detected = int(rng.random() < p_det)
                if force_available:
                    detected = 1
                    miss = 0
                else:
                    miss = int((detected == 0) or (rng.random() < clamp(miss_p_adj, 0.0, 1.0)))
            if detected == 1 and miss == 0:
                spl_lo, spl_hi = cfg.get("acoustic_detection", {}).get("spl_range_db", [35.0, 110.0])
                spl = clamp(
                    40.0 + 1.35 * float(q["SNRa"]) - 8.0 * float(q["n"]) - 7.0 * pen + rng.normal(0.0, 2.2),
                    spl_lo,
                    spl_hi,
                )
                energy = clamp((10.0 ** (spl / 20.0)) / 1e5, 0.0, 1.0)
            else:
                spl = np.nan
                energy = np.nan
                q["SNRa"] = 0.0
                q["n"] = 1.0
            row = {
                "timestamp": ta,
                "uav_id": uid,
                "detected_flag": detected,
                "spl_db": spl,
                "acoustic_energy": energy,
                "scenario_tag": tg,
                "missing_flag": miss,
                "blackout_flag": blackout_now,
                "arrival_time": arrival,
            }
        else:
            ne = rng.normal(0.0, pos_s)
            nn = rng.normal(0.0, pos_s)
            nu = rng.normal(0.0, 0.6 * pos_s)
            lat = lat0 + meters_to_deg_lat(nn + drift[1])
            lon = lon0 + meters_to_deg_lon(ne + drift[0], lat0)
            alt = alt0 + nu + drift[2]

            vx = vx0 + rng.normal(0.0, vel_s)
            vy = vy0 + rng.normal(0.0, vel_s)
            vz = vz0 + rng.normal(0.0, 0.8 * vel_s)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)

            if force_missing:
                miss = 1
            elif force_available:
                miss = 0
            else:
                miss = int(rng.random() < clamp(miss_p_adj, 0.0, 1.0))
            if miss == 1:
                lat = lon = alt = vx = vy = vz = speed = np.nan

            row = {
                "timestamp": ta,
                "uav_id": uid,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": speed,
                "scenario_tag": tg,
                "missing_flag": miss,
                "blackout_flag": blackout_now,
                "arrival_time": arrival,
            }
        row.update(q)
        rows.append(row)

    df = pd.DataFrame(rows)
    win = int(mc.get("reorder_window", 1))
    if win > 1 and len(df) > 1:
        idx = np.arange(len(df))
        for s in range(0, len(idx), win):
            blk = idx[s : s + win].copy()
            if blk.size > 1 and rng.random() < 0.75:
                rng.shuffle(blk)
                idx[s : s + blk.size] = blk
        df = df.iloc[idx].reset_index(drop=True)
    return df


class NumStat:
    def __init__(self):
        self.s = {}

    def update(self, df):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
            st = self.s.setdefault(c, {"n": 0, "sum": 0.0, "sum2": 0.0, "min": float("inf"), "max": float("-inf")})
            st["n"] += int(a.size)
            st["sum"] += float(a.sum())
            st["sum2"] += float(np.square(a).sum())
            st["min"] = min(st["min"], float(a.min()))
            st["max"] = max(st["max"], float(a.max()))

    def as_dict(self):
        out = {}
        for c, st in self.s.items():
            if st["n"] == 0:
                continue
            m = st["sum"] / st["n"]
            v = max(st["sum2"] / st["n"] - m * m, 0.0)
            out[c] = {"count": st["n"], "min": st["min"], "max": st["max"], "mean": m, "std": math.sqrt(v)}
        return out


def chunked(items, size):
    if size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def truth_numeric_stats(truth):
    trstat = {}
    for c in truth.columns:
        if pd.api.types.is_numeric_dtype(truth[c]):
            s = pd.to_numeric(truth[c], errors="coerce").dropna()
            trstat[c] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "std": float(s.std(ddof=0))}
    return trstat


def precision_audit(cfg, n_samples=3000):
    means = {}
    for i, modality in enumerate(DESIRED_PRECISION_ORDER):
        rng = np.random.default_rng(int(cfg["seed"]) + 2000 + i)
        state = {"fault": 0, "trend": 0.0}
        pos_samples = []
        for _ in range(int(n_samples)):
            _, pos_s, _, _, _ = sample_quality(modality, "A", 0.0, state, cfg, rng)
            pos_samples.append(pos_s)
        means[modality] = float(np.mean(pos_samples))

    order_pass = True
    for i in range(len(DESIRED_PRECISION_ORDER) - 1):
        left = DESIRED_PRECISION_ORDER[i]
        right = DESIRED_PRECISION_ORDER[i + 1]
        if not (means[left] < means[right]):
            order_pass = False
            break

    return {
        "good_scenario_mean_pos_sigma_m": means,
        "desired_order": DESIRED_PRECISION_ORDER,
        "order_pass": order_pass,
    }


def write_batch(batch_dir, batch_uids, cache, cfg, batch_label):
    os.makedirs(batch_dir, exist_ok=True)
    truth_cols = ["timestamp", "uav_id", "lat", "lon", "alt", "vx", "vy", "vz", "speed"]
    truth_rows = []
    for uid in batch_uids:
        tk = cache[uid]
        n = tk["timestamp"].shape[0]
        for i in range(n):
            truth_rows.append({
                "timestamp": float(tk["timestamp"][i]),
                "uav_id": uid,
                "lat": float(tk["lat"][i]),
                "lon": float(tk["lon"][i]),
                "alt": float(tk["alt"][i]),
                "vx": float(tk["vx"][i]),
                "vy": float(tk["vy"][i]),
                "vz": float(tk["vz"][i]),
                "speed": float(tk["speed"][i]),
            })
    truth_df = pd.DataFrame(truth_rows, columns=truth_cols).sort_values(["timestamp", "uav_id"]).reset_index(drop=True)
    truth_df.to_csv(os.path.join(batch_dir, "truth.csv"), index=False)

    mod_sum = {}
    for m, fn in MODALITY_FILES.items():
        p = os.path.join(batch_dir, fn)
        if os.path.exists(p):
            os.remove(p)
        wrote = False
        cnt = Counter()
        stat = NumStat()
        rows_total = 0

        for i, (uid, df) in enumerate(
            iter_modality_results(
                modality=m,
                uids=batch_uids,
                cache=cache,
                cfg=cfg,
                batch_label=batch_label,
                log_prefix=batch_label,
            ),
            start=1,
        ):
            if df.empty:
                continue
            df.to_csv(p, mode="a", header=not wrote, index=False)
            wrote = True
            rows_total += int(len(df))
            if "scenario_tag" in df.columns:
                cnt.update(df["scenario_tag"].astype(str).tolist())
            stat.update(df)
            if i % 200 == 0:
                print(f"    [{batch_label}:{m}] {i}/{len(batch_uids)}")
        mod_sum[m] = {"rows": rows_total, "scenario_counts": dict(cnt), "numeric_stats": stat.as_dict(), "path": fn}

    with open(os.path.join(batch_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch": batch_label,
                "uav_count": len(batch_uids),
                "uav_ids": [batch_uids[0], batch_uids[-1]] if batch_uids else [],
                "truth_rows": int(len(truth_df)),
                "truth_path": "truth.csv",
                "truth_numeric_stats": truth_numeric_stats(truth_df),
                "modalities": mod_sum,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"truth_rows": int(len(truth_df)), "modalities": mod_sum}


def run(cfg):
    np.random.seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    print("[1/4] BlueSky truth simulation")
    truth = run_bluesky_truth(cfg, rng)
    print("[2/4] Build truth cache")
    cache = truth_cache(truth)
    uids = sorted(cache.keys())
    batch_cfg = cfg.get("batching", {})
    batch_enabled = bool(batch_cfg.get("enabled", False))
    batch_size = int(batch_cfg.get("batch_size", len(uids)))
    prefix = str(batch_cfg.get("folder_prefix", "batch"))

    print("[3/4] Generate modalities")
    summary = {"seed": int(cfg["seed"]), "config": cfg}

    if batch_enabled:
        batches = list(chunked(uids, batch_size))
        batch_list = []
        for bi, buids in enumerate(batches, start=1):
            label = f"{prefix}{bi:02d}"
            bdir = os.path.join(out, label)
            print(f"  [{label}] UAVs={len(buids)}")
            bsum = write_batch(bdir, buids, cache, cfg, label)
            batch_list.append(
                {
                    "batch": label,
                    "path": label,
                    "uav_count": len(buids),
                    "uav_range": [buids[0], buids[-1]] if buids else [],
                    **bsum,
                }
            )
        summary["files"] = {
            "layout": "batched",
            "batch_count": len(batch_list),
            "batches": batch_list,
        }
    else:
        truth.to_csv(os.path.join(out, "truth.csv"), index=False)
        mod_sum = {}
        label = "root"
        for m, fn in MODALITY_FILES.items():
            p = os.path.join(out, fn)
            if os.path.exists(p):
                os.remove(p)
            wrote = False
            cnt = Counter()
            stat = NumStat()
            rows_total = 0

            for i, (uid, df) in enumerate(
                iter_modality_results(
                    modality=m,
                    uids=uids,
                    cache=cache,
                    cfg=cfg,
                    batch_label=label,
                    log_prefix=label,
                ),
                start=1,
            ):
                if df.empty:
                    continue
                df.to_csv(p, mode="a", header=not wrote, index=False)
                wrote = True
                rows_total += int(len(df))
                if "scenario_tag" in df.columns:
                    cnt.update(df["scenario_tag"].astype(str).tolist())
                stat.update(df)
                if i % 200 == 0:
                    print(f"  [{m}] {i}/{len(uids)}")
            mod_sum[m] = {"rows": rows_total, "scenario_counts": dict(cnt), "numeric_stats": stat.as_dict(), "path": fn}

        summary["files"] = {
            "layout": "flat",
            "truth": {"path": "truth.csv", "rows": int(len(truth)), "numeric_stats": truth_numeric_stats(truth)},
            "modalities": mod_sum,
        }

    print("[4/4] Write dataset_summary.json")
    summary["precision_audit"] = precision_audit(cfg, n_samples=3000)
    if not summary["precision_audit"]["order_pass"]:
        print("[WARN] precision order check failed, please tune cfg['modality_error_scale'].")
    with open(os.path.join(out, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {out}")


def run_profile_batch(batch_cfg_path):
    with open(batch_cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("profile json must be an object")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) == 0:
        raise ValueError("profile json must contain non-empty `profiles` list")

    base_override = payload.get("base_override", {})
    root_output_dir = payload.get("root_output_dir")

    for i, prof in enumerate(profiles, start=1):
        if not isinstance(prof, dict):
            raise ValueError(f"profile #{i} must be an object")
        name = str(prof.get("name", f"profile_{i:02d}"))
        override = prof.get("override", {})

        cfg = default_config()
        if isinstance(base_override, dict) and len(base_override) > 0:
            deep_update(cfg, base_override)
        if isinstance(override, dict) and len(override) > 0:
            deep_update(cfg, override)

        if root_output_dir is not None and "output_dir" not in override:
            cfg["output_dir"] = os.path.join(str(root_output_dir), name)

        print(f"\n[PROFILE {i}/{len(profiles)}] {name}")
        print(f"  output_dir: {cfg['output_dir']}")
        run(cfg)


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_DATASET_CONFIG_PATH):
        raise FileNotFoundError(
            f"dataset config not found: {DEFAULT_DATASET_CONFIG_PATH}\n"
            "create this file and define your dataset profiles."
        )
    run_profile_batch(DEFAULT_DATASET_CONFIG_PATH)
