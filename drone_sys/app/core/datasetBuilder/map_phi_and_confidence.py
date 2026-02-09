#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd


MODALITY_FILES = {
    "gps": "gps.csv",
    "radar": "radar.csv",
    "fiveg": "5g_a.csv",
    "tdoa": "tdoa.csv",
    "acoustic": "acoustic.csv",
}

BASE_COLS = [
    "timestamp",
    "uav_id",
    "lat",
    "lon",
    "alt",
    "vx",
    "vy",
    "vz",
    "speed",
    "scenario_tag",
    "missing_flag",
    "arrival_time",
]


def default_cfg():
    return {
        "confidence": {
            "mode": "linear",  # linear or sigmoid
            "alpha": 0.65,
            "sigmoid_k": 8.0,
            "sigmoid_b": 0.5,
        },
        "modalities": {
            "gps": {
                "alpha": 0.70,
                "metrics": {
                    "Nsat": {"op": "pos", "l": 4.0, "u": 20.0, "group": "rt", "weight": 0.60},
                    "DOP": {"op": "neg", "l": 0.5, "u": 8.0, "group": "rt", "weight": 0.40},
                    "RTK": {"op": "cat", "mapping": {"FIX": 1.0, "FLOAT": 0.6, "NONE": 0.2}, "group": "st", "weight": 1.0},
                },
            },
            "radar": {
                "alpha": 0.65,
                "metrics": {
                    "E": {"op": "pos", "l": 0.0, "u": 1.0, "group": "rt", "weight": 1.0},
                    "Ptrk": {"op": "pos", "l": 0.0, "u": 1.0, "group": "st", "weight": 1.0},
                },
            },
            "fiveg": {
                "alpha": 0.60,
                "metrics": {
                    "SNR": {"op": "pos", "l": 0.0, "u": 30.0, "group": "rt", "weight": 0.55},
                    "RSSI": {"op": "pos", "l": -110.0, "u": -50.0, "group": "rt", "weight": 0.45},
                    "d": {"op": "neg", "l": 5.0, "u": 300.0, "group": "st", "weight": 0.45},
                    "ploss": {"op": "neg", "l": 0.0, "u": 0.5, "group": "st", "weight": 0.55},
                },
            },
            "tdoa": {
                "alpha": 0.62,
                "metrics": {
                    "e": {"op": "neg", "l": 0.0, "u": 50.0, "group": "rt", "weight": 1.0},
                    "eps_sync": {"op": "neg", "l": 0.0, "u": 200.0, "group": "st", "weight": 1.0},
                },
            },
            "acoustic": {
                "alpha": 0.58,
                "metrics": {
                    "SNRa": {"op": "pos", "l": 0.0, "u": 30.0, "group": "rt", "weight": 1.0},
                    "n": {"op": "neg", "l": 0.0, "u": 1.0, "group": "st", "weight": 1.0},
                },
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


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def phi_pos(x, l, u):
    x = pd.to_numeric(x, errors="coerce")
    return clip01((x - l) / max(u - l, 1e-12))


def phi_neg(x, l, u):
    x = pd.to_numeric(x, errors="coerce")
    return clip01((u - x) / max(u - l, 1e-12))


def phi_cat(x, mapping):
    return x.astype(str).map(mapping).fillna(0.0).clip(0.0, 1.0)


def weighted_mean(df, items):
    if not items:
        return pd.Series(np.nan, index=df.index, dtype=float)
    sw = float(sum(w for _, w in items))
    if sw <= 0:
        sw = 1.0
    acc = np.zeros(len(df), dtype=float)
    for c, w in items:
        acc += df[c].fillna(0.0).to_numpy(dtype=float) * float(w)
    return pd.Series(np.clip(acc / sw, 0.0, 1.0), index=df.index)


def confidence(rt, st, alpha, mode, k, b):
    raw = (alpha * rt + (1.0 - alpha) * st).fillna(0.0).clip(0.0, 1.0)
    if mode == "sigmoid":
        val = 1.0 / (1.0 + np.exp(-k * (raw - b)))
        return pd.Series(np.clip(val, 0.0, 1.0), index=raw.index)
    return raw


def process_modality(modality, in_path, out_path, cfg):
    mc = cfg["modalities"][modality]
    gc = cfg["confidence"]
    df = pd.read_csv(in_path)
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return {"rows": 0}

    phi_cols = []
    rt_items = []
    st_items = []
    quality_cols = []

    for metric, spec in mc["metrics"].items():
        if metric not in df.columns:
            raise ValueError(f"[{modality}] missing quality column: {metric}")
        quality_cols.append(metric)
        pc = f"phi_{metric}"
        op = spec["op"]
        if op == "pos":
            df[pc] = phi_pos(df[metric], spec["l"], spec["u"])
        elif op == "neg":
            df[pc] = phi_neg(df[metric], spec["l"], spec["u"])
        elif op == "cat":
            df[pc] = phi_cat(df[metric], spec["mapping"])
        else:
            raise ValueError(f"[{modality}] unsupported op: {op}")
        phi_cols.append(pc)
        if spec["group"] == "rt":
            rt_items.append((pc, float(spec.get("weight", 1.0))))
        elif spec["group"] == "st":
            st_items.append((pc, float(spec.get("weight", 1.0))))
        else:
            raise ValueError(f"[{modality}] unsupported group: {spec['group']}")

    df["rt_m"] = weighted_mean(df, rt_items)
    df["st_m"] = weighted_mean(df, st_items)

    if df["rt_m"].isna().all() and not df["st_m"].isna().all():
        df["rt_m"] = df["st_m"]
    if df["st_m"].isna().all() and not df["rt_m"].isna().all():
        df["st_m"] = df["rt_m"]

    alpha = float(mc.get("alpha", gc["alpha"]))
    df["confidence"] = confidence(
        df["rt_m"],
        df["st_m"],
        alpha=alpha,
        mode=gc["mode"],
        k=float(gc["sigmoid_k"]),
        b=float(gc["sigmoid_b"]),
    )

    passthrough_cols = [
        c
        for c in df.columns
        if c not in quality_cols and not c.startswith("phi_") and c not in ("rt_m", "st_m", "confidence")
    ]
    keep_base = [c for c in BASE_COLS if c in passthrough_cols]
    keep_extra = [c for c in passthrough_cols if c not in keep_base]
    out_cols = keep_base + keep_extra + phi_cols + ["rt_m", "st_m", "confidence"]
    out_df = df[out_cols].copy()
    out_df.to_csv(out_path, index=False)

    by_tag = {}
    if "scenario_tag" in out_df.columns:
        g = out_df.groupby("scenario_tag")["confidence"].mean()
        by_tag = {str(k): float(v) for k, v in g.items()}

    return {
        "rows": int(len(out_df)),
        "dropped_quality_columns": quality_cols,
        "added_columns": phi_cols + ["rt_m", "st_m", "confidence"],
        "confidence_mean": float(out_df["confidence"].mean()),
        "confidence_std": float(out_df["confidence"].std(ddof=0)),
        "scenario_confidence_mean": by_tag,
    }


def discover_batch_dirs(root_dir, batch_prefix):
    if not os.path.isdir(root_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        if not name.startswith(batch_prefix):
            continue
        if any(os.path.exists(os.path.join(p, fn)) for fn in MODALITY_FILES.values()):
            dirs.append((name, p))
    return dirs


def process_one_dir(in_dir, out_dir, cfg, label=None):
    os.makedirs(out_dir, exist_ok=True)
    info = {"input_dir": in_dir, "output_dir": out_dir, "label": label, "truth": {"copied": False}, "modalities": {}}
    src_truth = os.path.join(in_dir, "truth.csv")
    dst_truth = os.path.join(out_dir, "truth.csv")
    if os.path.exists(src_truth):
        shutil.copy2(src_truth, dst_truth)
        info["truth"] = {"copied": True, "source": src_truth, "output": dst_truth}
        print(f"[COPY] truth: {src_truth} -> {dst_truth}")
    else:
        print(f"[WARN] truth not found: {src_truth}")
    for m, fn in MODALITY_FILES.items():
        src = os.path.join(in_dir, fn)
        dst = os.path.join(out_dir, fn)
        if not os.path.exists(src):
            print(f"[WARN] skip {m}: {src} not found")
            continue
        tag = f"{label}:{m}" if label else m
        print(f"[RUN] {tag}: {src} -> {dst}")
        info["modalities"][m] = process_modality(m, src, dst, cfg)
    with open(os.path.join(out_dir, "confidence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="./dataset/scenario_multi_source_5000x120")
    ap.add_argument("--output-dir", type=str, default="./dataset-processed/scenario_multi_source_5000x120")
    ap.add_argument("--batch-prefix", type=str, default="batch")
    ap.add_argument("--config-json", type=str, default=None)
    ap.add_argument("--mode", choices=["linear", "sigmoid"], default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = default_cfg()
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            deep_update(cfg, json.load(f))
    if args.mode is not None:
        cfg["confidence"]["mode"] = args.mode
    if args.alpha is not None:
        cfg["confidence"]["alpha"] = float(args.alpha)
    if args.k is not None:
        cfg["confidence"]["sigmoid_k"] = float(args.k)
    if args.b is not None:
        cfg["confidence"]["sigmoid_b"] = float(args.b)

    in_dir = args.dataset_dir
    out_dir = args.output_dir if args.output_dir else os.path.join(in_dir, "confidence")
    os.makedirs(out_dir, exist_ok=True)

    batch_dirs = discover_batch_dirs(in_dir, args.batch_prefix)
    summary = {
        "input_dir": in_dir,
        "output_dir": out_dir,
        "confidence_config": cfg["confidence"],
    }

    if batch_dirs:
        print(f"[INFO] batch mode detected: {len(batch_dirs)} batches")
        batch_summaries = []
        for bname, bpath in batch_dirs:
            b_out = os.path.join(out_dir, bname)
            print(f"[BATCH] {bname}")
            bsum = process_one_dir(bpath, b_out, cfg, label=bname)
            batch_summaries.append({"batch": bname, "path": bname, "summary": bsum})
        summary["layout"] = "batched"
        summary["batch_count"] = len(batch_summaries)
        summary["batches"] = batch_summaries
    else:
        print("[INFO] flat mode (no batch dirs found)")
        fs = process_one_dir(in_dir, out_dir, cfg, label=None)
        summary["layout"] = "flat"
        summary["modalities"] = fs["modalities"]

    with open(os.path.join(out_dir, "confidence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] confidence outputs: {out_dir}")
