#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def discover_dataset_dirs(root_dir, batch_prefix):
    """
    Discover dataset units under a dataset root.
    A dataset unit can be:
    1) a direct folder containing batch* subfolders; or
    2) a direct folder containing truth/modality csvs.
    """
    if not os.path.isdir(root_dir):
        return []

    units = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        has_batches = len(discover_batch_dirs(p, batch_prefix)) > 0
        has_flat = os.path.exists(os.path.join(p, "truth.csv")) and any(
            os.path.exists(os.path.join(p, fn)) for fn in MODALITY_FILES.values()
        )
        if has_batches or has_flat:
            units.append((name, p))
    return units


def _task_to_batch_label(batch_prefix, idx, total):
    width = max(2, len(str(max(int(total), 1))))
    return f"{batch_prefix}{int(idx):0{width}d}"


def collect_source_units_for_merge(root_dir, batch_prefix):
    """
    Flatten all dataset units under root_dir into source units to be converted.
    Each source unit maps to one output batch.
    """
    merged_sources = []
    for ds_name, ds_path in discover_dataset_dirs(root_dir, batch_prefix):
        batch_dirs = discover_batch_dirs(ds_path, batch_prefix)
        if batch_dirs:
            for bname, bpath in batch_dirs:
                merged_sources.append(
                    {
                        "source_dataset": ds_name,
                        "source_unit": bname,
                        "input_dir": bpath,
                    }
                )
        else:
            merged_sources.append(
                {
                    "source_dataset": ds_name,
                    "source_unit": "flat",
                    "input_dir": ds_path,
                }
            )
    return merged_sources


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


def _convert_task_worker(task, cfg):
    summary = process_one_dir(
        in_dir=task["input_dir"],
        out_dir=task["output_dir"],
        cfg=cfg,
        label=task.get("label"),
    )
    return {"task": task, "summary": summary}


def _run_convert_tasks(tasks, cfg, worker_num):
    if not tasks:
        return []
    worker_num = max(1, int(worker_num))
    if worker_num == 1 or len(tasks) == 1:
        return [_convert_task_worker(t, cfg) for t in tasks]

    results = []
    max_workers = min(worker_num, len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_convert_task_worker, t, cfg): t for t in tasks}
        for fut in as_completed(fut_map):
            task = fut_map[fut]
            try:
                results.append(fut.result())
            except Exception as ex_run:
                raise RuntimeError(
                    f"Conversion failed for source={task.get('source_dataset')}:{task.get('source_unit')} "
                    f"-> target={task.get('label')}: {ex_run}"
                ) from ex_run
    results.sort(key=lambda x: int(x["task"]["task_index"]))
    return results


def process_dataset_unit(dataset_dir, output_dir, cfg, batch_prefix, worker_num=1):
    """
    Process one dataset unit.
    Supports:
    - batched dataset (batchXX subdirs)
    - flat dataset (truth.csv + modality csv)
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_dirs = discover_batch_dirs(dataset_dir, batch_prefix)

    summary = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "confidence_config": cfg["confidence"],
    }

    if batch_dirs:
        print(f"[INFO] batch mode detected in {dataset_dir}: {len(batch_dirs)} batches")
        tasks = []
        for idx, (bname, bpath) in enumerate(batch_dirs, start=1):
            b_out = os.path.join(output_dir, bname)
            print(f"[BATCH] {bname} -> {b_out}")
            tasks.append(
                {
                    "task_index": idx,
                    "source_dataset": os.path.basename(dataset_dir),
                    "source_unit": bname,
                    "input_dir": bpath,
                    "output_dir": b_out,
                    "label": bname,
                }
            )
        task_results = _run_convert_tasks(tasks, cfg, worker_num=worker_num)
        batch_summaries = [
            {
                "batch": r["task"]["label"],
                "path": r["task"]["label"],
                "source_dataset": r["task"]["source_dataset"],
                "source_unit": r["task"]["source_unit"],
                "summary": r["summary"],
            }
            for r in task_results
        ]
        summary["layout"] = "batched"
        summary["batch_count"] = len(batch_summaries)
        summary["batches"] = batch_summaries
    else:
        print(f"[INFO] flat mode in {dataset_dir}")
        fs = process_one_dir(dataset_dir, output_dir, cfg, label=None)
        summary["layout"] = "flat"
        summary["modalities"] = fs["modalities"]

    # Preserve dataset-level metadata if present.
    src_dataset_summary = os.path.join(dataset_dir, "dataset_summary.json")
    if os.path.exists(src_dataset_summary):
        dst_dataset_summary = os.path.join(output_dir, "dataset_summary.raw.json")
        shutil.copy2(src_dataset_summary, dst_dataset_summary)
        summary["raw_dataset_summary"] = {"copied": True, "path": dst_dataset_summary}

    with open(os.path.join(output_dir, "confidence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def process_dataset_root(root_in_dir, root_out_dir, cfg, batch_prefix, worker_num=8):
    """
    Merge all dataset units under root_in_dir into one large batched dataset.
    Output layout:
    - root_out_dir/batchXX/*
    """
    os.makedirs(root_out_dir, exist_ok=True)
    source_units = collect_source_units_for_merge(root_in_dir, batch_prefix)
    if not source_units:
        print(f"[WARN] no dataset units found under: {root_in_dir}")
        return {
            "input_root": root_in_dir,
            "output_root": root_out_dir,
            "dataset_count": 0,
            "source_unit_count": 0,
            "batches": [],
        }

    tasks = []
    total = len(source_units)
    for idx, src in enumerate(source_units, start=1):
        target_batch = _task_to_batch_label(batch_prefix, idx, total)
        out_dir = os.path.join(root_out_dir, target_batch)
        tasks.append(
            {
                "task_index": idx,
                "source_dataset": src["source_dataset"],
                "source_unit": src["source_unit"],
                "input_dir": src["input_dir"],
                "output_dir": out_dir,
                "label": target_batch,
            }
        )
        print(
            f"[MERGE {idx}/{total}] {src['source_dataset']}:{src['source_unit']} "
            f"-> {target_batch}"
        )

    task_results = _run_convert_tasks(tasks, cfg, worker_num=worker_num)
    merged_batches = []
    for r in task_results:
        merged_batches.append(
            {
                "batch": r["task"]["label"],
                "source_dataset": r["task"]["source_dataset"],
                "source_unit": r["task"]["source_unit"],
                "input_dir": r["task"]["input_dir"],
                "output_dir": r["task"]["output_dir"],
                "summary": r["summary"],
            }
        )

    root_summary = {
        "input_root": root_in_dir,
        "output_root": root_out_dir,
        "layout": "merged_batched",
        "confidence_config": cfg["confidence"],
        "dataset_count": len({x["source_dataset"] for x in source_units}),
        "source_unit_count": len(source_units),
        "batch_count": len(merged_batches),
        "worker_num": max(1, int(worker_num)),
        "batches": merged_batches,
    }
    with open(os.path.join(root_out_dir, "confidence_summary.merged.json"), "w", encoding="utf-8") as f:
        json.dump(root_summary, f, ensure_ascii=False, indent=2)
    return root_summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="./dataset/train-datasets")
    ap.add_argument("--output-dir", type=str, default="./dataset-processed/train-datasets")
    ap.add_argument("--batch-prefix", type=str, default="batch")
    ap.add_argument("--root-mode", action="store_true", help="force merge mode under dataset-dir")
    ap.add_argument("--worker-num", type=int, default=16)
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
    if not os.path.isdir(in_dir):
        alt_in = None
        if "train-datasets" in in_dir:
            alt_in = in_dir.replace("train-datasets", "training-datasets")
        elif "training-datasets" in in_dir:
            alt_in = in_dir.replace("training-datasets", "train-datasets")
        if alt_in and os.path.isdir(alt_in):
            print(f"[INFO] dataset-dir fallback: {in_dir} -> {alt_in}")
            in_dir = alt_in

    out_dir = args.output_dir if args.output_dir else os.path.join(in_dir, "confidence")
    if not os.path.exists(os.path.dirname(os.path.abspath(out_dir))):
        alt_out = None
        if "train-datasets" in out_dir:
            alt_out = out_dir.replace("train-datasets", "training-datasets")
        elif "training-datasets" in out_dir:
            alt_out = out_dir.replace("training-datasets", "train-datasets")
        if alt_out:
            print(f"[INFO] output-dir fallback: {out_dir} -> {alt_out}")
            out_dir = alt_out
    os.makedirs(out_dir, exist_ok=True)

    # auto root-mode:
    # - enable when explicitly requested; or
    # - current folder itself is not a dataset unit, but has dataset-unit children.
    self_has_batches = len(discover_batch_dirs(in_dir, args.batch_prefix)) > 0
    self_has_flat = os.path.exists(os.path.join(in_dir, "truth.csv")) and any(
        os.path.exists(os.path.join(in_dir, fn)) for fn in MODALITY_FILES.values()
    )
    auto_units = discover_dataset_dirs(in_dir, args.batch_prefix)
    use_root_mode = bool(args.root_mode or ((not self_has_batches) and (not self_has_flat) and len(auto_units) > 0))

    if use_root_mode:
        print(f"[INFO] root mode enabled: {in_dir}")
        root_summary = process_dataset_root(
            root_in_dir=in_dir,
            root_out_dir=out_dir,
            cfg=cfg,
            batch_prefix=args.batch_prefix,
            worker_num=max(1, int(args.worker_num)),
        )
        print(
            "[DONE] merged confidence outputs: "
            f"{out_dir} | datasets={root_summary.get('dataset_count', 0)} "
            f"| source_units={root_summary.get('source_unit_count', 0)} "
            f"| batches={root_summary.get('batch_count', 0)}"
        )
    else:
        summary = process_dataset_unit(
            dataset_dir=in_dir,
            output_dir=out_dir,
            cfg=cfg,
            batch_prefix=args.batch_prefix,
            worker_num=max(1, int(args.worker_num)),
        )
        print(f"[DONE] confidence outputs: {out_dir} | layout={summary.get('layout')}")
