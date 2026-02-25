# -*- coding: utf-8 -*-
"""
Ablation runner for the 500-UAV complex crossing/non-crossing experiment.

This script reuses the cached outputs from:
    drone_sys/app/core/uavMatch/exp_bluesky_crossing500_match

It only reruns:
    - matching (uavMatch)
    - association evaluation (Precision / Recall / F1 / MOTA)

Ablation groups:
    1) Voting ablation
    2) Candidate generation/budget ablation
    3) Graph structure ablation

"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Allow direct execution from `drone_sys/app/core/uavMatch` (or any cwd).
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_sys.app.core.uavMatch import bluesky_fusion_match_pipeline as pipe
from drone_sys.app.core.uavMatch import evaluate_match_metrics as eval_metrics
from drone_sys.app.core.uavMatch import run_crossing1000_one_click as crossing500_cfg


# ============================================================
# CONFIG (all tunable params here)
# ============================================================

_CROSSING_BASE_CFG = crossing500_cfg._merge_pipeline_cfg()

CFG: Dict[str, Any] = {
    # -------- base experiment to reuse (must already exist) --------
    "BASE_WORK_ROOT": str(_CROSSING_BASE_CFG["WORK_ROOT"]),
    "BASE_RAW_DATASET_DIR": str(_CROSSING_BASE_CFG["RAW_DATASET_DIR"]),
    "BASE_PROCESSED_DATASET_DIR": str(_CROSSING_BASE_CFG["PROCESSED_DATASET_DIR"]),
    "BASE_OUTPUT_DIR": str(_CROSSING_BASE_CFG["OUTPUT_DIR"]),

    # -------- ablation output root --------
    "ABLATION_ROOT": str(Path(str(_CROSSING_BASE_CFG["WORK_ROOT"])) / "ablation_assoc"),

    # -------- execution control --------
    "RUN_GROUP_METHOD5": True,
    "RUN_GROUP_VOTING": False,
    "RUN_GROUP_CANDIDATE": False,
    "RUN_GROUP_GRAPH": False,
    "FORCE_RERUN": True,  # Re-run by default after baseline changes
    "VERBOSE": True,

    # -------- copy cached artifacts needed for match/eval --------
    "COPY_BASE_OUTPUT_FILES": [
        "passive_tracks.csv",
        "report_tracks.csv",
        "passive_track_truth_map.csv",
        "scenario_assignment.csv",
        "fused_tracks_truth_ids.csv",
        "truth_tracks_truth_ids.csv",
    ],

    # -------- pipeline stage switches (fixed for ablation) --------
    "PIPELINE_STAGE_FLAGS": {
        "RUN_GENERATE": False,
        "RUN_TRANSFER_CONFIDENCE": False,
        "RUN_FUSION": False,
        "RUN_MATCH": True,
        "RUN_PLOT_TRAJ_COMPARE": False,
    },

    # -------- paper-style method ablation (requested 5 variants) --------
    # Definitions in this codebase:
    # - Full: current tuned full method
    # - Full: current tuned baseline after removing quality from matching score
    # - No-G: disable hard candidate gating (Top-K / total candidate budget still remain)
    # - 1st-order only: remove 2nd-order graph structure term (edge affinity weight = 0)
    # - No-T: remove temporal context features used in matching (single-frame-like + no time/path kernels)
    # - Kernel swap: replace gaussian kernels with euclid/cosine similarities
    "METHOD5_VARIANTS": [
        {
            "name": "full",
            "desc": "Full method (current baseline; quality excluded from matching score)",
            "match_patch": {},
        },
        {
            "name": "no_g",
            "desc": "No-G: disable hard candidate gating (budget retained)",
            "match_patch": {
                "DISABLE_CANDIDATE_GATING": True,
            },
        },
        {
            "name": "first_order_only",
            "desc": "1st-order only: no second-order graph structure term",
            "match_patch": {
                "AFFINITY_EDGE_WEIGHT": 0.0,
            },
        },
        {
            "name": "no_t",
            "desc": "No-T: remove temporal consistency features in matching (single-frame-like, no time/path kernels)",
            "match_patch": {
                "WINDOW_DURATION_S": 0.0,
                "MATCH_ACTIVE_GAP_S": 0.25,
                "SIGMA_NODE_TIME_S": 1e9,
                "SIGMA_NODE_PATH_M": 1e9,
            },
        },
        {
            "name": "kernel_swap",
            "desc": "Kernel swap: gaussian -> euclid/cosine similarities",
            "match_patch": {
                "KERNEL_MODE": "euclid_cosine",
            },
        },
    ],

    # -------- voting ablation (mainly impacts anomaly voting, not association pairs) --------
    "VOTING_VARIANTS": [
        {
            "name": "vote_current",
            "desc": "Current vote settings (baseline for voting ablation)",
            "match_patch": {},
        },
        {
            "name": "vote_off",
            "desc": "Disable temporal vote accumulation (instant trigger)",
            "match_patch": {
                "VOTE_WINDOW_FRAMES": 1,
                "VOTE_MIN_OBS": 1,
                "VOTE_MIN_TRUE_COUNT": 1,
                "VOTE_TRIGGER_RATIO": 1.0,
            },
        },
        {
            "name": "vote_strict",
            "desc": "Stricter vote accumulation",
            "match_patch": {
                "VOTE_WINDOW_FRAMES": 7,
                "VOTE_MIN_OBS": 4,
                "VOTE_MIN_TRUE_COUNT": 4,
                "VOTE_TRIGGER_RATIO": 0.75,
            },
        },
    ],

    # -------- candidate ablation (coverage/budget vs accuracy tradeoff) --------
    "CANDIDATE_VARIANTS": [
        {
            "name": "cand_small",
            "desc": "Small candidate budget",
            "match_patch": {
                "TOPK_PER_FUSION": 8,
                "CANDIDATE_RADIUS_M": 1800.0,
                "MAX_CANDIDATE_PAIRS_FOR_AFFINITY": 2500,
            },
        },
        {
            "name": "cand_current",
            "desc": "Current candidate budget (baseline)",
            "match_patch": {},
        },
        {
            "name": "cand_large",
            "desc": "Large candidate coverage budget",
            "match_patch": {
                "TOPK_PER_FUSION": 24,
                "CANDIDATE_RADIUS_M": 3200.0,
                "MAX_CANDIDATE_PAIRS_FOR_AFFINITY": 12000,
            },
        },
    ],

    # -------- graph-structure ablation --------
    "GRAPH_VARIANTS": [
        {
            "name": "graph_node_only",
            "desc": "Node similarity only (no edge affinity)",
            "match_patch": {
                "AFFINITY_EDGE_WEIGHT": 0.0,
            },
        },
        {
            "name": "graph_edge_light",
            "desc": "Light graph constraints",
            "match_patch": {
                "AFFINITY_EDGE_WEIGHT": 0.05,
                "GRAPH_KNN_FUSION": 2,
                "GRAPH_KNN_REPORT": 2,
            },
        },
        {
            "name": "graph_current",
            "desc": "Current graph settings (baseline)",
            "match_patch": {},
        },
        {
            "name": "graph_edge_stronger",
            "desc": "Stronger graph constraints",
            "match_patch": {
                "AFFINITY_EDGE_WEIGHT": 0.18,
                "GRAPH_KNN_FUSION": 3,
                "GRAPH_KNN_REPORT": 3,
                "SIGMA_EDGE_DP_M": 1100.0,
                "SIGMA_EDGE_DV_MPS": 70.0,
            },
        },
    ],
}


# Keep a pristine snapshot so each run starts from the same pipeline defaults.
PIPELINE_MODULE_CFG_BASE = deepcopy(pipe.CFG)


def _log(msg: str) -> None:
    if bool(CFG.get("VERBOSE", True)):
        print(msg)


def _p(path_str: str) -> Path:
    return (REPO_ROOT / Path(path_str)).resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_load(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _require_base_output() -> Tuple[Path, Path, Path]:
    base_work = _p(str(CFG["BASE_WORK_ROOT"]))
    base_raw = _p(str(CFG["BASE_RAW_DATASET_DIR"]))
    base_processed = _p(str(CFG["BASE_PROCESSED_DATASET_DIR"]))
    base_output = _p(str(CFG["BASE_OUTPUT_DIR"]))

    required = [
        base_output / "passive_tracks.csv",
        base_output / "report_tracks.csv",
        base_output / "passive_track_truth_map.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Base crossing500 cached outputs not found. Please run the 500 one-click pipeline first.\n"
            + "\n".join(missing)
        )
    _log(f"[base] work={base_work}")
    _log(f"[base] output={base_output}")
    return base_raw, base_processed, base_output


def _copy_base_output_files(base_output_dir: Path, dst_output_dir: Path) -> None:
    _ensure_dir(dst_output_dir)
    for name in list(CFG.get("COPY_BASE_OUTPUT_FILES", [])):
        src = base_output_dir / str(name)
        if not src.exists():
            continue
        dst = dst_output_dir / str(name)
        # Copy only when missing or force rerun enabled.
        if dst.exists() and not bool(CFG.get("FORCE_RERUN", False)):
            continue
        shutil.copy2(src, dst)


def _build_pipeline_run_cfg(
    *,
    base_raw: Path,
    base_processed: Path,
    variant_work_root: Path,
    variant_output_dir: Path,
    match_patch: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = crossing500_cfg._merge_pipeline_cfg()

    # Fixed to "match-only ablation on cached 500-UAV experiment outputs".
    cfg.update(
        {
            "WORK_ROOT": str(variant_work_root),
            "RAW_DATASET_DIR": str(base_raw),
            "PROCESSED_DATASET_DIR": str(base_processed),
            "OUTPUT_DIR": str(variant_output_dir),
        }
    )
    cfg.update(dict(CFG.get("PIPELINE_STAGE_FLAGS", {})))

    # Merge match overrides on top of the crossing500 baseline.
    mo = deepcopy(cfg.get("MATCH_CONFIG_OVERRIDES", {}))
    mo.update(match_patch or {})
    cfg["MATCH_CONFIG_OVERRIDES"] = mo

    return cfg


def _extract_match_diag(output_dir: Path) -> Dict[str, Any]:
    match_summary = _json_load(output_dir / "match_summary.json", default={}) or {}
    frame_stats_path = output_dir / "match_frame_stats.csv"
    frame_stats: Optional[pd.DataFrame] = None
    if frame_stats_path.exists():
        try:
            frame_stats = pd.read_csv(frame_stats_path)
        except Exception:
            frame_stats = None

    voted_counts = match_summary.get("voted_anomaly_counts_union", {}) or {}
    event_counts = match_summary.get("event_type_counts", {}) or {}
    diag: Dict[str, Any] = {
        "match_frames": int(match_summary.get("frames", 0) or 0),
        "total_match_pairs_over_frames": int(match_summary.get("total_match_pairs_over_frames", 0) or 0),
        "last_global_score": float(match_summary.get("last_global_score", 0.0) or 0.0),
        "last_global_anomaly": bool(match_summary.get("last_global_anomaly", False)),
        "voted_unreported_union": int(voted_counts.get("unreported", 0) or 0),
        "voted_false_report_union": int(voted_counts.get("false_report", 0) or 0),
        "voted_duplicate_union": int(voted_counts.get("duplicate_report", 0) or 0),
        "voted_drift_union": int(voted_counts.get("drift", 0) or 0),
        "voted_deviation_union": int(voted_counts.get("deviation", 0) or 0),
        "event_count_total": int(sum(int(v) for v in event_counts.values())) if isinstance(event_counts, dict) else 0,
    }
    if frame_stats is not None and (not frame_stats.empty):
        ga = frame_stats["global_anomaly"].astype(bool)
        diag["global_anomaly_frame_ratio"] = float(ga.mean())
        diag["avg_matches_per_frame"] = float(frame_stats["match_count"].mean())
        diag["avg_global_score"] = float(frame_stats["global_score"].mean())
    else:
        diag["global_anomaly_frame_ratio"] = 0.0
        diag["avg_matches_per_frame"] = 0.0
        diag["avg_global_score"] = 0.0
    return diag


def _run_single_variant(
    *,
    group_name: str,
    variant_name: str,
    variant_desc: str,
    match_patch: Dict[str, Any],
    base_raw: Path,
    base_processed: Path,
    base_output: Path,
) -> Dict[str, Any]:
    ablation_root = _ensure_dir(_p(str(CFG["ABLATION_ROOT"])))
    variant_root = _ensure_dir(ablation_root / group_name / variant_name)
    variant_output = _ensure_dir(variant_root / "output")

    eval_summary_path = variant_output / str(eval_metrics.CFG["SAVE_SUMMARY_JSON"])
    cfg_snapshot_path = variant_root / "variant_config.json"

    _copy_base_output_files(base_output, variant_output)

    if eval_summary_path.exists() and (not bool(CFG.get("FORCE_RERUN", False))):
        _log(f"[reuse] {group_name}/{variant_name} -> {eval_summary_path}")
        eval_summary = _json_load(eval_summary_path)
        match_diag = _extract_match_diag(variant_output)
        return _build_result_record(
            group_name=group_name,
            variant_name=variant_name,
            variant_desc=variant_desc,
            match_patch=match_patch,
            variant_root=variant_root,
            variant_output=variant_output,
            eval_summary=eval_summary,
            match_diag=match_diag,
            runtime_s=None,
            reused=True,
        )

    run_cfg = _build_pipeline_run_cfg(
        base_raw=base_raw,
        base_processed=base_processed,
        variant_work_root=variant_root,
        variant_output_dir=variant_output,
        match_patch=match_patch,
    )
    _json_dump(
        {
            "group": group_name,
            "variant": variant_name,
            "desc": variant_desc,
            "match_patch": match_patch,
            "pipeline_run_cfg": run_cfg,
        },
        cfg_snapshot_path,
    )

    # Reset pipeline global config before every variant to avoid cross-run contamination.
    pipe.CFG = deepcopy(PIPELINE_MODULE_CFG_BASE)

    t0 = time.perf_counter()
    _log(f"[run] {group_name}/{variant_name} ...")
    pipe.run_pipeline(run_cfg)
    eval_summary = eval_metrics.evaluate_from_output_dir(variant_output)
    dt = time.perf_counter() - t0
    match_diag = _extract_match_diag(variant_output)

    return _build_result_record(
        group_name=group_name,
        variant_name=variant_name,
        variant_desc=variant_desc,
        match_patch=match_patch,
        variant_root=variant_root,
        variant_output=variant_output,
        eval_summary=eval_summary,
        match_diag=match_diag,
        runtime_s=dt,
        reused=False,
    )


def _build_result_record(
    *,
    group_name: str,
    variant_name: str,
    variant_desc: str,
    match_patch: Dict[str, Any],
    variant_root: Path,
    variant_output: Path,
    eval_summary: Dict[str, Any],
    match_diag: Dict[str, Any],
    runtime_s: Optional[float],
    reused: bool,
) -> Dict[str, Any]:
    metrics = dict((eval_summary or {}).get("metrics", {}) or {})
    counts = dict((eval_summary or {}).get("counts", {}) or {})
    rec: Dict[str, Any] = {
        "group": group_name,
        "variant": variant_name,
        "desc": variant_desc,
        "Precision": float(metrics.get("Precision", 0.0) or 0.0),
        "Recall": float(metrics.get("Recall", 0.0) or 0.0),
        "F1Score": float(metrics.get("F1Score", 0.0) or 0.0),
        "MOTA": float(metrics.get("MOTA", 0.0) or 0.0),
        "TP": int(counts.get("tp", 0) or 0),
        "FP": int(counts.get("fp", 0) or 0),
        "FN": int(counts.get("fn", 0) or 0),
        "IDSW": int(counts.get("idsw", 0) or 0),
        "GT": int(counts.get("gt_pairs", 0) or 0),
        "runtime_s": None if runtime_s is None else float(runtime_s),
        "reused": bool(reused),
        "variant_root": str(variant_root),
        "variant_output": str(variant_output),
        "match_patch_json": json.dumps(match_patch, ensure_ascii=False, sort_keys=True),
    }
    rec.update(match_diag or {})
    return rec


def _run_group(
    *,
    group_name: str,
    variants: List[Dict[str, Any]],
    base_raw: Path,
    base_processed: Path,
    base_output: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for v in variants:
        rows.append(
            _run_single_variant(
                group_name=group_name,
                variant_name=str(v["name"]),
                variant_desc=str(v.get("desc", "")),
                match_patch=dict(v.get("match_patch", {}) or {}),
                base_raw=base_raw,
                base_processed=base_processed,
                base_output=base_output,
            )
        )
        r = rows[-1]
        _log(
            f"[{group_name}] {r['variant']}: "
            f"P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1Score']:.4f} MOTA={r['MOTA']:.4f} "
            f"| TP={r['TP']} FP={r['FP']} FN={r['FN']} IDSW={r['IDSW']}"
        )
    return rows


def run_voting_ablation() -> List[Dict[str, Any]]:
    base_raw, base_processed, base_output = _require_base_output()
    return _run_group(
        group_name="voting",
        variants=list(CFG.get("VOTING_VARIANTS", [])),
        base_raw=base_raw,
        base_processed=base_processed,
        base_output=base_output,
    )


def run_method5_ablation() -> List[Dict[str, Any]]:
    base_raw, base_processed, base_output = _require_base_output()
    return _run_group(
        group_name="method5",
        variants=list(CFG.get("METHOD5_VARIANTS", [])),
        base_raw=base_raw,
        base_processed=base_processed,
        base_output=base_output,
    )


def run_candidate_ablation() -> List[Dict[str, Any]]:
    base_raw, base_processed, base_output = _require_base_output()
    return _run_group(
        group_name="candidate",
        variants=list(CFG.get("CANDIDATE_VARIANTS", [])),
        base_raw=base_raw,
        base_processed=base_processed,
        base_output=base_output,
    )


def run_graph_ablation() -> List[Dict[str, Any]]:
    base_raw, base_processed, base_output = _require_base_output()
    return _run_group(
        group_name="graph",
        variants=list(CFG.get("GRAPH_VARIANTS", [])),
        base_raw=base_raw,
        base_processed=base_processed,
        base_output=base_output,
    )


def _save_group_and_global_summaries(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ablation_root = _ensure_dir(_p(str(CFG["ABLATION_ROOT"])))
    df = pd.DataFrame(all_rows)
    if df.empty:
        summary = {"ablation_root": str(ablation_root), "groups": {}, "rows": []}
        _json_dump(summary, ablation_root / "ablation_results_summary.json")
        return summary

    sort_cols = ["group", "F1Score", "MOTA", "Precision", "Recall"]
    asc = [True, False, False, False, False]
    df_sorted = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    df_sorted.to_csv(ablation_root / "ablation_results_all.csv", index=False)

    groups_summary: Dict[str, Any] = {}
    for group_name, gdf in df.groupby("group", sort=False):
        gdf2 = gdf.sort_values(["F1Score", "MOTA"], ascending=[False, False]).reset_index(drop=True)
        gdir = _ensure_dir(ablation_root / str(group_name))
        gdf2.to_csv(gdir / f"{group_name}_ablation_results.csv", index=False)
        best = gdf2.iloc[0].to_dict() if not gdf2.empty else {}
        groups_summary[str(group_name)] = {
            "num_variants": int(len(gdf2)),
            "best_by_f1": {
                "variant": str(best.get("variant", "")),
                "Precision": float(best.get("Precision", 0.0) or 0.0),
                "Recall": float(best.get("Recall", 0.0) or 0.0),
                "F1Score": float(best.get("F1Score", 0.0) or 0.0),
                "MOTA": float(best.get("MOTA", 0.0) or 0.0),
                "IDSW": int(best.get("IDSW", 0) or 0),
            },
        }

    summary = {
        "ablation_root": str(ablation_root),
        "base_output_dir": str(_p(str(CFG["BASE_OUTPUT_DIR"]))),
        "groups": groups_summary,
        "rows": df_sorted.to_dict(orient="records"),
        "notes": [
            "method5/No-T is implemented as removing temporal context features in matching (single-frame-like + no time/path kernels).",
            "method5/No-G disables hard candidate gating while retaining Top-K and total candidate budget for tractability.",
            "Voting ablation primarily affects anomaly voting outputs; association metrics may remain unchanged.",
            "Association metrics are computed from exact pair matches in evaluate_match_metrics.py",
        ],
    }
    _json_dump(summary, ablation_root / "ablation_results_summary.json")
    return summary


def run_all_ablations() -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []

    if bool(CFG.get("RUN_GROUP_METHOD5", True)):
        all_rows.extend(run_method5_ablation())
    if bool(CFG.get("RUN_GROUP_VOTING", True)):
        all_rows.extend(run_voting_ablation())
    if bool(CFG.get("RUN_GROUP_CANDIDATE", True)):
        all_rows.extend(run_candidate_ablation())
    if bool(CFG.get("RUN_GROUP_GRAPH", True)):
        all_rows.extend(run_graph_ablation())

    return _save_group_and_global_summaries(all_rows)


def main() -> None:
    summary = run_all_ablations()
    ablation_root = Path(summary["ablation_root"])
    print(f"[ABLATION] done -> {ablation_root}")
    print(f"[ABLATION] summary -> {ablation_root / 'ablation_results_summary.json'}")
    print(f"[ABLATION] table   -> {ablation_root / 'ablation_results_all.csv'}")


if __name__ == "__main__":
    main()
