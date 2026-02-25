# -*- coding: utf-8 -*-
"""
One-click end-to-end experiment for a complex mixed crossing/non-crossing dataset.

Pipeline stages (all automated):
1) Generate raw BlueSky multi-source dataset
2) transfer_confidence (processed dataset)
3) no-GPS fusion -> passive trusted tracks
4) GPS report construction + anomaly injection
5) graph-feature matching
6) association metrics evaluation (Precision/Recall/F1/MOTA)

"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict

# Allow direct execution from `drone_sys/app/core/uavMatch` (or any cwd).
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_sys.app.core.uavMatch import bluesky_fusion_match_pipeline as pipe
from drone_sys.app.core.uavMatch import evaluate_match_metrics as eval_metrics


# ============================================================
# CONFIG (all tunable params here)
# ============================================================

CFG: Dict[str, Any] = {
    # -------- output/work dirs for this one-click experiment --------
    "WORK_ROOT": "drone_sys/app/core/uavMatch/exp_bluesky_crossing500_match",
    "RAW_DATASET_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_crossing500_match/raw_dataset",
    "PROCESSED_DATASET_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_crossing500_match/processed_dataset",
    "OUTPUT_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_crossing500_match/output",

    # -------- stage control --------
    "RUN_GENERATE": True,
    "RUN_TRANSFER_CONFIDENCE": True,
    "RUN_FUSION": True,
    "RUN_MATCH": True,
    "RUN_PLOT_TRAJ_COMPARE": True,
    "RUN_EVAL": True,
    "SKIP_IF_EXISTS": True,

    # -------- dataset scale --------
    "SEED": 20260327,
    "UAV_COUNT": 500,
    "DURATION_S": 120.0,
    "TRUTH_DT_S": 1.0,
    "BATCH_SIZE": 200,
    "GENERATOR_WORKERS": 12,
    "GENERATOR_USE_MULTIPROCESSING": True,
    "TRANSFER_WORKERS": 12,

    # -------- passive sensing side (keep radar stronger) --------
    "RADAR_RATE_HZ": 3.0,
    "RADAR_ERROR_SCALE": 0.85,

    # -------- scenario injection after fusion (for matching/anomaly test) --------
    "RATIO_UNREPORTED": 0.12,
    "RATIO_DRIFT": 0.10,
    "RATIO_DEVIATION": 0.10,
    "RATIO_CONSISTENT_MIN": 0.50,
    "FALSE_REPORT_COUNT": 40,

    # -------- plotting (avoid generating too many per-uav plots by default) --------
    "TRAJ_PLOT_LIMIT": 120,  # still save first 120 single-UAV plots for inspection
    "TRAJ_TOPN_OVERLAY_N": 20,
    "TRAJ_TOPN_OVERLAY_FILENAME": "traj_compare_top20_overlay.png",

    # -------- match config (dense multi-target setting) --------
    "MATCH_CONFIG_OVERRIDES": {
        "PRINT_DETAIL": False,
        "WINDOW_DURATION_S": 3.0,
        "MATCH_ACTIVE_GAP_S": 2.5,
        "GRAPH_KNN_FUSION": 2,
        "GRAPH_KNN_REPORT": 2,
        "TOPK_PER_FUSION": 12,
        "MAX_CANDIDATE_PAIRS_FOR_AFFINITY": 6000,
        "CANDIDATE_RADIUS_M": 2500.0,
        "SIGMA_NODE_POS_M": 320.0,
        "SIGMA_NODE_VEL_MPS": 70.0,
        "SIGMA_NODE_PATH_M": 1400.0,
        "SIGMA_EDGE_DP_M": 900.0,
        "SIGMA_EDGE_DV_MPS": 60.0,
        "AFFINITY_EDGE_WEIGHT": 0.06,
        "MIN_MATCH_NODE_SCORE": 0.001,
        "MIN_MATCH_PAIR_SCORE": 0.003,
        "PAIR_SCORE_THRESHOLD": 0.18,
        "DRIFT_OFFSET_M": 180.0,
        "DEVIATION_OFFSET_M": 140.0,
        "DEVIATION_HEADING_DEG": 30.0,
        "DEVIATION_GROWTH_FRAMES": 2,
        "VOTE_WINDOW_FRAMES": 5,
        "VOTE_MIN_OBS": 3,
        "VOTE_MIN_TRUE_COUNT": 3,
        "VOTE_TRIGGER_RATIO": 0.6,
    },

    # -------- generator deep override (complex mixed crossing/non-crossing geometry) --------
    "GENERATOR_CFG_DEEP_OVERRIDE": {
        "simulation": {
            # NOTE: `uav_count` is injected from top-level `UAV_COUNT` in `_merge_pipeline_cfg()`
            "duration_s": 120.0,
            "truth_dt_s": 1.0,
            "spawn_span_deg": 0.18,
            "alt_range_m": [90.0, 135.0],
            "speed_range_mps": [26.0, 62.0],
            "segment_count": [20, 36],
            "segment_step_deg": [0.0014, 0.0048],
            "segment_alt_step_m": 12.0,
        },
        "scenario_mix": {"A": 0.40, "B": 0.30, "C": 0.16, "D": 0.09, "E": 0.05},
        "modalities": {
            "gps": {
                "rate_hz": 5.0,
                "time_jitter_s": 0.010,
                "arrival_jitter_s": 0.012,
                "reorder_window": 4,
            },
            "radar": {
                "rate_hz": 3.0,
                "time_jitter_s": 0.020,
                "arrival_jitter_s": 0.025,
                "reorder_window": 6,
            },
        },
        "modality_error_scale": {
            "gps": 0.95,
            "radar": 0.85,
            "fiveg": 1.20,
            "tdoa": 1.45,
            "acoustic": 2.30,
        },
        "missing_control": {
            "force_available_modalities": ["gps", "radar", "fiveg", "tdoa", "acoustic"],
        },
    },

    "VERBOSE": True,
}


def _merge_pipeline_cfg() -> Dict[str, Any]:
    cfg = deepcopy(CFG)
    # Single source of truth for dataset size: top-level `UAV_COUNT`.
    cfg.setdefault("GENERATOR_CFG_DEEP_OVERRIDE", {}).setdefault("simulation", {})["uav_count"] = int(cfg["UAV_COUNT"])
    return cfg


def main() -> None:
    run_cfg = _merge_pipeline_cfg()
    summary = pipe.run_pipeline(run_cfg)

    if bool(CFG.get("RUN_EVAL", True)):
        out_dir = Path(pipe._p(str(CFG["OUTPUT_DIR"])))
        eval_summary = eval_metrics.evaluate_from_output_dir(out_dir)
        m = eval_summary["metrics"]
        c = eval_summary["counts"]
        print(
            "[ASSOC-EVAL] "
            f"Precision={m['Precision']:.4f} Recall={m['Recall']:.4f} F1={m['F1Score']:.4f} MOTA={m['MOTA']:.4f} | "
            f"TP={c['tp']} FP={c['fp']} FN={c['fn']} IDSW={c['idsw']} GT={c['gt_pairs']}"
        )
        print(f"[ASSOC-EVAL] summary: {out_dir / str(eval_metrics.CFG['SAVE_SUMMARY_JSON'])}")

    print(f"[PIPELINE] output_dir: {summary['output_dir']}")
    print(f"[PIPELINE] pipeline summary: {Path(summary['output_dir']) / 'pipeline_run_summary.json'}")


if __name__ == "__main__":
    main()
