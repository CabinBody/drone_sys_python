# -*- coding: utf-8 -*-
"""
Evaluate uavMatch association outputs on the synthetic BlueSky pipeline dataset.

Metrics (association-level):
- Precision / Recall / F1Score
- MOTA (MOT Accuracy, using frame-wise GT pair instances and a simple ID switch count)

Ground truth source:
- `passive_track_truth_map.csv` gives passive track -> true_uav_id
- `passive_tracks.csv` gives passive-track presence by frame
- `report_tracks.csv` gives reported-track presence by frame (including fake reports)

Predictions source:
- `match_results.json` flattened as (time, fusion_id, report_id)

Notes:
- This evaluates "association correctness" (which report track is matched to which passive track).
- It does NOT directly evaluate anomaly labels.
- MOTA here uses a pragmatic IDSW definition: for the same GT passive object, if predicted report_id changes
  between two consecutive frames where that object is predicted, count one ID switch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIG (all tunable params here)
# ============================================================

CFG: Dict[str, Any] = {
    # ---- default experiment output dir ----
    "OUTPUT_DIR": "drone_sys/app/core/uavMatch/exp_bluesky_crossing1000_match/output",

    # ---- input file names under OUTPUT_DIR ----
    "MATCH_RESULTS_JSON": "match_results.json",
    "PASSIVE_TRACKS_CSV": "passive_tracks.csv",
    "PASSIVE_TRACK_TRUTH_MAP_CSV": "passive_track_truth_map.csv",
    "REPORT_TRACKS_CSV": "report_tracks.csv",
    "SCENARIO_ASSIGNMENT_CSV": "scenario_assignment.csv",  # optional, only for breakdowns

    # ---- output file names under OUTPUT_DIR ----
    "SAVE_SUMMARY_JSON": "match_assoc_eval_summary.json",
    "SAVE_FRAME_CSV": "match_assoc_eval_frame_stats.csv",
    "SAVE_PRED_PAIRS_CSV": "match_assoc_eval_pred_pairs.csv",
    "SAVE_GT_PAIRS_CSV": "match_assoc_eval_gt_pairs.csv",
    "SAVE_MATCHED_PAIRS_CSV": "match_assoc_eval_tp_pairs.csv",
    "SAVE_OBJECT_TIMELINE_CSV": "match_assoc_eval_object_timeline.csv",

    # ---- evaluation behavior ----
    "TIME_ROUND_DECIMALS": 6,
    "IDSW_RESET_ON_MISS": False,  # if True, missing prediction resets previous assigned id for IDSW counting
    "CLIP_MOTA_TO_0_1": False,

    "VERBOSE": True,
}


# ============================================================
# Helpers
# ============================================================


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _p(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (_repo_root() / p)


def _log(msg: str) -> None:
    if bool(CFG.get("VERBOSE", True)):
        print(msg)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, ensure_ascii=False, indent=2)


def _safe_div(a: float, b: float) -> float:
    b = float(b)
    if abs(b) <= 1e-12:
        return 0.0
    return float(a) / b


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _norm_time_series(s: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round(int(decimals))


def _first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _flatten_match_results_to_pred_pairs(match_results: List[Dict[str, Any]], time_decimals: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fr in match_results:
        t = round(float(fr.get("time", 0.0)), int(time_decimals))
        for m in (fr.get("matches", []) or []):
            rows.append(
                {
                    "time": t,
                    "fusion_id": str(m.get("fusion_id", "")),
                    "report_id": str(m.get("report_id", "")),
                    "node_score": float(m.get("node_score", np.nan)) if m.get("node_score") is not None else np.nan,
                    "edge_score": float(m.get("edge_score", np.nan)) if m.get("edge_score") is not None else np.nan,
                    "pair_score": float(m.get("pair_score", np.nan)) if m.get("pair_score") is not None else np.nan,
                    "spectral_score": float(m.get("spectral_score", np.nan)) if m.get("spectral_score") is not None else np.nan,
                    "offset_m": float(m.get("offset_m", np.nan)) if m.get("offset_m") is not None else np.nan,
                    "vel_offset_mps": float(m.get("vel_offset_mps", np.nan)) if m.get("vel_offset_mps") is not None else np.nan,
                    "heading_diff_deg": float(m.get("heading_diff_deg", np.nan)) if m.get("heading_diff_deg") is not None else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "fusion_id",
                "report_id",
                "node_score",
                "edge_score",
                "pair_score",
                "spectral_score",
                "offset_m",
                "vel_offset_mps",
                "heading_diff_deg",
            ]
        )
    df = pd.DataFrame(rows)
    # Defensive de-duplication: keep the highest pair_score if duplicates exist.
    sort_cols = [c for c in ["pair_score", "node_score", "edge_score", "spectral_score"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["time", "fusion_id", "report_id"], keep="first")
    return df.sort_values(["time", "fusion_id", "report_id"]).reset_index(drop=True)


def _build_gt_pairs(
    passive_tracks: pd.DataFrame,
    passive_map: pd.DataFrame,
    report_tracks: pd.DataFrame,
    scenario_assignment: pd.DataFrame,
    time_decimals: int,
) -> pd.DataFrame:
    if passive_tracks.empty or passive_map.empty:
        return pd.DataFrame(columns=["time", "fusion_id", "report_id", "true_uav_id", "scenario_case"])

    p = passive_tracks.copy()
    p_id_col = _first_existing(p, ["id", "track_id"])
    p_time_col = _first_existing(p, ["time", "timestamp", "ts", "t"])
    if p_id_col is None or p_time_col is None:
        raise ValueError("passive_tracks.csv must contain id/time columns.")
    p["fusion_id"] = p[p_id_col].astype(str)
    p["time"] = _norm_time_series(p[p_time_col], time_decimals)
    p = p[np.isfinite(p["time"])].copy()
    p = p[["time", "fusion_id"]].drop_duplicates()

    m = passive_map.copy()
    map_pid_col = _first_existing(m, ["passive_track_id", "fusion_id", "id"])
    map_uid_col = _first_existing(m, ["true_uav_id", "uav_id"])
    if map_pid_col is None or map_uid_col is None:
        raise ValueError("passive_track_truth_map.csv must contain passive_track_id and true_uav_id.")
    m["fusion_id"] = m[map_pid_col].astype(str)
    m["true_uav_id"] = m[map_uid_col].astype(str)
    m = m[["fusion_id", "true_uav_id"]].drop_duplicates()

    gt_candidates = p.merge(m, on="fusion_id", how="inner")
    gt_candidates["report_id"] = gt_candidates["true_uav_id"].astype(str)

    r = report_tracks.copy()
    if r.empty:
        return pd.DataFrame(columns=["time", "fusion_id", "report_id", "true_uav_id", "scenario_case"])
    r_id_col = _first_existing(r, ["id", "track_id"])
    r_time_col = _first_existing(r, ["time", "timestamp", "ts", "t"])
    if r_id_col is None or r_time_col is None:
        raise ValueError("report_tracks.csv must contain id/time columns.")
    r["report_id"] = r[r_id_col].astype(str)
    r["time"] = _norm_time_series(r[r_time_col], time_decimals)
    r = r[np.isfinite(r["time"])].copy()
    r = r[["time", "report_id"]].drop_duplicates()

    gt = gt_candidates.merge(r, on=["time", "report_id"], how="inner")
    gt = gt[["time", "fusion_id", "report_id", "true_uav_id"]].drop_duplicates()

    if scenario_assignment is not None and not scenario_assignment.empty:
        sa = scenario_assignment.copy()
        if "true_uav_id" in sa.columns:
            sa["true_uav_id"] = sa["true_uav_id"].astype(str)
            cols = ["true_uav_id"]
            if "scenario_case" in sa.columns:
                cols.append("scenario_case")
            if len(cols) > 1:
                gt = gt.merge(sa[cols].drop_duplicates(), on="true_uav_id", how="left")
    if "scenario_case" not in gt.columns:
        gt["scenario_case"] = ""

    return gt.sort_values(["time", "fusion_id", "report_id"]).reset_index(drop=True)


def _compute_id_switches(
    gt_pairs: pd.DataFrame,
    pred_pairs: pd.DataFrame,
    reset_on_miss: bool,
) -> tuple[int, pd.DataFrame]:
    if gt_pairs.empty:
        cols = ["time", "fusion_id", "gt_report_id", "pred_report_id", "is_tp", "idsw_here", "scenario_case"]
        return 0, pd.DataFrame(columns=cols)

    pred_assign = pred_pairs.copy()
    pred_assign = pred_assign.sort_values(["time", "pair_score"], ascending=[True, False], na_position="last")
    pred_assign = pred_assign.drop_duplicates(subset=["time", "fusion_id"], keep="first")
    pred_assign = pred_assign[["time", "fusion_id", "report_id"]].rename(columns={"report_id": "pred_report_id"})

    timeline = gt_pairs.copy().rename(columns={"report_id": "gt_report_id"})
    timeline = timeline.merge(pred_assign, on=["time", "fusion_id"], how="left")
    timeline["pred_report_id"] = timeline["pred_report_id"].fillna("")
    timeline["is_tp"] = timeline["pred_report_id"].astype(str) == timeline["gt_report_id"].astype(str)
    timeline["idsw_here"] = 0

    idsw = 0
    for fusion_id, g in timeline.groupby("fusion_id", sort=False):
        prev_pred = None
        # Iterate in time order across GT-visible frames for this object.
        gg = g.sort_values("time")
        for idx, row in gg.iterrows():
            cur_pred = str(row["pred_report_id"] or "")
            if cur_pred == "":
                if bool(reset_on_miss):
                    prev_pred = None
                continue
            if prev_pred is not None and cur_pred != prev_pred:
                idsw += 1
                timeline.at[idx, "idsw_here"] = 1
            prev_pred = cur_pred

    return int(idsw), timeline.sort_values(["time", "fusion_id"]).reset_index(drop=True)


def evaluate_from_output_dir(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    time_decimals = int(CFG.get("TIME_ROUND_DECIMALS", 6))

    match_results = _load_json(output_dir / str(CFG["MATCH_RESULTS_JSON"]))
    passive_tracks = _load_csv(output_dir / str(CFG["PASSIVE_TRACKS_CSV"]))
    passive_map = _load_csv(output_dir / str(CFG["PASSIVE_TRACK_TRUTH_MAP_CSV"]))
    report_tracks = _load_csv(output_dir / str(CFG["REPORT_TRACKS_CSV"]))
    scenario_assignment = _load_csv(output_dir / str(CFG["SCENARIO_ASSIGNMENT_CSV"]), required=False)

    pred_pairs = _flatten_match_results_to_pred_pairs(match_results, time_decimals=time_decimals)
    gt_pairs = _build_gt_pairs(
        passive_tracks=passive_tracks,
        passive_map=passive_map,
        report_tracks=report_tracks,
        scenario_assignment=scenario_assignment,
        time_decimals=time_decimals,
    )

    # TP by exact pair match on (time, fusion_id, report_id)
    pred_key = pred_pairs[["time", "fusion_id", "report_id"]].drop_duplicates()
    gt_key = gt_pairs[["time", "fusion_id", "report_id"]].drop_duplicates()
    tp_pairs = gt_key.merge(pred_key, on=["time", "fusion_id", "report_id"], how="inner")

    tp = int(len(tp_pairs))
    pred_cnt = int(len(pred_key))
    gt_cnt = int(len(gt_key))
    fp = max(0, pred_cnt - tp)
    fn = max(0, gt_cnt - tp)

    precision = _safe_div(tp, pred_cnt)
    recall = _safe_div(tp, gt_cnt)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    idsw, object_timeline = _compute_id_switches(
        gt_pairs=gt_pairs,
        pred_pairs=pred_pairs,
        reset_on_miss=bool(CFG.get("IDSW_RESET_ON_MISS", False)),
    )
    mota = 1.0 - _safe_div(fn + fp + idsw, gt_cnt)
    if bool(CFG.get("CLIP_MOTA_TO_0_1", False)):
        mota = float(np.clip(mota, 0.0, 1.0))

    # Per-frame summary
    pred_frame = pred_key.groupby("time").size().rename("pred_count")
    gt_frame = gt_key.groupby("time").size().rename("gt_count")
    tp_frame = tp_pairs.groupby("time").size().rename("tp_count")
    frame_stats = pd.concat([gt_frame, pred_frame, tp_frame], axis=1).fillna(0).reset_index()
    frame_stats["gt_count"] = frame_stats["gt_count"].astype(int)
    frame_stats["pred_count"] = frame_stats["pred_count"].astype(int)
    frame_stats["tp_count"] = frame_stats["tp_count"].astype(int)
    frame_stats["fp_count"] = (frame_stats["pred_count"] - frame_stats["tp_count"]).clip(lower=0).astype(int)
    frame_stats["fn_count"] = (frame_stats["gt_count"] - frame_stats["tp_count"]).clip(lower=0).astype(int)
    if not object_timeline.empty:
        idsw_frame = object_timeline.groupby("time")["idsw_here"].sum().rename("idsw_count")
        frame_stats = frame_stats.merge(idsw_frame.reset_index(), on="time", how="left")
    else:
        frame_stats["idsw_count"] = 0
    frame_stats["idsw_count"] = frame_stats["idsw_count"].fillna(0).astype(int)
    frame_stats["precision"] = frame_stats.apply(lambda r: _safe_div(r["tp_count"], r["pred_count"]), axis=1)
    frame_stats["recall"] = frame_stats.apply(lambda r: _safe_div(r["tp_count"], r["gt_count"]), axis=1)
    frame_stats["f1"] = frame_stats.apply(
        lambda r: _safe_div(2.0 * r["precision"] * r["recall"], r["precision"] + r["recall"]),
        axis=1,
    )
    frame_stats["mota"] = frame_stats.apply(
        lambda r: 1.0 - _safe_div(r["fn_count"] + r["fp_count"] + r["idsw_count"], r["gt_count"]),
        axis=1,
    )
    frame_stats = frame_stats.sort_values("time").reset_index(drop=True)

    # Breakdown by scenario_case (for GT-positive categories only)
    gt_case = gt_pairs[["time", "fusion_id", "report_id", "scenario_case"]].copy()
    tp_case = tp_pairs.merge(gt_case, on=["time", "fusion_id", "report_id"], how="left")
    gt_case_cnt = gt_case.groupby("scenario_case").size().rename("gt_count")
    tp_case_cnt = tp_case.groupby("scenario_case").size().rename("tp_count")

    # For a case-conditioned precision, count predictions for GT objects of that case.
    pred_case = pred_key.merge(
        gt_case[["time", "fusion_id", "scenario_case"]].drop_duplicates(),
        on=["time", "fusion_id"],
        how="left",
    )
    pred_case_cnt = pred_case.groupby("scenario_case").size().rename("pred_count")

    by_case_df = pd.concat([gt_case_cnt, pred_case_cnt, tp_case_cnt], axis=1).fillna(0).reset_index()
    if by_case_df.empty:
        by_case_records: List[Dict[str, Any]] = []
    else:
        by_case_df["gt_count"] = by_case_df["gt_count"].astype(int)
        by_case_df["pred_count"] = by_case_df["pred_count"].astype(int)
        by_case_df["tp_count"] = by_case_df["tp_count"].astype(int)
        by_case_df["fp_count"] = (by_case_df["pred_count"] - by_case_df["tp_count"]).clip(lower=0).astype(int)
        by_case_df["fn_count"] = (by_case_df["gt_count"] - by_case_df["tp_count"]).clip(lower=0).astype(int)
        by_case_df["precision"] = by_case_df.apply(lambda r: _safe_div(r["tp_count"], r["pred_count"]), axis=1)
        by_case_df["recall"] = by_case_df.apply(lambda r: _safe_div(r["tp_count"], r["gt_count"]), axis=1)
        by_case_df["f1"] = by_case_df.apply(
            lambda r: _safe_div(2.0 * r["precision"] * r["recall"], r["precision"] + r["recall"]),
            axis=1,
        )
        by_case_records = [
            {
                "scenario_case": str(r["scenario_case"]),
                "gt_count": int(r["gt_count"]),
                "pred_count": int(r["pred_count"]),
                "tp_count": int(r["tp_count"]),
                "fp_count": int(r["fp_count"]),
                "fn_count": int(r["fn_count"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
            }
            for _, r in by_case_df.iterrows()
        ]

    summary: Dict[str, Any] = {
        "metric_definition": {
            "precision": "TP / (TP + FP), exact pair match on (time, fusion_id, report_id)",
            "recall": "TP / GT_pairs",
            "f1": "harmonic mean of precision and recall",
            "mota": "1 - (FN + FP + IDSW) / GT_pairs",
            "idsw_definition": "For the same passive GT object, predicted report_id changes between two predicted frames.",
        },
        "counts": {
            "gt_pairs": gt_cnt,
            "pred_pairs": pred_cnt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "idsw": int(idsw),
            "frames_in_match_results": int(len(match_results)),
            "frames_with_gt": int(frame_stats["time"].nunique()) if not frame_stats.empty else 0,
        },
        "metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1Score": f1,
            "MOTA": float(mota),
        },
        "frame_metrics_summary": {
            "avg_pred_pairs_per_frame": float(frame_stats["pred_count"].mean()) if not frame_stats.empty else 0.0,
            "avg_gt_pairs_per_frame": float(frame_stats["gt_count"].mean()) if not frame_stats.empty else 0.0,
            "avg_tp_per_frame": float(frame_stats["tp_count"].mean()) if not frame_stats.empty else 0.0,
            "avg_precision_per_frame": float(frame_stats["precision"].mean()) if not frame_stats.empty else 0.0,
            "avg_recall_per_frame": float(frame_stats["recall"].mean()) if not frame_stats.empty else 0.0,
            "avg_mota_per_frame": float(frame_stats["mota"].mean()) if not frame_stats.empty else 0.0,
            "median_pred_pairs_per_frame": float(frame_stats["pred_count"].median()) if not frame_stats.empty else 0.0,
            "median_gt_pairs_per_frame": float(frame_stats["gt_count"].median()) if not frame_stats.empty else 0.0,
            "median_tp_per_frame": float(frame_stats["tp_count"].median()) if not frame_stats.empty else 0.0,
        },
        "by_scenario_case": by_case_records,
        "files": {
            "output_dir": str(output_dir),
            "match_results_json": str(output_dir / str(CFG["MATCH_RESULTS_JSON"])),
            "passive_tracks_csv": str(output_dir / str(CFG["PASSIVE_TRACKS_CSV"])),
            "passive_track_truth_map_csv": str(output_dir / str(CFG["PASSIVE_TRACK_TRUTH_MAP_CSV"])),
            "report_tracks_csv": str(output_dir / str(CFG["REPORT_TRACKS_CSV"])),
        },
    }

    # Save artifacts
    frame_stats.to_csv(output_dir / str(CFG["SAVE_FRAME_CSV"]), index=False)
    pred_pairs.to_csv(output_dir / str(CFG["SAVE_PRED_PAIRS_CSV"]), index=False)
    gt_pairs.to_csv(output_dir / str(CFG["SAVE_GT_PAIRS_CSV"]), index=False)
    tp_pairs.to_csv(output_dir / str(CFG["SAVE_MATCHED_PAIRS_CSV"]), index=False)
    object_timeline.to_csv(output_dir / str(CFG["SAVE_OBJECT_TIMELINE_CSV"]), index=False)
    _json_dump(summary, output_dir / str(CFG["SAVE_SUMMARY_JSON"]))

    return summary


def main() -> None:
    out_dir = _p(str(CFG["OUTPUT_DIR"]))
    summary = evaluate_from_output_dir(out_dir)
    m = summary["metrics"]
    c = summary["counts"]
    _log(
        "[EVAL] "
        f"Precision={m['Precision']:.4f} Recall={m['Recall']:.4f} F1={m['F1Score']:.4f} MOTA={m['MOTA']:.4f} | "
        f"TP={c['tp']} FP={c['fp']} FN={c['fn']} IDSW={c['idsw']} GT={c['gt_pairs']}"
    )


if __name__ == "__main__":
    main()

