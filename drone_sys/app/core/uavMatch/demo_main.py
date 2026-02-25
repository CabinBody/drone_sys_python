# -*- coding: utf-8 -*-
"""
uavMatch 图特征匹配异常识别 Demo

运行方式（示例）:
    python drone_sys/app/core/uavMatch/demo_main.py
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from match import CONFIG as BASE_CONFIG
from match import GraphTrackAnomalyDetector, MatchConfig


# ================================================================
# Demo 可调参数（集中在前面）
# ================================================================

DEMO_CONFIG = {
    "BASE_LAT": 30.0,
    "BASE_LON": 120.0,
    "BASE_ALT": 80.0,
    "NUM_FRAMES": 7,
    "DT": 1.0,
    "SEED": 7,
    "PRINT_FRAME_DETAIL": True,
    "SAVE_PLOT": True,
    "PLOT_OUTPUT_DIR": "drone_sys/app/core/uavMatch/demo_plots",
    "PLOT_FILE_NAME": "demo_anomaly_trajectory_compare.png",
    "PLOT_DPI": 160,
}


def meters_to_lat(m: float) -> float:
    return m / 111320.0


def meters_to_lon(m: float, lat_deg: float) -> float:
    return m / (111320.0 * np.cos(np.radians(lat_deg)))


def build_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构造一个小型演示数据集:
    - F1 <-> R1 正常匹配
    - F2 <-> R2 在后半段出现持续漂移/偏航
    - F_missing 仅融合侧出现 -> 未上报
    - R_fake 仅上报侧出现 -> 伪报
    - R1_dup 与 R1 高度接近 -> 重复航迹
    """
    cfg = DEMO_CONFIG
    rng = np.random.default_rng(cfg["SEED"])

    base_lat = cfg["BASE_LAT"]
    base_lon = cfg["BASE_LON"]
    base_alt = cfg["BASE_ALT"]
    num_frames = int(cfg["NUM_FRAMES"])
    dt = float(cfg["DT"])

    fusion_rows: List[Dict] = []
    report_rows: List[Dict] = []

    for k in range(num_frames):
        t = (k + 1) * dt

        # ---------- F1 / R1: 稳定正常 ----------
        f1_e = 15.0 * k
        f1_n = 2.0 * k
        f1_lat = base_lat + meters_to_lat(f1_n)
        f1_lon = base_lon + meters_to_lon(f1_e, base_lat)

        fusion_rows.append(
            {
                "id": "F1",
                "time": t,
                "lat": f1_lat,
                "lon": f1_lon,
                "alt": base_alt,
                "vx": 15.0,
                "vy": 2.0,
                "vz": 0.0,
                "confidence": 0.95,
            }
        )

        report_rows.append(
            {
                "id": "R1",
                "time": t,
                "lat": f1_lat + meters_to_lat(float(rng.normal(0, 2.0))),
                "lon": f1_lon + meters_to_lon(float(rng.normal(0, 2.0)), base_lat),
                "alt": base_alt + float(rng.normal(0, 0.5)),
                "vx": 14.5,
                "vy": 2.5,
                "vz": 0.0,
                "confidence": 0.96,
            }
        )

        # ---------- F2 / R2: 后半段漂移 + 偏航 ----------
        f2_e = -20.0
        f2_n = 20.0 + 12.0 * k
        f2_lat = base_lat + meters_to_lat(f2_n)
        f2_lon = base_lon + meters_to_lon(f2_e, base_lat)

        fusion_rows.append(
            {
                "id": "F2",
                "time": t,
                "lat": f2_lat,
                "lon": f2_lon,
                "alt": base_alt + 10.0,
                "vx": 0.0,
                "vy": 12.0,
                "vz": 0.0,
                "confidence": 0.90,
            }
        )

        # k>=3 后逐帧偏移增大，且速度方向转向东偏北，制造偏航
        if k < 3:
            off_e, off_n = 3.0, 2.0
            r2_vx, r2_vy = 1.0, 11.5
        else:
            off_e = 80.0 + 45.0 * (k - 2)   # 125,170,215,260 m ...
            off_n = 10.0 + 8.0 * (k - 2)
            r2_vx, r2_vy = 10.0, 4.0        # 与 F2 方向明显不同

        report_rows.append(
            {
                "id": "R2",
                "time": t,
                "lat": f2_lat + meters_to_lat(off_n),
                "lon": f2_lon + meters_to_lon(off_e, base_lat),
                "alt": base_alt + 10.0,
                "vx": r2_vx,
                "vy": r2_vy,
                "vz": 0.0,
                "confidence": 0.90,
            }
        )

        # ---------- F_missing: 仅融合侧（未上报） ----------
        if 1 <= k <= 5:
            # 放在与所有上报轨迹明显分离的位置，避免被误匹配
            fm_e = -980.0 + 8.0 * k
            fm_n = -860.0 + 5.0 * k
            fusion_rows.append(
                {
                    "id": "F_missing",
                    "time": t,
                    "lat": base_lat + meters_to_lat(fm_n),
                    "lon": base_lon + meters_to_lon(fm_e, base_lat),
                    "alt": base_alt + 5.0,
                    "vx": 6.0,
                    "vy": 3.0,
                    "vz": 0.0,
                    "confidence": 0.88,
                }
            )

        # ---------- R_fake: 仅上报侧（伪报） ----------
        if 1 <= k <= 5:
            rf_e = 1300.0 + 10.0 * k  # 远离所有融合点，超过候选半径
            rf_n = 900.0
            report_rows.append(
                {
                    "id": "R_fake",
                    "time": t,
                    "lat": base_lat + meters_to_lat(rf_n),
                    "lon": base_lon + meters_to_lon(rf_e, base_lat),
                    "alt": base_alt + 20.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "confidence": 0.92,
                }
            )

        # ---------- R1_dup: 贴近 R1 的重复航迹 ----------
        if 2 <= k <= 4:
            report_rows.append(
                {
                    "id": "R1_dup",
                    "time": t,
                    "lat": f1_lat + meters_to_lat(4.0),
                    "lon": f1_lon + meters_to_lon(-3.0, base_lat),
                    "alt": base_alt,
                    "vx": 14.2,
                    "vy": 2.8,
                    "vz": 0.0,
                    "confidence": 0.70,
                }
            )

    df_fusion = pd.DataFrame(fusion_rows)
    df_report = pd.DataFrame(report_rows)
    return df_fusion, df_report


def _to_local_xy(df: pd.DataFrame, ref_lat: float, ref_lon: float) -> pd.DataFrame:
    out = df.copy()
    out["x_m"] = (out["lon"] - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    out["y_m"] = (out["lat"] - ref_lat) * 111320.0
    return out


def _plot_track(ax, df: pd.DataFrame, track_id: str, color: str, style: str, label: str) -> None:
    sub = df[df["id"] == track_id].sort_values("time")
    if sub.empty:
        return
    ax.plot(
        sub["x_m"].to_numpy(),
        sub["y_m"].to_numpy(),
        style,
        color=color,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=label,
    )
    # 标注起止点
    ax.scatter(sub["x_m"].iloc[0], sub["y_m"].iloc[0], c=color, marker="s", s=40)
    ax.scatter(sub["x_m"].iloc[-1], sub["y_m"].iloc[-1], c=color, marker="*", s=70)


def _extract_frame_flags(results: List[Dict]) -> Dict[Tuple[str, float], List[str]]:
    """
    返回 {(track_id, time): [flag_type, ...]}，便于在图上高亮异常点
    """
    flags: Dict[Tuple[str, float], List[str]] = {}
    for res in results:
        t = float(res["time"])
        for name, key in [
            ("unreported", "unreported"),
            ("false", "false_reports"),
            ("duplicate", "duplicate_reports"),
            ("drift", "drift_candidates"),
            ("deviation", "deviation_candidates"),
        ]:
            for tid in res.get(key, []):
                flags.setdefault((str(tid), t), []).append(name)
    return flags


def _scatter_flag_points(ax, df: pd.DataFrame, frame_flags: Dict[Tuple[str, float], List[str]], track_ids: List[str]) -> None:
    flag_style = {
        "unreported": ("x", "tab:red"),
        "false": ("x", "tab:orange"),
        "duplicate": ("D", "tab:purple"),
        "drift": ("^", "tab:brown"),
        "deviation": ("P", "tab:green"),
    }
    for tid in track_ids:
        sub = df[df["id"] == tid].sort_values("time")
        for row in sub.itertuples(index=False):
            marks = frame_flags.get((str(row.id), float(row.time)), [])
            for m in marks:
                mk, color = flag_style[m]
                ax.scatter(row.x_m, row.y_m, marker=mk, c=color, s=65, linewidths=1.4)


def plot_demo_anomaly_trajectories(
    df_fusion: pd.DataFrame,
    df_report: pd.DataFrame,
    results: List[Dict],
) -> Path:
    cfg = DEMO_CONFIG
    out_dir = Path(cfg["PLOT_OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["PLOT_FILE_NAME"]

    ref_lat = float(pd.concat([df_fusion["lat"], df_report["lat"]], axis=0).mean())
    ref_lon = float(pd.concat([df_fusion["lon"], df_report["lon"]], axis=0).mean())
    fxy = _to_local_xy(df_fusion, ref_lat, ref_lon)
    rxy = _to_local_xy(df_report, ref_lat, ref_lon)
    frame_flags = _extract_frame_flags(results)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    # 1) 总览图
    ax = axes[0]
    for tid in sorted(fxy["id"].unique()):
        _plot_track(ax, fxy, tid, color="tab:blue", style="-", label=f"fusion:{tid}")
    for tid in sorted(rxy["id"].unique()):
        _plot_track(ax, rxy, tid, color="tab:orange", style="--", label=f"report:{tid}")
    _scatter_flag_points(ax, fxy, frame_flags, sorted(fxy["id"].unique().tolist()))
    _scatter_flag_points(ax, rxy, frame_flags, sorted(rxy["id"].unique().tolist()))
    ax.set_title("Overview (all demo tracks)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    # 去重图例
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")

    # 2) 未上报：F_missing 与其他上报轨迹对比
    ax = axes[1]
    for tid in sorted(rxy["id"].unique()):
        _plot_track(ax, rxy, tid, color="lightgray", style="--", label=f"report:{tid}")
    _plot_track(ax, fxy, "F_missing", color="tab:red", style="-", label="fusion:F_missing")
    _scatter_flag_points(ax, fxy, frame_flags, ["F_missing"])
    ax.set_title("Unreported track (fusion only)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 3) 伪报：R_fake 与融合轨迹对比
    ax = axes[2]
    for tid in sorted(fxy["id"].unique()):
        _plot_track(ax, fxy, tid, color="lightgray", style="-", label=f"fusion:{tid}")
    _plot_track(ax, rxy, "R_fake", color="tab:orange", style="--", label="report:R_fake")
    _scatter_flag_points(ax, rxy, frame_flags, ["R_fake"])
    ax.set_title("False report track (report only)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 4) 重复航迹：R1 / R1_dup / F1
    ax = axes[3]
    _plot_track(ax, fxy, "F1", color="tab:blue", style="-", label="fusion:F1")
    _plot_track(ax, rxy, "R1", color="tab:orange", style="--", label="report:R1")
    if "R1_dup" in set(rxy["id"].unique()):
        _plot_track(ax, rxy, "R1_dup", color="tab:purple", style="--", label="report:R1_dup")
        _scatter_flag_points(ax, rxy, frame_flags, ["R1_dup"])
    ax.set_title("Duplicate report comparison")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 5) 偏航/漂移：F2 vs R2
    ax = axes[4]
    _plot_track(ax, fxy, "F2", color="tab:blue", style="-", label="fusion:F2")
    _plot_track(ax, rxy, "R2", color="tab:green", style="--", label="report:R2")
    _scatter_flag_points(ax, fxy, frame_flags, ["F2"])
    _scatter_flag_points(ax, rxy, frame_flags, ["R2"])

    # 连接同一时刻 F2-R2，突出偏移增长
    f2 = fxy[fxy["id"] == "F2"][["time", "x_m", "y_m"]].copy()
    r2 = rxy[rxy["id"] == "R2"][["time", "x_m", "y_m"]].copy()
    merged = f2.merge(r2, on="time", suffixes=("_f", "_r"))
    for row in merged.itertuples(index=False):
        ax.plot([row.x_m_f, row.x_m_r], [row.y_m_f, row.y_m_r], color="gray", alpha=0.35, linewidth=1.0)
    ax.set_title("Drift / deviation pair comparison (F2 vs R2)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 6) 全局评分时间序列（辅助理解）
    ax = axes[5]
    ts = [float(r["time"]) for r in results]
    gs = [float(r["global_score"]) for r in results]
    ax.plot(ts, gs, "-o", color="tab:blue", label="global_score")
    ax.axhline(0.35, color="tab:red", linestyle="--", linewidth=1.5, label="global_threshold(0.35)")
    for r in results:
        if r.get("events"):
            ax.scatter(float(r["time"]), float(r["global_score"]), c="tab:red", s=70, marker="x")
    ax.set_title("Global score by frame")
    ax.set_xlabel("time")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("uavMatch Demo: anomaly trajectory comparison", fontsize=14)
    fig.savefig(out_path, dpi=int(cfg["PLOT_DPI"]))
    plt.close(fig)
    return out_path


def run_demo() -> List[Dict]:
    demo_cfg = deepcopy(BASE_CONFIG)

    # 为演示效果调整少量阈值（不改你算法主体）
    demo_cfg.update(
        {
            "PRINT_DETAIL": False,
            "WINDOW_DURATION_S": 4.0,
            "GRAPH_KNN_FUSION": 3,
            "GRAPH_KNN_REPORT": 3,
            "TOPK_PER_FUSION": 4,
            "CANDIDATE_RADIUS_M": 1000.0,
            "VOTE_WINDOW_FRAMES": 4,
            "VOTE_MIN_OBS": 2,
            "VOTE_MIN_TRUE_COUNT": 2,
            "VOTE_TRIGGER_RATIO": 0.5,
            "MIN_MATCH_NODE_SCORE": 0.03,
            "MIN_MATCH_PAIR_SCORE": 0.05,
            "PAIR_SCORE_THRESHOLD": 0.35,
            "DRIFT_OFFSET_M": 140.0,
            "DEVIATION_OFFSET_M": 100.0,
            "DEVIATION_HEADING_DEG": 30.0,
            "DEVIATION_GROWTH_FRAMES": 2,
            "DUPLICATE_REPORT_DIST_M": 60.0,
            "DUPLICATE_REPORT_VEL_MPS": 6.0,
        }
    )

    detector = GraphTrackAnomalyDetector(MatchConfig(demo_cfg))
    df_fusion, df_report = build_demo_data()

    results = detector.process_stream(df_fusion, df_report)

    print("========== Demo 输入概览 ==========")
    print(f"融合样本数: {len(df_fusion)}, 上报样本数: {len(df_report)}")
    print(f"融合轨迹ID: {sorted(df_fusion['id'].unique().tolist())}")
    print(f"上报轨迹ID: {sorted(df_report['id'].unique().tolist())}")

    print("\n========== Demo 帧级结果 ==========")
    for res in results:
        t = res["time"]
        print(
            f"[t={t:.0f}] matches={len(res['matches'])} "
            f"global_score={res['global_score']:.3f} "
            f"unreported={res['unreported']} false={res['false_reports']} "
            f"dup={res['duplicate_reports']}"
        )
        if DEMO_CONFIG["PRINT_FRAME_DETAIL"]:
            if res["drift_candidates"]:
                print("  漂移候选:", res["drift_candidates"])
            if res["deviation_candidates"]:
                print("  偏航候选:", res["deviation_candidates"])
            if res["events"]:
                print("  触发事件:", res["events"])

    print("\n========== Demo 最终累计异常实体 ==========")
    print(sorted(detector.abnormal_entities))

    if results:
        print("\n========== 最后一帧完整结果（节选） ==========")
        last = results[-1]
        print("matches:", last["matches"])
        print("voted_anomalies:", last["voted_anomalies"])
        print("events:", last["events"])

    if DEMO_CONFIG.get("SAVE_PLOT", True):
        plot_path = plot_demo_anomaly_trajectories(df_fusion, df_report, results)
        print("\n========== Demo 异常轨迹对比图 ==========")
        print(f"已保存: {plot_path}")

    return results


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
