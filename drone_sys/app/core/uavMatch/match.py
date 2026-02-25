# -*- coding: utf-8 -*-
"""
基于图特征匹配的低空异常航迹识别（滑动窗版本）

实现要点（对应 `图特征匹配.md`）：
1. 在滑动时间窗内将“融合航迹”和“上报航迹”聚合为轨迹段节点（一个当前航迹ID -> 一个节点）
2. 基于节点特征（位置/速度/轨迹段趋势/质量）计算高斯核相似度
3. 基于图边特征（邻接位移/速度差）构建边相似度
4. 构建加权亲和矩阵，并使用谱松弛 + 贪心离散化求一对一匹配
5. 输出一致性评分，并结合连续帧投票判定异常类型

注意：
- 为保证工程可用性，代码对常见列名做了兼容；缺失字段会使用默认值（如 alt/vz/quality）
- 若需更严格的QAP/匈牙利离散化，可在此基础上替换 `_discretize_matches`
"""

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ================================================================
#  所有可配置参数（按你的要求集中放在代码前面）
# ================================================================

CONFIG = {
    # ---------------- 输入输出 ----------------
    'FUSION_CSV_PATH': 'data/fusion_output.csv',
    'REPORT_CSV_PATH': 'data/report_stream.csv',
    'OUTPUT_JSON_PATH': '',  # 为空表示不写文件
    'PRINT_DETAIL': False,

    # ---------------- 时间窗/状态管理 ----------------
    'WINDOW_DURATION_S': 5.0,        # 滑动时间窗长度（秒）
    'WINDOW_MIN_POINTS': 1,          # 构建轨迹段节点至少需要的点数
    'MAX_TRACK_IDLE_S': 10.0,        # 历史缓存中轨迹最大闲置时间（秒）
    'MATCH_ACTIVE_GAP_S': 2.0,       # 当前帧节点允许的最后观测滞后（秒）

    # ---------------- 图构建（kNN邻接） ----------------
    'GRAPH_KNN_FUSION': 5,
    'GRAPH_KNN_REPORT': 5,
    'MAX_CANDIDATE_PAIRS_FOR_AFFINITY': 400,  # 限制亲和矩阵规模

    # ---------------- 候选门控 + Top-K预算 ----------------
    'CANDIDATE_RADIUS_M': 800.0,     # 空间门控半径（米）
    'CANDIDATE_TIME_DIFF_S': 1.5,    # 时间门控（秒）
    'MIN_CANDIDATE_NODE_SCORE': 1e-4,
    'TOPK_PER_FUSION': 5,
    'RELAXED_CANDIDATE_FALLBACK': True,  # 门控后为空时保留最佳候选（弱约束）
    'RELAXED_MIN_NODE_SCORE': 1e-6,
    'DISABLE_CANDIDATE_GATING': False,  # No-G消融：关闭候选硬门控（仍保留Top-K与总预算）

    # ---------------- 节点相似度（高斯核） ----------------
    'SIGMA_NODE_POS_M': 120.0,
    'SIGMA_NODE_VEL_MPS': 18.0,
    'SIGMA_NODE_PATH_M': 180.0,      # 窗口轨迹段位移趋势差
    'SIGMA_NODE_TIME_S': 0.8,
    'SIGMA_NODE_QUALITY': 0.35,      # 质量差（0~1）核
    'KERNEL_MODE': 'gaussian',       # 'gaussian' | 'euclid_cosine'（消融用）
    'NODE_QUALITY_EXP_FUSION': 1.0,  # 质量权重指数
    'NODE_QUALITY_EXP_REPORT': 1.0,

    # ---------------- 边相似度（高斯核） ----------------
    'SIGMA_EDGE_DP_M': 200.0,
    'SIGMA_EDGE_DV_MPS': 20.0,

    # ---------------- 亲和矩阵/图匹配 ----------------
    'AFFINITY_NODE_WEIGHT': 1.0,
    'AFFINITY_EDGE_WEIGHT': 0.7,
    'SPECTRAL_POWER_ITERS': 30,
    'SPECTRAL_DAMPING': 0.85,        # 与对角节点项混合，增强稳定性
    'MIN_MATCH_NODE_SCORE': 0.05,    # 最低节点匹配分数
    'MIN_MATCH_PAIR_SCORE': 0.10,    # 最低综合匹配分数（节点+边）

    # ---------------- 一致性评分/异常阈值 ----------------
    'GLOBAL_SCORE_THRESHOLD': 0.35,  # 全局一致性阈值（可用于整体告警）
    'PAIR_SCORE_THRESHOLD': 0.28,    # 匹配对低一致性阈值（漂移）
    'DRIFT_OFFSET_M': 150.0,         # 漂移偏移阈值
    'DEVIATION_OFFSET_M': 120.0,     # 偏航偏移阈值（走廊外）
    'DEVIATION_HEADING_DEG': 35.0,   # 偏航方向差阈值
    'DEVIATION_GROWTH_FRAMES': 3,    # 偏航要求偏移持续增长的帧数
    'OFFSET_GROWTH_TOLERANCE_M': 3.0,

    # ---------------- 重复航迹（上报侧） ----------------
    'DUPLICATE_NODE_SCORE_THRESHOLD': 0.25,
    'DUPLICATE_REPORT_DIST_M': 80.0,
    'DUPLICATE_REPORT_VEL_MPS': 8.0,

    # ---------------- 连续帧投票机制 ----------------
    'VOTE_WINDOW_FRAMES': 5,         # 投票窗口长度 T
    'VOTE_TRIGGER_RATIO': 0.6,       # 投票比例阈值 gamma
    'VOTE_MIN_TRUE_COUNT': 3,        # 最小异常帧数（与上式同时满足）
    'VOTE_MIN_OBS': 3,               # 至少观察帧数

    # ---------------- 列名兼容（按需改） ----------------
    'ID_COL_CANDIDATES': ['id', 'track_id', 'tid', 'uav_id', 'target_id'],
    'TIME_COL_CANDIDATES': ['time', 'timestamp', 'ts', 't'],
    'LAT_COL_CANDIDATES': ['lat', 'latitude'],
    'LON_COL_CANDIDATES': ['lon', 'lng', 'longitude'],
    'ALT_COL_CANDIDATES': ['alt', 'height', 'h'],
    'VX_COL_CANDIDATES': ['vx', 'vel_x'],
    'VY_COL_CANDIDATES': ['vy', 'vel_y'],
    'VZ_COL_CANDIDATES': ['vz', 'vel_z'],
    'HEADING_COL_CANDIDATES': ['heading', 'course', 'yaw'],
    'QUALITY_COL_CANDIDATES_FUSION': ['confidence', 'quality', 'score'],
    'QUALITY_COL_CANDIDATES_REPORT': ['confidence', 'quality', 'report_confidence'],
    'SNR_COL_CANDIDATES': ['snr'],
    'RSSI_COL_CANDIDATES': ['rssi'],
    'DELAY_COL_CANDIDATES': ['delay', 'latency'],
    'COVERAGE_COL_CANDIDATES': ['coverage'],
}


# ================================================================
#  基础工具
# ================================================================

EARTH_R_M = 6378137.0
EPS = 1e-12


def _first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _numeric_series(df: pd.DataFrame, names: List[str], default: float) -> pd.Series:
    col = _first_existing(df, names)
    if col is None:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(default).astype(float)


def _infer_time_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').astype(float)

    numeric_try = pd.to_numeric(series, errors='coerce')
    if numeric_try.notna().sum() >= max(1, int(0.8 * len(series))):
        return numeric_try.astype(float)

    dt = pd.to_datetime(series, errors='coerce')
    if dt.notna().sum() == 0:
        raise ValueError('无法解析时间列，请检查 time/timestamp 字段格式')
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = dt.notna()
    out.loc[valid] = (dt.loc[valid].astype('int64') / 1e9).astype(float)
    return out


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _comm_quality_from_cols(
    snr: np.ndarray,
    rssi: np.ndarray,
    delay: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    snr_n = np.clip((snr - 0.0) / 30.0, 0.0, 1.0)
    rssi_n = np.clip((rssi + 90.0) / 50.0, 0.0, 1.0)
    delay_n = np.clip(1.0 - (delay / 150.0), 0.0, 1.0)
    cov_n = np.clip(coverage, 0.0, 1.0)
    return np.power((snr_n + EPS) * (rssi_n + EPS) * (delay_n + EPS) * (cov_n + EPS), 0.25)


def _latlonalt_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> np.ndarray:
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    east = dlon * EARTH_R_M * math.cos(math.radians(ref_lat))
    north = dlat * EARTH_R_M
    up = alt - ref_alt
    return np.stack([east, north, up], axis=1)


def _pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(dist2, 0.0)


def _gaussian_from_sqdist(dist2: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return (dist2 <= 0).astype(float)
    return np.exp(-0.5 * dist2 / (sigma * sigma))


def _gaussian_from_absdiff(diff: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return (np.abs(diff) <= 0).astype(float)
    return np.exp(-0.5 * (diff * diff) / (sigma * sigma))


def _euclid_sim_from_sqdist(dist2: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return (dist2 <= 0).astype(float)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    return 1.0 / (1.0 + dist / max(scale, EPS))


def _euclid_sim_from_absdiff(diff: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return (np.abs(diff) <= 0).astype(float)
    return 1.0 / (1.0 + np.abs(diff) / max(scale, EPS))


def _pairwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = a @ b.T
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = an[:, None] * bn[None, :]
    sim = np.zeros_like(dot, dtype=float)
    np.divide(dot, np.maximum(denom, EPS), out=sim, where=denom > EPS)
    # Handle zero vectors gracefully: both zero -> identical (1.0), one zero -> neutral (0.5 after mapping)
    a_zero = an <= EPS
    b_zero = bn <= EPS
    if np.any(a_zero) or np.any(b_zero):
        both_zero = a_zero[:, None] & b_zero[None, :]
        sim[both_zero] = 1.0
    return np.clip(0.5 * (sim + 1.0), 0.0, 1.0)


def _cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= EPS and nb <= EPS:
        return 1.0
    if na <= EPS or nb <= EPS:
        return 0.5
    sim = float(np.dot(aa, bb) / max(na * nb, EPS))
    return float(np.clip(0.5 * (sim + 1.0), 0.0, 1.0))


def _wrap_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def _heading_from_vector_deg(vec_xy: np.ndarray) -> float:
    if vec_xy.shape[0] < 2 or (abs(vec_xy[0]) < EPS and abs(vec_xy[1]) < EPS):
        return 0.0
    return float((math.degrees(math.atan2(vec_xy[1], vec_xy[0])) + 360.0) % 360.0)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


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


# ================================================================
#  数据结构
# ================================================================


@dataclass
class TrackSample:
    track_id: str
    time_s: float
    lat: float
    lon: float
    alt: float
    vx: float
    vy: float
    vz: float
    quality: float
    heading_deg: float


@dataclass
class TrackNode:
    track_id: str
    start_time: float
    end_time: float
    lat: float
    lon: float
    alt: float
    start_lat: float
    start_lon: float
    start_alt: float
    vx: float
    vy: float
    vz: float
    quality: float
    heading_deg: float
    num_points: int
    pos_enu: Optional[np.ndarray] = None
    start_pos_enu: Optional[np.ndarray] = None
    vel_vec: Optional[np.ndarray] = None
    path_vec: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    src: int
    dst: int
    dp: np.ndarray
    dv: np.ndarray


@dataclass
class CandidatePair:
    fusion_idx: int
    report_idx: int
    node_score: float


@dataclass
class MatchPair:
    fusion_idx: int
    report_idx: int
    fusion_id: str
    report_id: str
    node_score: float
    spectral_score: float
    edge_score: float = 0.0
    pair_score: float = 0.0
    offset_m: float = 0.0
    vel_offset_mps: float = 0.0
    heading_diff_deg: float = 0.0


class MatchConfig:
    def __init__(self, cfg: Dict[str, Any]):
        self.raw = dict(cfg)
        for key, value in cfg.items():
            setattr(self, key.lower(), value)


# ================================================================
#  图特征匹配异常识别器
# ================================================================


class GraphTrackAnomalyDetector:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.fusion_history: Dict[str, Deque[TrackSample]] = defaultdict(deque)
        self.report_history: Dict[str, Deque[TrackSample]] = defaultdict(deque)
        self.event_votes: Dict[Tuple[str, str], Deque[int]] = {}
        self.pair_metric_history: Dict[str, Dict[str, Deque[float]]] = {}
        self.abnormal_entities: set = set()

    # ---------------- 对外接口 ----------------

    def process_stream(self, df_fusion: pd.DataFrame, df_report: pd.DataFrame) -> List[Dict[str, Any]]:
        fusion_all = self._canonicalize_dataframe(df_fusion, source='fusion')
        report_all = self._canonicalize_dataframe(df_report, source='report')

        if fusion_all.empty and report_all.empty:
            return []

        all_times = sorted(set(fusion_all['time_s'].tolist()) | set(report_all['time_s'].tolist()))
        results: List[Dict[str, Any]] = []

        for t in all_times:
            frame_f = fusion_all[fusion_all['time_s'] == t].copy()
            frame_r = report_all[report_all['time_s'] == t].copy()
            frame_result = self.process_frame(frame_f, frame_r, current_time=float(t))
            results.append(frame_result)
            if self.cfg.print_detail:
                self._print_frame_result(frame_result)
        return results

    def process_frame(
        self,
        frame_fusion: pd.DataFrame,
        frame_report: pd.DataFrame,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        if current_time is None:
            candidate_times = []
            if not frame_fusion.empty:
                candidate_times.extend(frame_fusion['time_s'].astype(float).tolist())
            if not frame_report.empty:
                candidate_times.extend(frame_report['time_s'].astype(float).tolist())
            current_time = max(candidate_times) if candidate_times else 0.0

        current_time = float(current_time)
        self._append_frame_to_history(frame_fusion, self.fusion_history)
        self._append_frame_to_history(frame_report, self.report_history)
        self._prune_history(current_time)

        fusion_nodes = self._build_nodes_from_frame(frame_fusion, self.fusion_history, current_time)
        report_nodes = self._build_nodes_from_frame(frame_report, self.report_history, current_time)
        self._attach_enu_to_nodes(fusion_nodes, report_nodes)

        result: Dict[str, Any] = {
            'time': current_time,
            'window_start': current_time - float(self.cfg.window_duration_s),
            'window_end': current_time,
            'num_fusion_nodes': len(fusion_nodes),
            'num_report_nodes': len(report_nodes),
            'matches': [],
            'global_score': 0.0,
            'global_anomaly': False,
            'unreported': [],
            'false_reports': [],
            'duplicate_reports': [],
            'drift_candidates': [],
            'deviation_candidates': [],
            'events': [],
            'voted_anomalies': {
                '未上报': [],
                '伪报': [],
                '重复航迹': [],
                '漂移异常': [],
                '偏航异常': [],
            },
            'abnormal_entities': sorted(self.abnormal_entities),
        }

        if len(fusion_nodes) == 0 and len(report_nodes) == 0:
            return result
        if len(fusion_nodes) == 0 or len(report_nodes) == 0:
            empty_res = self._handle_one_side_empty(current_time, fusion_nodes, report_nodes)
            result.update(empty_res)
            result['abnormal_entities'] = sorted(self.abnormal_entities)
            return result

        fusion_edges = self._build_graph_edges(fusion_nodes, int(self.cfg.graph_knn_fusion))
        report_edges = self._build_graph_edges(report_nodes, int(self.cfg.graph_knn_report))

        node_sim, gated_mask = self._compute_node_similarity_and_gates(fusion_nodes, report_nodes)
        candidates = self._build_candidate_pairs(node_sim, gated_mask)
        affinity = self._build_affinity_matrix(candidates, fusion_edges, report_edges)
        soft_scores = self._spectral_relaxation(affinity)
        match_pairs = self._discretize_matches(
            candidates=candidates,
            soft_scores=soft_scores,
            node_sim=node_sim,
            fusion_nodes=fusion_nodes,
            report_nodes=report_nodes,
            fusion_edges=fusion_edges,
            report_edges=report_edges,
        )

        duplicate_reports = self._detect_duplicate_reports(
            fusion_nodes=fusion_nodes,
            report_nodes=report_nodes,
            node_sim=node_sim,
            match_pairs=match_pairs,
        )

        global_score = float(np.mean([m.pair_score for m in match_pairs])) if match_pairs else 0.0
        global_anomaly = bool(global_score < float(self.cfg.global_score_threshold))

        classify = self._classify_anomalies_with_votes(
            fusion_nodes=fusion_nodes,
            report_nodes=report_nodes,
            match_pairs=match_pairs,
            duplicate_reports=duplicate_reports,
            global_score=global_score,
        )

        result.update(
            {
                'matches': [self._match_to_dict(m) for m in match_pairs],
                'global_score': global_score,
                'global_anomaly': global_anomaly,
                'unreported': classify['current_flags']['未上报'],
                'false_reports': classify['current_flags']['伪报'],
                'duplicate_reports': classify['current_flags']['重复航迹'],
                'drift_candidates': classify['current_flags']['漂移异常'],
                'deviation_candidates': classify['current_flags']['偏航异常'],
                'events': classify['events'],
                'voted_anomalies': classify['voted_anomalies'],
                'abnormal_entities': sorted(self.abnormal_entities),
                'debug': {
                    'candidate_pairs': len(candidates),
                    'affinity_size': int(affinity.shape[0]),
                },
            }
        )
        return result

    # ---------------- 标准化/数据预处理 ----------------

    def _canonicalize_dataframe(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=['track_id', 'time_s', 'lat', 'lon', 'alt', 'vx', 'vy', 'vz', 'quality', 'heading_deg']
            )

        c = self.cfg
        id_col = _first_existing(df, c.id_col_candidates)
        time_col = _first_existing(df, c.time_col_candidates)
        lat_col = _first_existing(df, c.lat_col_candidates)
        lon_col = _first_existing(df, c.lon_col_candidates)
        if id_col is None or time_col is None or lat_col is None or lon_col is None:
            raise ValueError(f'{source} 数据缺少必要列。需要 id/time/lat/lon，当前列: {list(df.columns)}')

        out = pd.DataFrame(index=df.index)
        out['track_id'] = df[id_col].astype(str)
        out['time_s'] = _infer_time_seconds(df[time_col]).astype(float)
        out['lat'] = pd.to_numeric(df[lat_col], errors='coerce').astype(float)
        out['lon'] = pd.to_numeric(df[lon_col], errors='coerce').astype(float)
        out['alt'] = _numeric_series(df, c.alt_col_candidates, 0.0)
        out['vx'] = _numeric_series(df, c.vx_col_candidates, 0.0)
        out['vy'] = _numeric_series(df, c.vy_col_candidates, 0.0)
        out['vz'] = _numeric_series(df, c.vz_col_candidates, 0.0)

        heading_col = _first_existing(df, c.heading_col_candidates)
        if heading_col is not None:
            out['heading_deg'] = pd.to_numeric(df[heading_col], errors='coerce').fillna(np.nan)
        else:
            out['heading_deg'] = np.nan

        if source == 'fusion':
            q_col = _first_existing(df, c.quality_col_candidates_fusion)
            if q_col is not None:
                quality = pd.to_numeric(df[q_col], errors='coerce').fillna(1.0).astype(float).to_numpy()
                if np.nanmax(quality) > 1.5:
                    quality = np.clip(quality, 0.0, 100.0) / 100.0
                out['quality'] = _clip01(quality)
            else:
                out['quality'] = 1.0
        else:
            q_col = _first_existing(df, c.quality_col_candidates_report)
            if q_col is not None:
                quality = pd.to_numeric(df[q_col], errors='coerce').fillna(1.0).astype(float).to_numpy()
                if np.nanmax(quality) > 1.5:
                    quality = np.clip(quality, 0.0, 100.0) / 100.0
                out['quality'] = _clip01(quality)
            else:
                snr = _numeric_series(df, c.snr_col_candidates, 15.0).to_numpy()
                rssi = _numeric_series(df, c.rssi_col_candidates, -65.0).to_numpy()
                delay = _numeric_series(df, c.delay_col_candidates, 30.0).to_numpy()
                coverage = _numeric_series(df, c.coverage_col_candidates, 1.0).to_numpy()
                out['quality'] = _comm_quality_from_cols(snr, rssi, delay, coverage)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=['track_id', 'time_s', 'lat', 'lon'])
        out['alt'] = out['alt'].fillna(0.0)
        out['vx'] = out['vx'].fillna(0.0)
        out['vy'] = out['vy'].fillna(0.0)
        out['vz'] = out['vz'].fillna(0.0)
        out['quality'] = out['quality'].fillna(1.0).clip(0.0, 1.0)
        out['heading_deg'] = out['heading_deg'].astype(float)
        out = out.sort_values(['time_s', 'track_id']).reset_index(drop=True)
        return out

    # ---------------- 历史缓存/滑动窗节点构建 ----------------

    def _append_frame_to_history(
        self,
        frame_df: pd.DataFrame,
        history: Dict[str, Deque[TrackSample]],
    ) -> None:
        if frame_df is None or frame_df.empty:
            return
        for row in frame_df.itertuples(index=False):
            heading = _safe_float(getattr(row, 'heading_deg', np.nan), np.nan)
            if math.isnan(heading):
                heading = _heading_from_vector_deg(np.array([_safe_float(row.vx), _safe_float(row.vy)]))
            sample = TrackSample(
                track_id=str(row.track_id),
                time_s=float(row.time_s),
                lat=float(row.lat),
                lon=float(row.lon),
                alt=float(row.alt),
                vx=float(row.vx),
                vy=float(row.vy),
                vz=float(row.vz),
                quality=float(np.clip(row.quality, 0.0, 1.0)),
                heading_deg=float(heading),
            )
            history[sample.track_id].append(sample)

    def _prune_history(self, current_time: float) -> None:
        cutoff = current_time - float(self.cfg.window_duration_s) - float(self.cfg.max_track_idle_s)
        for history in (self.fusion_history, self.report_history):
            to_delete = []
            for track_id, dq in history.items():
                while dq and dq[0].time_s < cutoff:
                    dq.popleft()
                if not dq:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del history[track_id]

    def _build_nodes_from_frame(
        self,
        frame_df: pd.DataFrame,
        history: Dict[str, Deque[TrackSample]],
        current_time: float,
    ) -> List[TrackNode]:
        if frame_df is None or frame_df.empty:
            return []

        window_cutoff = current_time - float(self.cfg.window_duration_s)
        active_gap = float(self.cfg.match_active_gap_s)
        min_points = int(self.cfg.window_min_points)
        nodes: List[TrackNode] = []

        for track_id in sorted(frame_df['track_id'].astype(str).unique().tolist()):
            if track_id not in history:
                continue

            samples = [s for s in history[track_id] if (window_cutoff <= s.time_s <= current_time + EPS)]
            if not samples:
                continue
            samples.sort(key=lambda s: s.time_s)
            if current_time - samples[-1].time_s > active_gap:
                continue
            if len(samples) < min_points:
                continue

            s0 = samples[0]
            s1 = samples[-1]

            vx, vy, vz = s1.vx, s1.vy, s1.vz
            if len(samples) >= 2 and abs(vx) < EPS and abs(vy) < EPS and abs(vz) < EPS:
                dt = max(samples[-1].time_s - samples[0].time_s, EPS)
                vx = (s1.lon - s0.lon) / dt
                vy = (s1.lat - s0.lat) / dt
                vz = (s1.alt - s0.alt) / dt

            quality = float(np.mean([s.quality for s in samples]))
            heading = s1.heading_deg
            if math.isnan(heading):
                heading = _heading_from_vector_deg(np.array([vx, vy]))

            nodes.append(
                TrackNode(
                    track_id=track_id,
                    start_time=float(s0.time_s),
                    end_time=float(s1.time_s),
                    lat=float(s1.lat),
                    lon=float(s1.lon),
                    alt=float(s1.alt),
                    start_lat=float(s0.lat),
                    start_lon=float(s0.lon),
                    start_alt=float(s0.alt),
                    vx=float(vx),
                    vy=float(vy),
                    vz=float(vz),
                    quality=float(np.clip(quality, 0.0, 1.0)),
                    heading_deg=float(heading),
                    num_points=len(samples),
                )
            )

        return nodes

    def _attach_enu_to_nodes(self, fusion_nodes: List[TrackNode], report_nodes: List[TrackNode]) -> None:
        all_nodes = fusion_nodes + report_nodes
        if not all_nodes:
            return

        ref_lat = float(np.mean([n.lat for n in all_nodes]))
        ref_lon = float(np.mean([n.lon for n in all_nodes]))
        ref_alt = float(np.mean([n.alt for n in all_nodes]))

        end_lat = np.array([n.lat for n in all_nodes], dtype=float)
        end_lon = np.array([n.lon for n in all_nodes], dtype=float)
        end_alt = np.array([n.alt for n in all_nodes], dtype=float)
        start_lat = np.array([n.start_lat for n in all_nodes], dtype=float)
        start_lon = np.array([n.start_lon for n in all_nodes], dtype=float)
        start_alt = np.array([n.start_alt for n in all_nodes], dtype=float)

        end_enu = _latlonalt_to_enu(end_lat, end_lon, end_alt, ref_lat, ref_lon, ref_alt)
        start_enu = _latlonalt_to_enu(start_lat, start_lon, start_alt, ref_lat, ref_lon, ref_alt)

        for idx, node in enumerate(all_nodes):
            node.pos_enu = end_enu[idx]
            node.start_pos_enu = start_enu[idx]
            node.path_vec = end_enu[idx] - start_enu[idx]
            node.vel_vec = np.array([node.vx, node.vy, node.vz], dtype=float)
            if abs(node.vx) < EPS and abs(node.vy) < EPS and np.linalg.norm(node.path_vec[:2]) > EPS:
                node.heading_deg = _heading_from_vector_deg(node.path_vec[:2])

    # ---------------- 图构建与相似度 ----------------

    def _build_graph_edges(self, nodes: List[TrackNode], knn: int) -> Dict[Tuple[int, int], GraphEdge]:
        if len(nodes) <= 1:
            return {}

        pos = np.stack([n.pos_enu for n in nodes], axis=0)
        vel = np.stack([n.vel_vec for n in nodes], axis=0)
        dist2 = _pairwise_sqdist(pos, pos)
        np.fill_diagonal(dist2, np.inf)

        k = max(0, min(knn, len(nodes) - 1))
        if k == 0:
            return {}

        nbrs = np.argsort(dist2, axis=1)[:, :k]
        edges: Dict[Tuple[int, int], GraphEdge] = {}
        for i in range(len(nodes)):
            for j in nbrs[i]:
                j = int(j)
                if i == j:
                    continue
                dp = pos[j] - pos[i]
                dv = vel[j] - vel[i]
                edges[(i, j)] = GraphEdge(src=i, dst=j, dp=dp, dv=dv)
                edges[(j, i)] = GraphEdge(src=j, dst=i, dp=-dp, dv=-dv)
        return edges

    def _compute_node_similarity_and_gates(
        self,
        fusion_nodes: List[TrackNode],
        report_nodes: List[TrackNode],
    ) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        pos_f = np.stack([n.pos_enu for n in fusion_nodes], axis=0)
        pos_r = np.stack([n.pos_enu for n in report_nodes], axis=0)
        vel_f = np.stack([n.vel_vec for n in fusion_nodes], axis=0)
        vel_r = np.stack([n.vel_vec for n in report_nodes], axis=0)
        path_f = np.stack([n.path_vec for n in fusion_nodes], axis=0)
        path_r = np.stack([n.path_vec for n in report_nodes], axis=0)

        t_f = np.array([n.end_time for n in fusion_nodes], dtype=float)
        t_r = np.array([n.end_time for n in report_nodes], dtype=float)

        pos_dist2 = _pairwise_sqdist(pos_f, pos_r)
        vel_dist2 = _pairwise_sqdist(vel_f, vel_r)
        path_dist2 = _pairwise_sqdist(path_f, path_r)
        dt = np.abs(t_f[:, None] - t_r[None, :])
        kernel_mode = str(getattr(c, 'kernel_mode', 'gaussian') or 'gaussian').lower()
        if kernel_mode == 'euclid_cosine':
            s_pos = _euclid_sim_from_sqdist(pos_dist2, float(c.sigma_node_pos_m))
            s_vel = _pairwise_cosine_similarity(vel_f, vel_r)
            s_path = _pairwise_cosine_similarity(path_f, path_r)
            s_time = _euclid_sim_from_absdiff(dt, float(c.sigma_node_time_s))
        else:
            s_pos = _gaussian_from_sqdist(pos_dist2, float(c.sigma_node_pos_m))
            s_vel = _gaussian_from_sqdist(vel_dist2, float(c.sigma_node_vel_mps))
            s_path = _gaussian_from_sqdist(path_dist2, float(c.sigma_node_path_m))
            s_time = _gaussian_from_absdiff(dt, float(c.sigma_node_time_s))
        # Quality/confidence is intentionally excluded from matching score.
        # Keep reading quality fields upstream for compatibility, but do not couple
        # track association to modality confidence calibration.
        node_sim = s_pos * s_vel * s_path * s_time

        if bool(getattr(c, 'disable_candidate_gating', False)):
            # "No-G" ablation: no hard gating; Top-K and total pair budget still limit complexity.
            gated_mask = np.ones_like(node_sim, dtype=bool)
        else:
            spatial_dist = np.sqrt(np.maximum(pos_dist2, 0.0))
            gated_mask = spatial_dist <= float(c.candidate_radius_m)
            gated_mask &= dt <= float(c.candidate_time_diff_s)
            gated_mask &= node_sim >= float(c.min_candidate_node_score)
        return node_sim, gated_mask

    def _edge_similarity(self, edge_f: GraphEdge, edge_r: GraphEdge) -> float:
        c = self.cfg
        dp_dist2 = float(np.sum((edge_f.dp - edge_r.dp) ** 2))
        dv_dist2 = float(np.sum((edge_f.dv - edge_r.dv) ** 2))
        kernel_mode = str(getattr(c, 'kernel_mode', 'gaussian') or 'gaussian').lower()
        if kernel_mode == 'euclid_cosine':
            s_dp = float(_euclid_sim_from_sqdist(np.array([[dp_dist2]], dtype=float), float(c.sigma_edge_dp_m))[0, 0])
            s_dv = _cosine_similarity_vec(edge_f.dv, edge_r.dv)
        else:
            s_dp = math.exp(-0.5 * dp_dist2 / max(float(c.sigma_edge_dp_m) ** 2, EPS))
            s_dv = math.exp(-0.5 * dv_dist2 / max(float(c.sigma_edge_dv_mps) ** 2, EPS))
        return float(s_dp * s_dv)

    def _build_candidate_pairs(self, node_sim: np.ndarray, gated_mask: np.ndarray) -> List[CandidatePair]:
        c = self.cfg
        N, M = node_sim.shape
        if N == 0 or M == 0:
            return []

        candidates: List[CandidatePair] = []
        seen = set()
        for i in range(N):
            valid = np.where(gated_mask[i])[0]
            if len(valid) == 0 and bool(c.relaxed_candidate_fallback):
                best_j = int(np.argmax(node_sim[i]))
                if node_sim[i, best_j] >= float(c.relaxed_min_node_score):
                    valid = np.array([best_j], dtype=int)
            if len(valid) == 0:
                continue

            scores = node_sim[i, valid]
            order = valid[np.argsort(-scores)]
            for j in order[: int(c.topk_per_fusion)]:
                key = (int(i), int(j))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(CandidatePair(fusion_idx=int(i), report_idx=int(j), node_score=float(node_sim[i, j])))

        if len(candidates) > int(c.max_candidate_pairs_for_affinity):
            candidates.sort(key=lambda x: x.node_score, reverse=True)
            candidates = candidates[: int(c.max_candidate_pairs_for_affinity)]
        return candidates

    def _build_affinity_matrix(
        self,
        candidates: List[CandidatePair],
        fusion_edges: Dict[Tuple[int, int], GraphEdge],
        report_edges: Dict[Tuple[int, int], GraphEdge],
    ) -> np.ndarray:
        c = self.cfg
        C = len(candidates)
        if C == 0:
            return np.zeros((0, 0), dtype=float)

        A = np.zeros((C, C), dtype=float)
        node_w = float(c.affinity_node_weight)
        edge_w = float(c.affinity_edge_weight)

        for p, cand in enumerate(candidates):
            A[p, p] = node_w * cand.node_score

        # Fast path for large-scale matching when using node-only affinity.
        if edge_w <= 0.0:
            return A

        for p in range(C):
            cp = candidates[p]
            for q in range(p + 1, C):
                cq = candidates[q]
                if cp.fusion_idx == cq.fusion_idx or cp.report_idx == cq.report_idx:
                    continue
                ef = fusion_edges.get((cp.fusion_idx, cq.fusion_idx))
                er = report_edges.get((cp.report_idx, cq.report_idx))
                if ef is None or er is None:
                    continue
                s_edge = self._edge_similarity(ef, er)
                if s_edge <= 0:
                    continue
                A[p, q] = edge_w * s_edge
                A[q, p] = A[p, q]
        return A

    def _spectral_relaxation(self, affinity: np.ndarray) -> np.ndarray:
        c = self.cfg
        C = affinity.shape[0]
        if C == 0:
            return np.zeros((0,), dtype=float)
        if C == 1:
            return np.array([1.0], dtype=float)

        diag = np.clip(np.diag(affinity), 0.0, None)
        if diag.sum() <= EPS:
            diag = np.ones_like(diag, dtype=float)
        diag = diag / (diag.sum() + EPS)

        x = diag.copy()
        damping = float(c.spectral_damping)
        iters = int(c.spectral_power_iters)
        for _ in range(max(1, iters)):
            x_new = affinity @ x
            x_new = np.clip(x_new, 0.0, None)
            if x_new.sum() <= EPS:
                x_new = diag.copy()
            else:
                x_new = x_new / (x_new.sum() + EPS)
            x = damping * x_new + (1.0 - damping) * diag
            x = x / (x.sum() + EPS)
        return x

    def _discretize_matches(
        self,
        candidates: List[CandidatePair],
        soft_scores: np.ndarray,
        node_sim: np.ndarray,
        fusion_nodes: List[TrackNode],
        report_nodes: List[TrackNode],
        fusion_edges: Dict[Tuple[int, int], GraphEdge],
        report_edges: Dict[Tuple[int, int], GraphEdge],
    ) -> List[MatchPair]:
        c = self.cfg
        if not candidates:
            return []

        priorities = []
        for idx, cand in enumerate(candidates):
            spectral = float(soft_scores[idx]) if idx < len(soft_scores) else 0.0
            priority = spectral * max(cand.node_score, EPS)
            priorities.append((priority, cand.node_score, spectral, idx))
        priorities.sort(reverse=True)

        used_fusion = set()
        used_report = set()
        chosen: List[Tuple[int, CandidatePair, float]] = []
        for _, node_score, spectral, idx in priorities:
            cand = candidates[idx]
            if node_score < float(c.min_match_node_score):
                continue
            if cand.fusion_idx in used_fusion or cand.report_idx in used_report:
                continue
            used_fusion.add(cand.fusion_idx)
            used_report.add(cand.report_idx)
            chosen.append((idx, cand, float(spectral)))

        if not chosen:
            return []

        edge_supports: Dict[int, List[float]] = defaultdict(list)
        for a in range(len(chosen)):
            idx_a, cand_a, _ = chosen[a]
            for b in range(a + 1, len(chosen)):
                idx_b, cand_b, _ = chosen[b]
                ef = fusion_edges.get((cand_a.fusion_idx, cand_b.fusion_idx))
                er = report_edges.get((cand_a.report_idx, cand_b.report_idx))
                if ef is None or er is None:
                    continue
                s_edge = self._edge_similarity(ef, er)
                edge_supports[idx_a].append(s_edge)
                edge_supports[idx_b].append(s_edge)

        node_w = float(c.affinity_node_weight)
        edge_w = float(c.affinity_edge_weight)
        matches: List[MatchPair] = []

        for cand_idx, cand, spectral in chosen:
            edge_list = edge_supports.get(cand_idx, [])
            edge_score = float(np.mean(edge_list)) if edge_list else 0.0
            denom = node_w + (edge_w if edge_list else 0.0)
            pair_score = (node_w * cand.node_score + (edge_w * edge_score if edge_list else 0.0)) / max(denom, EPS)

            fn = fusion_nodes[cand.fusion_idx]
            rn = report_nodes[cand.report_idx]
            offset_m = float(np.linalg.norm(fn.pos_enu - rn.pos_enu))
            vel_offset = float(np.linalg.norm(fn.vel_vec - rn.vel_vec))
            heading_diff = _wrap_angle_diff_deg(float(fn.heading_deg), float(rn.heading_deg))

            if pair_score < float(c.min_match_pair_score):
                continue

            matches.append(
                MatchPair(
                    fusion_idx=cand.fusion_idx,
                    report_idx=cand.report_idx,
                    fusion_id=fn.track_id,
                    report_id=rn.track_id,
                    node_score=float(cand.node_score),
                    spectral_score=float(spectral),
                    edge_score=edge_score,
                    pair_score=float(pair_score),
                    offset_m=offset_m,
                    vel_offset_mps=vel_offset,
                    heading_diff_deg=heading_diff,
                )
            )
        return matches

    # ---------------- 异常分类与投票 ----------------

    def _detect_duplicate_reports(
        self,
        fusion_nodes: List[TrackNode],
        report_nodes: List[TrackNode],
        node_sim: np.ndarray,
        match_pairs: List[MatchPair],
    ) -> List[str]:
        c = self.cfg
        if len(fusion_nodes) == 0 or len(report_nodes) <= 1:
            return []

        report_pos = np.stack([n.pos_enu for n in report_nodes], axis=0)
        report_vel = np.stack([n.vel_vec for n in report_nodes], axis=0)
        pos_dist = np.sqrt(np.maximum(_pairwise_sqdist(report_pos, report_pos), 0.0))
        vel_dist = np.sqrt(np.maximum(_pairwise_sqdist(report_vel, report_vel), 0.0))

        matched_score_by_report = {m.report_idx: m.pair_score for m in match_pairs}
        best_fusion_idx = np.argmax(node_sim, axis=0) if node_sim.size else np.array([], dtype=int)
        best_scores = node_sim[best_fusion_idx, np.arange(node_sim.shape[1])] if node_sim.size else np.array([])

        duplicates = set()
        for a in range(len(report_nodes)):
            for b in range(a + 1, len(report_nodes)):
                if pos_dist[a, b] > float(c.duplicate_report_dist_m):
                    continue
                if vel_dist[a, b] > float(c.duplicate_report_vel_mps):
                    continue
                if int(best_fusion_idx[a]) != int(best_fusion_idx[b]):
                    continue
                if float(best_scores[a]) < float(c.duplicate_node_score_threshold):
                    continue
                if float(best_scores[b]) < float(c.duplicate_node_score_threshold):
                    continue

                score_a = matched_score_by_report.get(a, float(best_scores[a]))
                score_b = matched_score_by_report.get(b, float(best_scores[b]))
                quality_a = report_nodes[a].quality
                quality_b = report_nodes[b].quality

                if (score_a, quality_a) >= (score_b, quality_b):
                    duplicates.add(report_nodes[b].track_id)
                else:
                    duplicates.add(report_nodes[a].track_id)

        return sorted(duplicates)

    def _classify_anomalies_with_votes(
        self,
        fusion_nodes: List[TrackNode],
        report_nodes: List[TrackNode],
        match_pairs: List[MatchPair],
        duplicate_reports: List[str],
        global_score: float,
    ) -> Dict[str, Any]:
        c = self.cfg
        events: List[Dict[str, Any]] = []
        current_flags = {
            '未上报': [],
            '伪报': [],
            '重复航迹': [],
            '漂移异常': [],
            '偏航异常': [],
        }
        voted = {k: [] for k in current_flags.keys()}

        match_by_fusion: Dict[str, MatchPair] = {m.fusion_id: m for m in match_pairs}
        match_by_report: Dict[str, MatchPair] = {m.report_id: m for m in match_pairs}
        duplicate_set = set(duplicate_reports)

        for fn in fusion_nodes:
            mid = match_by_fusion.get(fn.track_id)
            if mid is None:
                current_flags['未上报'].append(fn.track_id)
                info = self._update_event_vote('未上报', fn.track_id, True)
                if info['triggered']:
                    voted['未上报'].append(fn.track_id)
                    self.abnormal_entities.add(fn.track_id)
                    events.append(self._build_event('未上报', fn.track_id, info))
                self._update_event_vote('漂移异常', fn.track_id, False)
                self._update_event_vote('偏航异常', fn.track_id, False)
                continue

            self._update_event_vote('未上报', fn.track_id, False)

            pair_key = f'{mid.fusion_id}|{mid.report_id}'
            self._update_pair_metric(pair_key, mid)

            drift_now = (mid.pair_score < float(c.pair_score_threshold)) or (mid.offset_m > float(c.drift_offset_m))
            deviation_now = (
                (mid.offset_m > float(c.deviation_offset_m))
                and (mid.heading_diff_deg > float(c.deviation_heading_deg))
                and self._offset_is_growing(
                    pair_key=pair_key,
                    n_frames=int(c.deviation_growth_frames),
                    tolerance_m=float(c.offset_growth_tolerance_m),
                )
            )

            if drift_now:
                current_flags['漂移异常'].append(fn.track_id)
            if deviation_now:
                current_flags['偏航异常'].append(fn.track_id)

            drift_vote = self._update_event_vote('漂移异常', fn.track_id, drift_now)
            if drift_now and drift_vote['triggered']:
                voted['漂移异常'].append(fn.track_id)
                self.abnormal_entities.add(fn.track_id)
                events.append(
                    self._build_event(
                        '漂移异常',
                        fn.track_id,
                        drift_vote,
                        extra={'pair': [mid.fusion_id, mid.report_id], 'pair_score': mid.pair_score, 'offset_m': mid.offset_m},
                    )
                )

            deviation_vote = self._update_event_vote('偏航异常', fn.track_id, deviation_now)
            if deviation_now and deviation_vote['triggered']:
                voted['偏航异常'].append(fn.track_id)
                self.abnormal_entities.add(fn.track_id)
                events.append(
                    self._build_event(
                        '偏航异常',
                        fn.track_id,
                        deviation_vote,
                        extra={
                            'pair': [mid.fusion_id, mid.report_id],
                            'offset_m': mid.offset_m,
                            'heading_diff_deg': mid.heading_diff_deg,
                        },
                    )
                )

        for rn in report_nodes:
            rid = rn.track_id
            mid = match_by_report.get(rid)
            false_now = mid is None
            dup_now = rid in duplicate_set

            if false_now:
                current_flags['伪报'].append(rid)
            if dup_now:
                current_flags['重复航迹'].append(rid)

            false_vote = self._update_event_vote('伪报', rid, false_now)
            if false_now and false_vote['triggered']:
                voted['伪报'].append(rid)
                self.abnormal_entities.add(rid)
                events.append(self._build_event('伪报', rid, false_vote))

            dup_vote = self._update_event_vote('重复航迹', rid, dup_now)
            if dup_now and dup_vote['triggered']:
                voted['重复航迹'].append(rid)
                self.abnormal_entities.add(rid)
                events.append(self._build_event('重复航迹', rid, dup_vote))

        if global_score < float(c.global_score_threshold):
            ginfo = self._update_event_vote('全局低一致性', '__global__', True)
            if ginfo['triggered']:
                events.append(self._build_event('全局低一致性', '__global__', ginfo, extra={'global_score': float(global_score)}))
        else:
            self._update_event_vote('全局低一致性', '__global__', False)

        for k in current_flags:
            current_flags[k] = sorted(set(current_flags[k]))
            voted[k] = sorted(set(voted[k]))

        events.sort(key=lambda e: (e.get('type', ''), str(e.get('entity_id', ''))))
        return {'current_flags': current_flags, 'voted_anomalies': voted, 'events': events}

    def _update_event_vote(self, event_type: str, entity_id: str, is_true: bool) -> Dict[str, Any]:
        key = (event_type, str(entity_id))
        if key not in self.event_votes:
            self.event_votes[key] = deque(maxlen=int(self.cfg.vote_window_frames))
        dq = self.event_votes[key]
        dq.append(1 if is_true else 0)

        true_count = int(sum(dq))
        obs = int(len(dq))
        ratio = float(true_count / max(obs, 1))
        triggered = bool(
            (obs >= int(self.cfg.vote_min_obs))
            and (true_count >= int(self.cfg.vote_min_true_count))
            and (ratio >= float(self.cfg.vote_trigger_ratio))
        )
        return {'obs': obs, 'true_count': true_count, 'ratio': ratio, 'triggered': triggered}

    def _update_pair_metric(self, pair_key: str, match: MatchPair) -> None:
        if pair_key not in self.pair_metric_history:
            self.pair_metric_history[pair_key] = {
                'offset': deque(maxlen=int(self.cfg.vote_window_frames)),
                'pair_score': deque(maxlen=int(self.cfg.vote_window_frames)),
                'heading_diff': deque(maxlen=int(self.cfg.vote_window_frames)),
            }
        state = self.pair_metric_history[pair_key]
        state['offset'].append(float(match.offset_m))
        state['pair_score'].append(float(match.pair_score))
        state['heading_diff'].append(float(match.heading_diff_deg))

    def _offset_is_growing(self, pair_key: str, n_frames: int, tolerance_m: float) -> bool:
        if n_frames <= 1:
            return True
        state = self.pair_metric_history.get(pair_key)
        if not state:
            return False
        vals = list(state['offset'])
        if len(vals) < n_frames:
            return False
        tail = np.array(vals[-n_frames:], dtype=float)
        diffs = np.diff(tail)
        return bool(np.all(diffs >= -float(tolerance_m)))

    def _build_event(
        self,
        event_type: str,
        entity_id: str,
        vote_info: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            'type': event_type,
            'entity_id': entity_id,
            'vote_ratio': float(vote_info['ratio']),
            'vote_true_count': int(vote_info['true_count']),
            'vote_obs': int(vote_info['obs']),
        }
        if extra:
            event.update(extra)
        return event

    def _handle_one_side_empty(
        self,
        current_time: float,
        fusion_nodes: List[TrackNode],
        report_nodes: List[TrackNode],
    ) -> Dict[str, Any]:
        result = {
            'matches': [],
            'global_score': 0.0,
            'global_anomaly': True,
            'unreported': [],
            'false_reports': [],
            'duplicate_reports': [],
            'drift_candidates': [],
            'deviation_candidates': [],
            'events': [],
            'voted_anomalies': {
                '未上报': [],
                '伪报': [],
                '重复航迹': [],
                '漂移异常': [],
                '偏航异常': [],
            },
        }

        if fusion_nodes and not report_nodes:
            unreported = []
            voted = []
            events = []
            for fn in fusion_nodes:
                unreported.append(fn.track_id)
                info = self._update_event_vote('未上报', fn.track_id, True)
                if info['triggered']:
                    voted.append(fn.track_id)
                    self.abnormal_entities.add(fn.track_id)
                    events.append(self._build_event('未上报', fn.track_id, info))
                self._update_event_vote('漂移异常', fn.track_id, False)
                self._update_event_vote('偏航异常', fn.track_id, False)
            result['unreported'] = sorted(unreported)
            result['voted_anomalies']['未上报'] = sorted(set(voted))
            result['events'].extend(events)

        if report_nodes and not fusion_nodes:
            false_reports = []
            voted = []
            events = []
            for rn in report_nodes:
                false_reports.append(rn.track_id)
                info = self._update_event_vote('伪报', rn.track_id, True)
                if info['triggered']:
                    voted.append(rn.track_id)
                    self.abnormal_entities.add(rn.track_id)
                    events.append(self._build_event('伪报', rn.track_id, info))
                self._update_event_vote('重复航迹', rn.track_id, False)
            result['false_reports'] = sorted(false_reports)
            result['voted_anomalies']['伪报'] = sorted(set(voted))
            result['events'].extend(events)

        ginfo = self._update_event_vote('全局低一致性', '__global__', True)
        if ginfo['triggered']:
            result['events'].append(self._build_event('全局低一致性', '__global__', ginfo, extra={'time': current_time}))

        return result

    # ---------------- 打印/序列化 ----------------

    def _match_to_dict(self, m: MatchPair) -> Dict[str, Any]:
        return {
            'fusion_id': m.fusion_id,
            'report_id': m.report_id,
            'node_score': float(m.node_score),
            'edge_score': float(m.edge_score),
            'pair_score': float(m.pair_score),
            'spectral_score': float(m.spectral_score),
            'offset_m': float(m.offset_m),
            'vel_offset_mps': float(m.vel_offset_mps),
            'heading_diff_deg': float(m.heading_diff_deg),
        }

    def _print_frame_result(self, result: Dict[str, Any]) -> None:
        t = result.get('time')
        print(
            f'\n[Frame {t}] matches={len(result.get("matches", []))} '
            f'global_score={result.get("global_score", 0.0):.3f}'
        )
        if result.get('unreported'):
            print('  未上报:', result['unreported'])
        if result.get('false_reports'):
            print('  伪报:', result['false_reports'])
        if result.get('duplicate_reports'):
            print('  重复航迹:', result['duplicate_reports'])
        if result.get('drift_candidates'):
            print('  漂移候选:', result['drift_candidates'])
        if result.get('deviation_candidates'):
            print('  偏航候选:', result['deviation_candidates'])
        if result.get('events'):
            print('  投票触发事件:', result['events'])


# ================================================================
#  主程序（CSV批处理）
# ================================================================


def main() -> None:
    cfg = MatchConfig(CONFIG)
    detector = GraphTrackAnomalyDetector(cfg)

    df_fusion = pd.read_csv(cfg.fusion_csv_path)
    df_report = pd.read_csv(cfg.report_csv_path)

    results = detector.process_stream(df_fusion, df_report)
    abnormal_entities = sorted(detector.abnormal_entities)

    print('\n================ 图特征匹配异常识别完成 ================')
    print(f'处理帧数: {len(results)}')
    print(f'累计异常实体数: {len(abnormal_entities)}')
    if abnormal_entities:
        print('异常实体:', abnormal_entities)

    if cfg.output_json_path:
        payload = {
            'summary': {
                'num_frames': len(results),
                'abnormal_entities': abnormal_entities,
            },
            'frames': results,
        }
        with open(cfg.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)
        print(f'结果已写入: {cfg.output_json_path}')


if __name__ == '__main__':
    main()
