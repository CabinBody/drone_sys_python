# ================================================================
#  🚀 UAV Track Graph Matching & Anomaly Detection
#  Author: ChatGPT (for 尹睿杰)
#  Function: 低空无人机探测轨迹与上报轨迹一致性识别 + 异常检测
# ================================================================

import math
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Optional
import torch

# ==============================
# 🔧 全局配置参数
# ==============================

CONFIG = {
    # --- 模型路径（已训练好的融合模型） ---
    "MODEL_PATH": "pt_backup/fusion_model.pt",   # 你的权重文件
    "NORM_PATH": "pt_backup/fusion_norm.pth",     # 归一化参数路径
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # --- 输入数据路径 ---
    "DET_CSV_PATH": "data/fusion_output.csv",     # 在线融合结果（探测）
    "REP_CSV_PATH": "data/report_stream.csv",     # 上报轨迹

    # --- 候选门控参数 ---
    "R_GATE_M": 600.0,     # 空间半径门 (米)
    "TAU_T": 1.0,          # 时间门限 (秒)

    # --- 自适应阈值参数 ---
    "KEEP_RATIO": 0.20,    # 保留比例 r∈(0,1)
    "EMA_BETA": 0.80,      # EMA平滑系数

    # --- 候选与核参数 ---
    "TOPK": 5,             # 每个节点候选数
    "POS_SIGMA_M": 120.0,  # 位置核sigma
    "VEL_SIGMA": 20.0,     # 速度核sigma
    "T_SIGMA": 0.6,        # 时间核sigma
    "LAMBDA_A": 1.0,       # 置信度指数
    "LAMBDA_B": 1.0,       # 通信质量指数


    # --- 边一致性核 ---
    "EDGE_POS_SIGMA_M": 180.0,
    "EDGE_VEL_SIGMA": 25.0,
    "EDGE_LAMBDA": 0.6,    # 边项增益
    "DET_KNN": 6,
    "REP_KNN": 6,

    # --- 异常判断 ---
    "TAU_NODE": 0.25,      # 节点匹配强度阈值
    "D_MAX_M": 120.0,      # 航线走廊偏离距离阈
    "THOLD": 3,            # 连续偏离判定阈
    "L_VOTE": 3,           # 连续帧异常投票数

    # --- 其他 ---
    "PRINT_DETAIL": False, # 是否打印每帧结果
}

# ================================================================
#  以下为算法实现（与配置独立）
# ================================================================

EARTH_R = 6378137.0

def latlon_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    east  = dlon * EARTH_R * math.cos(math.radians(ref_lat))
    north = dlat * EARTH_R
    up    = alt - ref_alt
    return np.stack([east, north, up], axis=1)

def pairwise_l2(a, b, inv_sigma2):
    N, d = a.shape
    M, _ = b.shape
    aa = np.sum(a*a, axis=1, keepdims=True)
    bb = np.sum(b*b, axis=1, keepdims=True).T
    dist2 = np.maximum(aa + bb - 2*a@b.T, 0.0)
    return np.exp(-0.5 * dist2 * inv_sigma2), dist2

def comm_quality(snr, rssi, delay, coverage):
    snr_n = np.clip((snr - 0.0) / 30.0, 0.0, 1.0)
    rssi_n = np.clip((rssi + 90.0) / 50.0, 0.0, 1.0)
    delay_n = np.clip(1.0 - (delay / 150.0), 0.0, 1.0)
    cov_n = np.clip(coverage, 0.0, 1.0)
    eps = 1e-6
    return np.power((snr_n+eps)*(rssi_n+eps)*(delay_n+eps)*(cov_n+eps), 0.25)

def softmax_row(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

class MatchConfig:
    def __init__(self, cfg: dict):
        for k, v in cfg.items():
            setattr(self, k.lower(), v)

class GateState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha_hist = deque(maxlen=5000)
        self.q_hist = deque(maxlen=5000)
        self.tau_alpha = 0.5
        self.tau_q = 0.5
    def update(self, alphas, qs):
        self.alpha_hist.extend(list(alphas))
        self.q_hist.extend(list(qs))
        qa = np.quantile(np.asarray(self.alpha_hist), 1.0 - self.cfg.keep_ratio)
        qq = np.quantile(np.asarray(self.q_hist), 1.0 - self.cfg.keep_ratio)
        b = self.cfg.ema_beta
        self.tau_alpha = b*self.tau_alpha + (1-b)*qa
        self.tau_q = b*self.tau_q + (1-b)*qq

def node_kernel(Pi, Pa, Vi, Va, alpha_i, q_a, dt, cfg):
    inv_sig_p2 = 1.0 / (cfg.pos_sigma_m**2)
    inv_sig_v2 = 1.0 / (cfg.vel_sigma**2)
    inv_sig_t2 = 1.0 / (cfg.t_sigma**2)
    kp = math.exp(-0.5 * np.sum((Pi-Pa)**2) * inv_sig_p2)
    kv = math.exp(-0.5 * np.sum((Vi-Va)**2) * inv_sig_v2)
    kt = math.exp(-0.5 * (dt**2) * inv_sig_t2)
    return kp * kv * kt * (alpha_i**cfg.lambda_a) * (q_a**cfg.lambda_b)

def edge_kernel(dP_ij, dP_ab, dV_ij, dV_ab, cfg):
    inv_sig_dp2 = 1.0 / (cfg.edge_pos_sigma_m**2)
    inv_sig_dv2 = 1.0 / (cfg.edge_vel_sigma**2)
    kp = math.exp(-0.5 * np.sum((dP_ij - dP_ab)**2) * inv_sig_dp2)
    kv = math.exp(-0.5 * np.sum((dV_ij - dV_ab)**2) * inv_sig_dv2)
    return kp * kv

def knn_indices(X, k):
    if len(X) == 0:
        return []
    _, dist2 = pairwise_l2(X, X, inv_sigma2=1.0)
    np.fill_diagonal(dist2, np.inf)
    return np.argsort(dist2, axis=1)[:, :min(k, X.shape[0]-1)].tolist()

class FrameMatcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gate = GateState(cfg)
        self.miss_det = defaultdict(int)
        self.miss_rep = defaultdict(int)
        self.AbnormalQueue = set()

    def process(self, df_det, df_rep):
        cfg = self.cfg
        if len(df_det)==0 or len(df_rep)==0:
            return {}
        ref_lat = df_det['lat'].mean()
        ref_lon = df_det['lon'].mean()
        ref_alt = df_det['alt'].mean()

        P_det = latlon_to_enu(df_det['lat'], df_det['lon'], df_det['alt'], ref_lat, ref_lon, ref_alt)
        P_rep = latlon_to_enu(df_rep['lat'], df_rep['lon'], df_rep['alt'], ref_lat, ref_lon, ref_alt)
        V_det = df_det[['vx','vy']].to_numpy()
        V_rep = df_rep[['vx','vy']].to_numpy()

        alpha = df_det['confidence'].to_numpy(dtype=float)
        snr = df_rep['snr'].to_numpy(dtype=float)
        rssi = df_rep['rssi'].to_numpy(dtype=float)
        delay = df_rep['delay'].to_numpy(dtype=float)
        coverage = df_rep['coverage'].to_numpy(dtype=float)

        q = comm_quality(snr, rssi, delay, coverage)
        q = np.asarray(q, dtype=float)  # 保证是 numpy 数组

        self.gate.update(alpha, q)
        tau_alpha, tau_q = self.gate.tau_alpha, self.gate.tau_q

        N, M = len(df_det), len(df_rep)
        _, dist2 = pairwise_l2(P_det, P_rep, inv_sigma2=1.0)
        dist = np.sqrt(dist2)

        s_geo = np.exp(-0.5 * (dist**2) / (cfg.pos_sigma_m**2))
        mask = (dist <= cfg.r_gate_m)
        det_ok = np.asarray((alpha >= tau_alpha).astype(float))
        rep_ok = np.asarray((q >= tau_q).astype(float))
        s_geo *= mask * det_ok[:, None] * rep_ok[None, :]

        # TopK 候选
        Knode = np.zeros((N,M))
        for i in range(N):
            top_idx = np.argsort(-s_geo[i])[:cfg.topk]
            for a in top_idx:
                Knode[i,a] = node_kernel(P_det[i], P_rep[a], V_det[i], V_rep[a], alpha[i], q[a], 0, cfg)

        # 贪心匹配
        flat = [(Knode[i,a], i,a) for i in range(N) for a in range(M) if Knode[i,a]>0]
        flat.sort(reverse=True)
        used_i, used_a, matches = set(), set(), []
        for s,i,a in flat:
            if i not in used_i and a not in used_a:
                used_i.add(i); used_a.add(a)
                matches.append((df_det.iloc[i]['id'], df_rep.iloc[a]['id'], s))

        # 异常判断
        unreported = [df_det.iloc[i]['id'] for i in range(N) if Knode[i].max() < cfg.tau_node]
        undetected = [df_rep.iloc[a]['id'] for a in range(M) if Knode[:,a].max() < cfg.tau_node]

        for u in unreported:
            self.miss_rep[u]+=1
            if self.miss_rep[u]>=cfg.l_vote:
                self.AbnormalQueue.add(u)
        for u in undetected:
            self.miss_det[u]+=1
            if self.miss_det[u]>=cfg.l_vote:
                self.AbnormalQueue.add(u)

        return {
            "matches": matches,
            "unreported": unreported,
            "undetected": undetected,
            "AbnormalQueue": list(self.AbnormalQueue),
            "tau_alpha": tau_alpha,
            "tau_q": tau_q
        }

