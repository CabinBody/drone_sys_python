import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==============================================================
# CONFIG
# ==============================================================
DATA_ROOT = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multi_source_5000x120"

WINDOW_SIZE = 20
STRIDE = 5
MODALITIES = ["gps", "radar", "5g_a", "tdoa"]
MODALITY_TO_ID = {m: i for i, m in enumerate(MODALITIES)}
N_MODALITIES = len(MODALITIES)

R = 6378137.0
R_EARTH = 6378137.0

# feat: east/north/up + vx/vy/vz + speed + conf + t_norm + modality_one_hot(4)
NODE_FEAT_DIM = 9 + N_MODALITIES
NORM_STATS_PATH = "graph_norm_stats_sparse_enu.pth"


# ==============================================================
# 经纬度 <-> ENU
# ==============================================================
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east = R * dlon * np.cos(np.radians(lat0))
    north = R * dlat
    up = alt - alt0
    return east, north, up


def to_enu_single_point(lat, lon, alt, lat0, lon0, alt0):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)

    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r

    east = dlon * np.cos(lat0_r) * R_EARTH
    north = dlat * R_EARTH
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float32)


def enu_to_llh(east, north, up, lat0, lon0, alt0):
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)

    lat = north / R_EARTH + lat0_r
    lon = east / (R_EARTH * np.cos(lat0_r)) + lon0_r
    alt = up + alt0
    return np.degrees(lat), np.degrees(lon), alt


# ==============================================================
# 单窗口构造: 供推理或调试
# ==============================================================
def build_input_feature(df_truth: pd.DataFrame, mod_frames: Dict[str, pd.DataFrame]):
    """
    将一个 UAV 的完整序列转换为可变节点表示（不做滑窗）。
    返回 dict:
      node_feat: [N,F]
      node_t: [N]
      node_m: [N]
      y_true: [T,3]
      obs_json: List[Dict]
    """
    df_truth = df_truth.sort_values("timestamp").reset_index(drop=True)
    if len(df_truth) == 0:
        return {
            "node_feat": np.zeros((0, NODE_FEAT_DIM), dtype=np.float32),
            "node_t": np.zeros((0,), dtype=np.int64),
            "node_m": np.zeros((0,), dtype=np.int64),
            "y_true": np.zeros((0, 3), dtype=np.float32),
            "obs_json": [],
        }

    lat0 = df_truth.iloc[0]["lat"]
    lon0 = df_truth.iloc[0]["lon"]
    alt0 = df_truth.iloc[0]["alt"]

    e_t, n_t, u_t = latlon_to_enu(
        df_truth["lat"].values, df_truth["lon"].values, df_truth["alt"].values, lat0, lon0, alt0
    )
    y_true = np.stack([e_t, n_t, u_t], axis=-1).astype(np.float32)

    ts_to_row = {}
    for name in MODALITIES:
        df_m = mod_frames.get(name)
        if df_m is None or len(df_m) == 0:
            ts_to_row[name] = {}
            continue
        cols = ["timestamp", "lat", "lon", "alt", "source_conf"]
        if not all(c in df_m.columns for c in cols):
            ts_to_row[name] = {}
            continue
        ts_to_row[name] = {
            int(r.timestamp): r
            for r in df_m[cols].itertuples(index=False)
        }

    vx = df_truth["vx"].values.astype(np.float32)
    vy = df_truth["vy"].values.astype(np.float32)
    vz = df_truth["vz"].values.astype(np.float32)
    spd = df_truth["speed"].values.astype(np.float32)

    feats: List[np.ndarray] = []
    t_ids: List[int] = []
    m_ids: List[int] = []
    obs_json: List[Dict] = []

    T = len(df_truth)
    denom = max(T - 1, 1)

    for ti, row_t in enumerate(df_truth.itertuples(index=False)):
        ts = int(row_t.timestamp)
        item = {}

        for m_name in MODALITIES:
            r = ts_to_row[m_name].get(ts)
            if r is None:
                continue

            east, north, up = latlon_to_enu(r.lat, r.lon, r.alt, lat0, lon0, alt0)
            feat = np.zeros((NODE_FEAT_DIM,), dtype=np.float32)
            feat[0:3] = np.array([east, north, up], dtype=np.float32)
            feat[3] = vx[ti]
            feat[4] = vy[ti]
            feat[5] = vz[ti]
            feat[6] = spd[ti]
            feat[7] = float(r.source_conf)
            feat[8] = float(ti / denom)
            feat[9 + MODALITY_TO_ID[m_name]] = 1.0

            feats.append(feat)
            t_ids.append(ti)
            m_ids.append(MODALITY_TO_ID[m_name])

            item[m_name] = {
                "lat": float(r.lat),
                "lon": float(r.lon),
                "alt": float(r.alt),
                "source_conf": float(r.source_conf),
            }

        obs_json.append(item)

    if len(feats) == 0:
        node_feat = np.zeros((0, NODE_FEAT_DIM), dtype=np.float32)
        node_t = np.zeros((0,), dtype=np.int64)
        node_m = np.zeros((0,), dtype=np.int64)
    else:
        node_feat = np.stack(feats).astype(np.float32)
        node_t = np.array(t_ids, dtype=np.int64)
        node_m = np.array(m_ids, dtype=np.int64)

    return {
        "node_feat": node_feat,
        "node_t": node_t,
        "node_m": node_m,
        "y_true": y_true,
        "obs_json": obs_json,
    }


# ==============================================================
class MultiSourceGraphDataset(Dataset):
    """
    可变模态稀疏图数据集。

    __getitem__ 返回 dict:
      node_feat: [N,F]
      node_t: [N]
      node_m: [N]
      y: [T,3]
      obs_json: List[Dict]  # 每个时间步的动态模态字段
    """

    def __init__(self, data_root=DATA_ROOT, window_size=WINDOW_SIZE, stride=STRIDE):
        super().__init__()
        self.data_root = data_root
        self.window_size = window_size
        self.stride = stride

        samples = self._build()
        if len(samples) == 0:
            raise RuntimeError(f"No samples found in {data_root}")

        # 归一化统计
        if os.path.exists(NORM_STATS_PATH):
            stats = torch.load(NORM_STATS_PATH)
            x_mean = stats["x_mean"].cpu().numpy().astype(np.float32)
            x_std = stats["x_std"].cpu().numpy().astype(np.float32)
            y_mean = stats["y_mean"].cpu().numpy().astype(np.float32)
            y_std = stats["y_std"].cpu().numpy().astype(np.float32)
        else:
            x_all = np.concatenate([s["node_feat"] for s in samples if len(s["node_feat"]) > 0], axis=0)
            x_mean = x_all.mean(0)
            x_std = x_all.std(0) + 1e-6

            # t_norm + one-hot 不做标准化
            x_mean[8:] = 0
            x_std[8:] = 1

            y_all = np.concatenate([s["y"] for s in samples], axis=0)
            y_mean = y_all.mean(0)
            y_std = y_all.std(0) + 1e-6

            torch.save(
                {
                    "x_mean": torch.tensor(x_mean),
                    "x_std": torch.tensor(x_std),
                    "y_mean": torch.tensor(y_mean),
                    "y_std": torch.tensor(y_std),
                },
                NORM_STATS_PATH,
            )

        # 应用标准化
        for s in samples:
            if len(s["node_feat"]) > 0:
                s["node_feat"] = (s["node_feat"] - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
            s["y"] = (s["y"] - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)

        self.samples = samples
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.tensor(x_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.tensor(y_std, dtype=torch.float32)

        print(f"[SparseDataset ENU] Loaded {len(self.samples)} samples.")

    def _read_mod_csv(self, bdir, fname):
        path = os.path.join(bdir, fname)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["timestamp", "id", "lat", "lon", "alt", "source_conf"])
        return pd.read_csv(path)

    def _build(self):
        samples = []

        for bdir in sorted(glob.glob(os.path.join(self.data_root, "batch*"))):
            truth = pd.read_csv(os.path.join(bdir, "truth.csv"))
            dfs = {
                "gps": self._read_mod_csv(bdir, "gps.csv"),
                "radar": self._read_mod_csv(bdir, "radar.csv"),
                "5g_a": self._read_mod_csv(bdir, "5g_a.csv"),
                "tdoa": self._read_mod_csv(bdir, "tdoa.csv"),
            }

            for uav in sorted(truth["id"].unique()):
                df_t = truth[truth["id"] == uav].copy().sort_values("timestamp").reset_index(drop=True)
                if len(df_t) < self.window_size:
                    continue

                lat0 = df_t.iloc[0]["lat"]
                lon0 = df_t.iloc[0]["lon"]
                alt0 = df_t.iloc[0]["alt"]

                e_t, n_t, u_t = latlon_to_enu(
                    df_t["lat"].values, df_t["lon"].values, df_t["alt"].values, lat0, lon0, alt0
                )

                # 预构建 timestamp -> row 映射，避免滑窗内重复过滤
                mod_maps = {}
                for m_name in MODALITIES:
                    df_m = dfs[m_name]
                    df_m_u = df_m[df_m["id"] == uav]
                    cols = ["timestamp", "lat", "lon", "alt", "source_conf"]
                    if len(df_m_u) == 0 or not all(c in df_m_u.columns for c in cols):
                        mod_maps[m_name] = {}
                    else:
                        mod_maps[m_name] = {
                            int(r.timestamp): r for r in df_m_u[cols].itertuples(index=False)
                        }

                vx_all = df_t["vx"].values.astype(np.float32)
                vy_all = df_t["vy"].values.astype(np.float32)
                vz_all = df_t["vz"].values.astype(np.float32)
                spd_all = df_t["speed"].values.astype(np.float32)
                ts_all = df_t["timestamp"].values.astype(np.int64)

                T_total = len(df_t)
                for s in range(0, T_total - self.window_size + 1, self.stride):
                    e_win = e_t[s:s + self.window_size]
                    n_win = n_t[s:s + self.window_size]
                    u_win = u_t[s:s + self.window_size]
                    y = np.stack([e_win, n_win, u_win], axis=-1).astype(np.float32)

                    vx = vx_all[s:s + self.window_size]
                    vy = vy_all[s:s + self.window_size]
                    vz = vz_all[s:s + self.window_size]
                    spd = spd_all[s:s + self.window_size]
                    ts_win = ts_all[s:s + self.window_size]

                    feats = []
                    t_ids = []
                    m_ids = []
                    obs_json = []

                    denom = max(self.window_size - 1, 1)
                    for ti in range(self.window_size):
                        ts = int(ts_win[ti])
                        item = {}

                        for m_name in MODALITIES:
                            row = mod_maps[m_name].get(ts)
                            if row is None:
                                continue

                            east, north, up = latlon_to_enu(row.lat, row.lon, row.alt, lat0, lon0, alt0)
                            feat = np.zeros((NODE_FEAT_DIM,), dtype=np.float32)
                            feat[0:3] = np.array([east, north, up], dtype=np.float32)
                            feat[3] = vx[ti]
                            feat[4] = vy[ti]
                            feat[5] = vz[ti]
                            feat[6] = spd[ti]
                            feat[7] = float(row.source_conf)
                            feat[8] = float(ti / denom)
                            feat[9 + MODALITY_TO_ID[m_name]] = 1.0

                            feats.append(feat)
                            t_ids.append(ti)
                            m_ids.append(MODALITY_TO_ID[m_name])

                            item[m_name] = {
                                "lat": float(row.lat),
                                "lon": float(row.lon),
                                "alt": float(row.alt),
                                "source_conf": float(row.source_conf),
                            }

                        obs_json.append(item)

                    if len(feats) == 0:
                        continue

                    sample = {
                        "node_feat": np.stack(feats).astype(np.float32),
                        "node_t": np.array(t_ids, dtype=np.int64),
                        "node_m": np.array(m_ids, dtype=np.int64),
                        "y": y,
                        "obs_json": obs_json,
                    }
                    samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "node_feat": torch.tensor(s["node_feat"], dtype=torch.float32),
            "node_t": torch.tensor(s["node_t"], dtype=torch.long),
            "node_m": torch.tensor(s["node_m"], dtype=torch.long),
            "y": torch.tensor(s["y"], dtype=torch.float32),
            "obs_json": s["obs_json"],
        }


def sparse_collate_fn(batch):
    """
    将可变节点批次 pad 成张量:
      node_feat: [B, Lmax, F]
      node_t: [B, Lmax]
      node_m: [B, Lmax]
      node_mask: [B, Lmax]
      y: [B, T, 3]
    """
    bsz = len(batch)
    feat_dim = batch[0]["node_feat"].shape[-1]
    t_len = batch[0]["y"].shape[0]
    lmax = max(x["node_feat"].shape[0] for x in batch)

    node_feat = torch.zeros((bsz, lmax, feat_dim), dtype=torch.float32)
    node_t = torch.full((bsz, lmax), -1, dtype=torch.long)
    node_m = torch.full((bsz, lmax), -1, dtype=torch.long)
    node_mask = torch.zeros((bsz, lmax), dtype=torch.float32)
    y = torch.zeros((bsz, t_len, 3), dtype=torch.float32)

    obs_json = []
    for i, item in enumerate(batch):
        n = item["node_feat"].shape[0]
        node_feat[i, :n] = item["node_feat"]
        node_t[i, :n] = item["node_t"]
        node_m[i, :n] = item["node_m"]
        node_mask[i, :n] = 1.0
        y[i] = item["y"]
        obs_json.append(item["obs_json"])

    return {
        "node_feat": node_feat,
        "node_t": node_t,
        "node_m": node_m,
        "node_mask": node_mask,
        "y": y,
        "obs_json": obs_json,
    }
