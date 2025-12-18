# ================================================================
# dataset.py —— Multi-source dataset (one sample = ALL UAV in window)
# ================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ================================================================
# Config
# ================================================================
USE_ENU      = True
DO_NORMALIZE = True
SEQ_LEN      = 20

TRAJ_FEAT_DIM = 7      # pos3 + vel3 + speed1
TRAJ_KEYS     = ["radar", "5g_a", "tdoa"]
EXIST_KEYS    = ["em", "acoustic"]

R_EARTH = 6378137.0


# ================================================================
# 经纬度 -> ENU
# ================================================================
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    mx = np.radians((lat + lat0) / 2)
    east  = R_EARTH * dlon * np.cos(mx)
    north = R_EARTH * dlat
    up    = alt - alt0
    return east, north, up


# ================================================================
# Dataset（一个样本 = 所有 UAV × 一个时间窗）
# ================================================================
class MultiSourceDataset(Dataset):
    def __init__(self, batch_dir, seq_len=SEQ_LEN):
        super().__init__()
        self.batch_dir = batch_dir
        self.seq_len = seq_len

        # ------------------ 载入 truth ------------------
        df_truth = pd.read_csv(os.path.join(batch_dir, "truth.csv"))
        self.uav_ids = sorted(df_truth["id"].unique().tolist())
        self.timestamps = sorted(df_truth["timestamp"].unique())
        self.T = len(self.timestamps)
        self.N = len(self.uav_ids)

        # ENU origin
        self.lat0 = df_truth["lat"].iloc[0]
        self.lon0 = df_truth["lon"].iloc[0]
        self.alt0 = df_truth["alt"].iloc[0]

        # truth → (T,N,3)
        self.gt = self._load_truth(df_truth)

        # ------------------ 多源轨迹 ------------------
        self.traj = self._load_traj_sources()

        # ------------------ 角度源 ------------------
        self.exist = self._load_exist_sources()

        # ------------------ 归一化 ------------------
        if DO_NORMALIZE:
            self._compute_norm()
            self._apply_norm()

        # 样本数量 = 滑窗数量
        self.num_samples = self.T - seq_len

        print(f"[Dataset] UAV={self.N}, T={self.T}, Samples={self.num_samples}")

    # ================================================================
    def _load_truth(self, df):
        if USE_ENU:
            df["east"], df["north"], df["up"] = latlon_to_enu(
                df["lat"], df["lon"], df["alt"],
                self.lat0, self.lon0, self.alt0
            )
            cols = ["east","north","up"]
        else:
            cols = ["lat","lon","alt"]

        pv = df.pivot_table(index="timestamp", columns="id", values=cols)
        return pv.values.reshape(self.T, self.N, 3)

    # ================================================================
    def _load_traj_sources(self):
        out = {}

        for key in TRAJ_KEYS:
            fname = "5g_a.csv" if key == "5g_a" else f"{key}.csv"
            df = pd.read_csv(os.path.join(self.batch_dir, fname))

            # ENU 或 lat/lon/alt
            if USE_ENU:
                df["east"], df["north"], df["up"] = latlon_to_enu(
                    df["lat"], df["lon"], df["alt"],
                    self.lat0, self.lon0, self.alt0
                )
                cols = ["east","north","up","vx","vy","vz","speed"]
            else:
                cols = ["lat","lon","alt","vx","vy","vz","speed"]

            feat = df.pivot_table(index="timestamp", columns="id", values=cols).values
            conf = df.pivot_table(index="timestamp", columns="id", values="source_conf").values

            feat = np.nan_to_num(feat.reshape(self.T, self.N, TRAJ_FEAT_DIM))
            conf = np.nan_to_num(conf.reshape(self.T, self.N))
            mask = (conf > 0).astype(np.float32)

            out[key] = {
                "feat": feat,     # (T,N,7)
                "conf": conf,     # (T,N)
                "mask": mask      # (T,N)
            }

        return out

    # ================================================================
    def _load_exist_sources(self):
        out = {}

        for key in EXIST_KEYS:
            df = pd.read_csv(os.path.join(self.batch_dir, f"{key}.csv"))

            angle = df.pivot_table(index="timestamp", columns="id", values="angle_deg").values
            conf  = df.pivot_table(index="timestamp", columns="id", values="source_conf").values

            out[key] = {
                "angle": np.nan_to_num(angle.reshape(self.T, self.N), nan=-1.0),
                "conf":  np.nan_to_num(conf.reshape(self.T, self.N), nan=0.0),
                "mask":  (conf.reshape(self.T, self.N) > 0).astype(np.float32),
            }

        return out

    # ================================================================
    def _compute_norm(self):
        # 全模态合并计算
        big = []
        for key in TRAJ_KEYS:
            big.append(self.traj[key]["feat"])
        big = np.concatenate(big, axis=1).reshape(-1, TRAJ_FEAT_DIM)

        self.mean = big.mean(0)
        self.std  = big.std(0) + 1e-6

    # ================================================================
    def _apply_norm(self):
        # 轨迹源归一化
        for key in TRAJ_KEYS:
            self.traj[key]["feat"] = (self.traj[key]["feat"] - self.mean) / self.std

        # truth 也归一化
        g = self.gt.reshape(-1,3)
        m = g.mean(0)
        s = g.std(0) + 1e-6

        self.gt = (self.gt - m) / s

    # ================================================================
    def __len__(self):
        return self.num_samples

    # ================================================================
    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + self.seq_len

        # ---------------- truth ----------------
        gt = torch.tensor(self.gt[t0:t1], dtype=torch.float32)    # (T,N,3)

        # ---------------- 轨迹源 ----------------
        feats, confs, masks = [], [], []

        for key in TRAJ_KEYS:
            feats.append(self.traj[key]["feat"][t0:t1])   # (T,N,7)
            confs.append(self.traj[key]["conf"][t0:t1])   # (T,N)
            masks.append(self.traj[key]["mask"][t0:t1])   # (T,N)

        traj_obs  = torch.tensor(np.stack(feats,axis=2), dtype=torch.float32)  # (T,N,3,7)
        traj_conf = torch.tensor(np.stack(confs,axis=2), dtype=torch.float32)  # (T,N,3)
        traj_mask = torch.tensor(np.stack(masks,axis=2), dtype=torch.float32)  # (T,N,3)

        # ---------------- 存在性源 ----------------
        angles, econfs, emasks = [], [], []

        for key in EXIST_KEYS:
            angles.append(self.exist[key]["angle"][t0:t1])
            econfs.append(self.exist[key]["conf"][t0:t1])
            emasks.append(self.exist[key]["mask"][t0:t1])

        angle = torch.tensor(np.stack(angles,axis=2), dtype=torch.float32)  # (T,N,2)
        econf = torch.tensor(np.stack(econfs,axis=2), dtype=torch.float32)
        emask = torch.tensor(np.stack(emasks,axis=2), dtype=torch.float32)

        return {
            "traj_sources": {
                "obs": traj_obs,      # (T,N,3,7)
                "conf": traj_conf,    # (T,N,3)
                "mask": traj_mask
            },
            "exist_sources": {
                "angle": angle,       # (T,N,2)
                "conf":  econf,
                "mask":  emask
            },
            "gt": gt                 # (T,N,3)
        }
