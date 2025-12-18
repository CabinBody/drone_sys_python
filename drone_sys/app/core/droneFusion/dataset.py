# dataset.py  —— 完全 ENU 版本

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==============================================================
# ⚙ CONFIG
# ==============================================================
DATA_ROOT = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multi_source_5000x120"

WINDOW_SIZE = 20
STRIDE      = 5
MODALITIES = ["radar", "5g_a", "tdoa"]
N_MODALITIES = 3

R = 6378137.0  # earth radius

# ENU 输入维度（12维：east/north/up + vx/vy/vz + speed + conf + t_norm + one-hot）
NODE_FEAT_DIM = 12
NORM_STATS_PATH = "graph_norm_stats_enu.pth"

# ================================
# 🌍 地球模型参数（WGS-84）
# ================================
R_EARTH = 6378137.0
f = 1 / 298.257223563
e2 = f * (2 - f)        # 第一偏心率平方


def build_input_feature(df_truth, df_radar, df_5g, df_tdoa):
    """
    构建 inference 输入:
    输出:
      X      [T,3,12]
      M      [T,3]
      Y_true [T,3]  # ENU
    """

    T = len(df_truth)
    NODE_FEAT_DIM = 12

    X = np.zeros((T, 3, NODE_FEAT_DIM), dtype=np.float32)
    M = np.zeros((T, 3), dtype=np.float32)

    # Truth ENU
    Y_true = df_truth[["east", "north", "up"]].values.astype(np.float32)

    vx = df_truth["vx"].values
    vy = df_truth["vy"].values
    vz = df_truth["vz"].values
    spd = df_truth["speed"].values
    t_norm = np.linspace(0, 1, T)

    sources = [
        (0, df_radar),
        (1, df_5g),
        (2, df_tdoa),
    ]

    for mi, df_m in sources:
        mask = (~df_m["lat"].isna()).astype(np.float32).values

        X[:, mi, 0] = df_m["east"].fillna(0).values
        X[:, mi, 1] = df_m["north"].fillna(0).values
        X[:, mi, 2] = df_m["up"].fillna(0).values

        X[:, mi, 3] = vx
        X[:, mi, 4] = vy
        X[:, mi, 5] = vz
        X[:, mi, 6] = spd

        X[:, mi, 7] = df_m["source_conf"].fillna(0).values.astype(np.float32)
        X[:, mi, 8] = t_norm

        # one-hot source type
        if mi == 0: X[:, mi, 9]  = 1      # radar
        if mi == 1: X[:, mi, 10] = 1      # 5g
        if mi == 2: X[:, mi, 11] = 1      # tdoa

        M[:, mi] = mask

    return X, M, Y_true



# ==============================================================
# 经纬度 → ENU 转换
# ==============================================================
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east  = R * dlon * np.cos(np.radians(lat0))
    north = R * dlat
    up    = alt - alt0
    return east, north, up

def to_enu_single_point(lat, lon, alt, lat0, lon0, alt0):
    lat_r  = np.radians(lat)
    lon_r  = np.radians(lon)
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)

    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r

    east  = dlon * np.cos(lat0_r) * R_EARTH
    north = dlat * R_EARTH
    up    = alt - alt0

    return np.array([east, north, up], dtype=np.float32)


def enu_to_llh(east, north, up, lat0, lon0, alt0):
    """
    ENU → 经纬度 LLH
    east, north, up 为 numpy 数组
    """
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)

    lat = north / R_EARTH + lat0_r
    lon = east  / (R_EARTH * np.cos(lat0_r)) + lon0_r
    alt = up + alt0

    return np.degrees(lat), np.degrees(lon), alt



# ==============================================================
class MultiSourceGraphDataset(Dataset):
    """
    输出:
    X: [N, T, 3, F]
    Y: [N, T, 3]        # ENU truth
    M: [N, T, 3]        # mask
    """
    def __init__(self, data_root=DATA_ROOT, window_size=WINDOW_SIZE, stride=STRIDE):
        super().__init__()
        self.data_root = data_root
        self.window_size = window_size
        self.stride = stride

        X_list, Y_list, M_list = self._build()

        X = np.stack(X_list)   # [N, T, M, F]
        Y = np.stack(Y_list)   # [N, T, 3]
        M = np.stack(M_list)   # [N, T, M]

        # ==================================================
        # 归一化
        # ==================================================
        if os.path.exists(NORM_STATS_PATH):
            stats = torch.load(NORM_STATS_PATH)

            # ⭐ 保证全部是 numpy
            x_mean = stats["x_mean"].cpu().numpy().astype(np.float32)
            x_std = stats["x_std"].cpu().numpy().astype(np.float32)
            y_mean = stats["y_mean"].cpu().numpy().astype(np.float32)
            y_std = stats["y_std"].cpu().numpy().astype(np.float32)

        else:
            X_flat = X.reshape(-1, NODE_FEAT_DIM)
            x_mean = X_flat.mean(0)
            x_std  = X_flat.std(0) + 1e-6

            # non-continuous 不归一化
            x_mean[8:] = 0
            x_std[8:]  = 1

            Y_flat = Y.reshape(-1, 3)
            y_mean = Y_flat.mean(0)
            y_std  = Y_flat.std(0) + 1e-6

            torch.save({
                "x_mean": torch.tensor(x_mean),
                "x_std":  torch.tensor(x_std),
                "y_mean": torch.tensor(y_mean),
                "y_std":  torch.tensor(y_std),
            }, NORM_STATS_PATH)

        # 做标准化
        X = (X - x_mean.reshape(1, 1, 1, -1)) / x_std.reshape(1, 1, 1, -1)
        Y = (Y - y_mean.reshape(1, 1, -1)) / y_std.reshape(1, 1, -1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std  = torch.tensor(x_std,  dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32)

        print(f"[Dataset ENU] Loaded {self.X.shape[0]} samples.")

    # ==============================================================
    def _build(self):
        X_list = []
        Y_list = []
        M_list = []

        for bdir in sorted(glob.glob(os.path.join(self.data_root, "batch*"))):
            truth = pd.read_csv(os.path.join(bdir, "truth.csv"))
            radar = pd.read_csv(os.path.join(bdir, "radar.csv"))
            fiveg = pd.read_csv(os.path.join(bdir, "5g_a.csv"))
            tdoa  = pd.read_csv(os.path.join(bdir, "tdoa.csv"))

            for uav in sorted(truth["id"].unique()):
                df_t = truth[truth["id"] == uav].copy().sort_values("timestamp")

                if len(df_t) < self.window_size:
                    continue

                # ===========================
                # ENU 基准点（trueth 第1帧）
                # ===========================
                lat0 = df_t.iloc[0]["lat"]
                lon0 = df_t.iloc[0]["lon"]
                alt0 = df_t.iloc[0]["alt"]

                # truth ENU
                e_t, n_t, u_t = latlon_to_enu(
                    df_t["lat"].values, df_t["lon"].values, df_t["alt"].values,
                    lat0, lon0, alt0
                )

                vx = df_t["vx"].values
                vy = df_t["vy"].values
                vz = df_t["vz"].values
                spd = df_t["speed"].values

                # 三源对齐
                mod_data = {}
                for name, df_m in zip(MODALITIES, [radar, fiveg, tdoa]):
                    md = self._align_mod(df_t, df_m, uav, lat0, lon0, alt0)
                    mod_data[name] = md

                # 滑动窗口
                T_total = len(df_t)
                for s in range(0, T_total - self.window_size + 1, self.stride):
                    e_win = e_t[s:s+self.window_size]
                    n_win = n_t[s:s+self.window_size]
                    u_win = u_t[s:s+self.window_size]

                    # truth -> Y
                    Y = np.stack([e_win, n_win, u_win], axis=-1)

                    # X
                    X = np.zeros((self.window_size, N_MODALITIES, NODE_FEAT_DIM))
                    M = np.zeros((self.window_size, N_MODALITIES))

                    t_norm = np.linspace(0, 1, self.window_size)

                    for mi, name in enumerate(MODALITIES):
                        md = mod_data[name]

                        X[:, mi, 0] = md["east"][s:s+self.window_size]
                        X[:, mi, 1] = md["north"][s:s+self.window_size]
                        X[:, mi, 2] = md["up"][s:s+self.window_size]

                        X[:, mi, 3] = vx[s:s+self.window_size]
                        X[:, mi, 4] = vy[s:s+self.window_size]
                        X[:, mi, 5] = vz[s:s+self.window_size]
                        X[:, mi, 6] = spd[s:s+self.window_size]
                        X[:, mi, 7] = md["conf"][s:s+self.window_size]
                        X[:, mi, 8] = t_norm

                        if name == "radar":
                            X[:, mi, 9] = 1
                        elif name == "5g_a":
                            X[:, mi, 10] = 1
                        elif name == "tdoa":
                            X[:, mi, 11] = 1

                        M[:, mi] = md["mask"][s:s+self.window_size]

                    X_list.append(X)
                    Y_list.append(Y)
                    M_list.append(M)

        return X_list, Y_list, M_list

    # ==============================================================
    def _align_mod(self, df_t, df_m, uav_id, lat0, lon0, alt0):
        df_truth_u = df_t[["timestamp", "lat", "lon", "alt"]].rename(
            columns={"lat": "lat_truth", "lon": "lon_truth", "alt": "alt_truth"}
        )

        df_m_u = df_m[df_m["id"] == uav_id][["timestamp", "lat", "lon", "alt", "source_conf"]]

        merged = df_truth_u.merge(df_m_u, on="timestamp", how="left")

        # mask = 有观测 = 非缺失
        mask = (~merged["lat"].isna()).astype(np.float32).values

        # 原始观测值（可能为 NaN）
        lat = merged["lat"].values
        lon = merged["lon"].values
        alt = merged["alt"].values

        # 缺失时不应该用 truth 填！应该用基准值
        lat = pd.Series(lat).fillna(method="ffill").fillna(method="bfill").values
        lon = pd.Series(lon).fillna(method="ffill").fillna(method="bfill").values
        alt = pd.Series(alt).fillna(method="ffill").fillna(method="bfill").values

        east, north, up = latlon_to_enu(lat, lon, alt, lat0, lon0, alt0)
        conf = merged["source_conf"].fillna(0).values.astype(np.float32)

        return dict(east=east, north=north, up=up, conf=conf, mask=mask)

    # ==============================================================
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], self.M[idx]
