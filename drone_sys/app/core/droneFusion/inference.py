import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import MultiSourceGraphDataset, NODE_FEAT_DIM, WINDOW_SIZE
from dataset import latlon_to_enu, to_enu_single_point, enu_to_llh
from model import GraphFusionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================================================
# CONFIG
# ===========================================================
DATA_ROOT  = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multi_source_test"
BATCH_DIR  = os.path.join(DATA_ROOT, "batch01")
MODEL_PATH = "graph_fusion_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UID = 2

STRIDE = 5


# ===========================================================
# ERROR
# ===========================================================
def calc_err(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(dist ** 2))),
        "MAE":  float(np.mean(np.abs(dist))),
        "MAX":  float(np.max(dist)),
    }


# ===========================================================
# OAA 融合
# ===========================================================
def merge_windows(preds, T_total, window=20, stride=5):
    fusion = np.zeros((T_total, 3), dtype=np.float32)
    count  = np.zeros(T_total, dtype=np.float32)

    idx = 0
    for p in preds:
        for i in range(window):
            t = idx + i
            if t < T_total:
                fusion[t] += p[i]
                count[t]  += 1
        idx += stride

    return fusion / (count[:,None] + 1e-6)


# ===========================================================
# 完全使用 dataset._align_mod 构造滑动窗口输入
# ===========================================================
def build_inputs_from_dataset(batch_dir, uav):
    ds = MultiSourceGraphDataset(DATA_ROOT)

    truth = pd.read_csv(os.path.join(batch_dir, "truth.csv"))
    radar = pd.read_csv(os.path.join(batch_dir, "radar.csv"))
    fiveg = pd.read_csv(os.path.join(batch_dir, "5g_a.csv"))
    tdoa  = pd.read_csv(os.path.join(batch_dir, "tdoa.csv"))

    truth = truth[truth["id"] == uav].sort_values("timestamp")

    # ENU 基准点
    lat0, lon0, alt0 = truth.iloc[0][["lat", "lon", "alt"]]

    # truth ENU（全序列）
    e_gt, n_gt, u_gt = latlon_to_enu(
        truth["lat"].values, truth["lon"].values, truth["alt"].values,
        lat0, lon0, alt0
    )
    truth_full_enu = np.stack([e_gt, n_gt, u_gt], axis=-1)

    # --- dataset 的对齐方法 ---
    align = ds._align_mod
    radar_m = align(truth, radar, uav, lat0, lon0, alt0)
    fiveg_m = align(truth, fiveg, uav, lat0, lon0, alt0)
    tdoa_m  = align(truth, tdoa,  uav, lat0, lon0, alt0)

    # 保存模态经纬度（用于绘图）
    radar_lat = truth["lat"].values
    radar_lon = truth["lon"].values
    fiveg_lat = truth["lat"].values
    fiveg_lon = truth["lon"].values
    tdoa_lat  = truth["lat"].values
    tdoa_lon  = truth["lon"].values

    # ------------------- 滑动窗口构造 -------------------
    X_list, M_list = [], []
    T_total = len(truth)

    vx = truth["vx"].values
    vy = truth["vy"].values
    vz = truth["vz"].values
    spd = truth["speed"].values

    for s in range(0, T_total - WINDOW_SIZE + 1, STRIDE):
        X = np.zeros((WINDOW_SIZE, 3, NODE_FEAT_DIM), dtype=np.float32)
        M = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
        t_norm = np.linspace(0, 1, WINDOW_SIZE)

        mods = [radar_m, fiveg_m, tdoa_m]
        for mi, m in enumerate(mods):
            X[:, mi, 0:3] = np.stack([
                m["east"][s:s+WINDOW_SIZE],
                m["north"][s:s+WINDOW_SIZE],
                m["up"][s:s+WINDOW_SIZE],
            ], axis=-1)

            X[:, mi, 3] = vx[s:s+WINDOW_SIZE]
            X[:, mi, 4] = vy[s:s+WINDOW_SIZE]
            X[:, mi, 5] = vz[s:s+WINDOW_SIZE]
            X[:, mi, 6] = spd[s:s+WINDOW_SIZE]

            X[:, mi, 7] = m["conf"][s:s+WINDOW_SIZE]
            X[:, mi, 8] = t_norm

            if mi == 0: X[:, mi, 9] = 1
            if mi == 1: X[:, mi,10] = 1
            if mi == 2: X[:, mi,11] = 1

            M[:, mi] = m["mask"][s:s+WINDOW_SIZE]

        X_list.append(X)
        M_list.append(M)

    return (
        np.array(X_list),
        np.array(M_list),
        truth_full_enu,
        radar_m,
        fiveg_m,
        tdoa_m,
        (lat0, lon0, alt0)
    )


# ===========================================================
# MAIN
# ===========================================================
def main():
    truth = pd.read_csv(os.path.join(BATCH_DIR, "truth.csv"))
    uav = truth["id"].unique()[UID]
    print("🛸 UAV:", uav)

    # --- 构造输入 ---
    (X_win, M_win, truth_full_enu,
     radar_m, fiveg_m, tdoa_m, origin) = build_inputs_from_dataset(BATCH_DIR, uav)

    # --- 加载模型 ---
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg  = ckpt["config"]

    model = GraphFusionModel(
        in_dim=cfg["in_dim"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        window_size=cfg["window_size"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    x_mean = ckpt["x_mean"].to(DEVICE)
    x_std  = ckpt["x_std"].to(DEVICE)
    y_mean = ckpt["y_mean"].to(DEVICE)
    y_std  = ckpt["y_std"].to(DEVICE)

    # --- 推理 ---
    preds = []
    for X, M in zip(X_win, M_win):
        X = torch.tensor(X).unsqueeze(0).to(DEVICE)
        M = torch.tensor(M).unsqueeze(0).to(DEVICE)

        X_norm = (X - x_mean.reshape(1,1,1,-1)) / x_std.reshape(1,1,1,-1)
        with torch.no_grad():
            pred_norm = model(X_norm, M)
            pred = pred_norm * y_std + y_mean
            preds.append(pred.squeeze(0).cpu().numpy())

    preds = np.array(preds)   # [num_windows,20,3]

    # --- OAA 融合 ---
    fusion_enu = merge_windows(preds, T_total=len(truth_full_enu))

    # --- 误差 ---
    fusion_err = calc_err(fusion_enu, truth_full_enu)
    radar_err  = calc_err(np.stack([radar_m["east"], radar_m["north"], radar_m["up"]],axis=1), truth_full_enu)
    fiveg_err  = calc_err(np.stack([fiveg_m["east"], fiveg_m["north"], fiveg_m["up"]],axis=1), truth_full_enu)
    tdoa_err   = calc_err(np.stack([tdoa_m["east"],  tdoa_m["north"],  tdoa_m["up"]], axis=1), truth_full_enu)

    print("\n📊 Fusion Error:", fusion_err)
    print("Radar :", radar_err)
    print("5G-A  :", fiveg_err)
    print("TDOA  :", tdoa_err)

    # --- ENU → 经纬度 ---
    lat0, lon0, alt0 = origin
    pred_lat, pred_lon, _ = enu_to_llh(
        fusion_enu[:,0], fusion_enu[:,1], fusion_enu[:,2], lat0, lon0, alt0
    )
    truth_lat = truth[truth["id"]==uav].sort_values("timestamp")["lat"].values
    truth_lon = truth[truth["id"]==uav].sort_values("timestamp")["lon"].values

    # --- 绘图 ---
    plt.figure(figsize=(10,8))

    # Truth
    plt.scatter(truth_lon, truth_lat, s=15, c="black", label="Truth")

    # Fusion
    plt.scatter(pred_lon, pred_lat, s=15, c="red", label="Fusion")

    # Radar
    radar_lat_plot = truth_lat   # 原始经纬度等于 truth 对齐后的纬度
    radar_lon_plot = truth_lon
    plt.scatter(radar_lon_plot, radar_lat_plot, s=8, c="blue", label="Radar", alpha=0.5)

    # 5G-A
    fiveg_lat_plot = truth_lat
    fiveg_lon_plot = truth_lon
    plt.scatter(fiveg_lon_plot, fiveg_lat_plot, s=8, c="green", label="5G-A", alpha=0.5)

    # TDOA
    tdoa_lat_plot = truth_lat
    tdoa_lon_plot = truth_lon
    plt.scatter(tdoa_lon_plot, tdoa_lat_plot, s=8, c="purple", label="TDOA", alpha=0.5)

    plt.title("Truth / Fusion / Radar / 5G-A / TDOA (Dot Plot)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
