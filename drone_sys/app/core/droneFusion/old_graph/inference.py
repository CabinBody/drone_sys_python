import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import FusionModel
from dataset import geodetic_to_enu

# ==============================
# ⚙️ 基本配置
# ==============================
MODEL_PATH = "../pt_backup/fusion_model.pt"
NORM_PATH  = "../pt_backup/fusion_norm.pth"
DATA_DIR   = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 10

# ==============================
# 🧠 加载模型与归一化参数
# ==============================
norm = torch.load(NORM_PATH)
y_mean = pd.Series(norm["y_mean"])
y_std  = pd.Series(norm["y_std"])
x_mean = pd.Series(norm["x_mean"])
x_std  = pd.Series(norm["x_std"])

model = FusionModel(
    input_dim=11, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
    tau=0.3, quality_idx=(6,7,8,9,10), ce_indices=(7,8,10)
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==============================
# 📡 读取数据并转换 ENU
# ==============================
gps = pd.read_csv(f"{DATA_DIR}/gps.csv").sort_values(["id","time"])
radar = pd.read_csv(f"{DATA_DIR}/radar.csv").sort_values(["id","time"])
truth = pd.read_csv(f"{DATA_DIR}/truth.csv").sort_values(["id","time"])

ref_lat, ref_lon, ref_alt = truth.iloc[0][["lat","lon","alt"]]
for df in [gps, radar, truth]:
    e, n, u = geodetic_to_enu(df["lat"], df["lon"], df["alt"], ref_lat, ref_lon, ref_alt)
    df["east"], df["north"], df["up"] = e, n, u

cols = ["east","north","up","vx","vy","confidence","snr","rssi","delay","coverage","noiseVar"]

# ==============================
# 🔮 模型推理（对所有 UAV）
# ==============================
pred_all, speed_all = [], []
truth_all = []

ids = gps["id"].unique()
for uid in ids:
    g = gps[gps["id"]==uid].reset_index(drop=True)
    r = radar[radar["id"]==uid].reset_index(drop=True)
    t = truth[truth["id"]==uid].reset_index(drop=True)
    tlen = min(len(g), len(r), len(t))
    if tlen < SEQ_LEN: continue

    # 取最后 seq_len 帧
    g_seq = g.iloc[-SEQ_LEN:][cols].copy()
    r_seq = r.iloc[-SEQ_LEN:][cols].copy()

    # 应用归一化
    for c in ["east","north","up"]:
        g_seq[c] = (g_seq[c] - y_mean[c]) / (y_std[c] + 1e-6)
        r_seq[c] = (r_seq[c] - y_mean[c]) / (y_std[c] + 1e-6)
    for c in ["vx","vy","confidence","snr","rssi","delay","coverage","noiseVar"]:
        g_seq[c] = (g_seq[c] - x_mean[c]) / (x_std[c] + 1e-6)
        r_seq[c] = (r_seq[c] - x_mean[c]) / (x_std[c] + 1e-6)

    x = np.stack([g_seq.values, r_seq.values], axis=1).astype(np.float32)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred, _ = model(x)

    pred = pred.cpu().numpy().flatten()
    pred = pred * y_std.values + y_mean.values  # 反归一化
    pred_all.append(pred)

    # 计算融合速度（vx, vy）来自最后帧的平均
    vxs = np.mean([g["vx"].iloc[-1], r["vx"].iloc[-1]])
    vys = np.mean([g["vy"].iloc[-1], r["vy"].iloc[-1]])
    speed = np.sqrt(vxs**2 + vys**2)
    speed_all.append(speed)

    # 对应 truth
    truth_pt = t.iloc[-1][["east","north","up"]].to_numpy()
    truth_all.append(truth_pt)

pred_all = np.array(pred_all)
truth_all = np.array(truth_all)
speed_all = np.array(speed_all)
print(f"✅ 已融合 {len(pred_all)} 架无人机的态势数据")


# ==============================
# 🎨 可视化：速度热力图 + Truth对比
# ==============================
plt.figure(figsize=(8,6))
sc = plt.scatter(pred_all[:,0], pred_all[:,1], c=speed_all, cmap='RdYlGn_r', s=80, alpha=0.8, label="Fused UAVs")
plt.scatter(truth_all[:,0], truth_all[:,1], c='black', s=30, marker='x', label="Truth")
plt.colorbar(sc, label="Speed (m/s)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Fused Airspace Situation (Speed Heatmap)")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
norm = torch.load("../pt_backup/fusion_norm.pth")
print(norm["y_std"])
