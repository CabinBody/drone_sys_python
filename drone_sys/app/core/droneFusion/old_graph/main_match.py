import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import FusionModel
from uav_match import FrameMatcher, MatchConfig
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================
# ⚙️ 配置参数
# ==============================
MODEL_PATH = "../pt_backup/fusion_model.pt"
NORM_PATH = "../pt_backup/fusion_norm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 输入路径 =====
RADAR_CSV    = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_test\radar.csv"
GPS_CSV      = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_test\gps.csv"
REPORTED_CSV = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_test\truth.csv"

PRINT_DETAIL = True

# 匹配参数（使用你调宽阈值的版本）
cfg = MatchConfig({
    "R_GATE_M": 1200.0,
    "TAU_T": 2.0,
    "KEEP_RATIO": 0.35,
    "EMA_BETA": 0.90,
    "TOPK": 8,
    "POS_SIGMA_M": 250.0,
    "VEL_SIGMA": 40.0,
    "T_SIGMA": 1.2,
    "LAMBDA_A": 0.8,
    "LAMBDA_B": 0.8,
    "EDGE_POS_SIGMA_M": 300.0,
    "EDGE_VEL_SIGMA": 50.0,
    "EDGE_LAMBDA": 0.4,
    "DET_KNN": 8,
    "REP_KNN": 8,
    "TAU_NODE": 0.15,
    "D_MAX_M": 200.0,
    "THOLD": 5,
    "L_VOTE": 4,
    "PRINT_DETAIL": False
})
matcher = FrameMatcher(cfg)

# ==============================
# 🚀 加载模型 + 归一化参数
# ==============================
model = FusionModel(input_dim=11)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

norm = torch.load(NORM_PATH, map_location=DEVICE)


# ===== 加载 + 对齐三个来源的时间戳 =====
def load_modal_and_report_data():
    radar_df   = pd.read_csv(RADAR_CSV)
    gps_df     = pd.read_csv(GPS_CSV)
    reported_df= pd.read_csv(REPORTED_CSV)

    # 取三者时间戳交集，确保同一帧比较
    common_times = sorted(set(radar_df["time"]) & set(gps_df["time"]) & set(reported_df["time"]))
    radar_df    = radar_df[radar_df["time"].isin(common_times)].reset_index(drop=True)
    gps_df      = gps_df[gps_df["time"].isin(common_times)].reset_index(drop=True)
    reported_df = reported_df[reported_df["time"].isin(common_times)].reset_index(drop=True)
    return radar_df, gps_df, reported_df


def dict_to_tensor(d):
    vals = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in d.values()]
    return torch.tensor(np.array([v.item() if v.numel()==1 else v.mean().item() for v in vals]), dtype=torch.float32)

x_mean = dict_to_tensor(norm["x_mean"]).to(DEVICE)
x_std  = dict_to_tensor(norm["x_std"]).to(DEVICE)
y_mean = dict_to_tensor(norm["y_mean"]).to(DEVICE)
y_std  = dict_to_tensor(norm["y_std"]).to(DEVICE)

pos_idx = [0,1,2]
feat_idx = [3,4,5,6,7,8,9,10]
def normalize_x(x):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    x_norm = x.clone()
    x_norm[..., pos_idx] = (x[..., pos_idx] - y_mean[None,None,None,:]) / (y_std[None,None,None,:] + 1e-6)
    for i, idx in enumerate(feat_idx):
        x_norm[..., idx] = (x[..., idx] - x_mean[i]) / (x_std[i] + 1e-6)
    return x_norm

def denormalize_y(y):
    return y * (y_std + 1e-6) + y_mean

# ==============================
# 📡 数据加载
# ==============================
def load_modal_data():
    radar_df = pd.read_csv(RADAR_CSV)
    gps_df = pd.read_csv(GPS_CSV)
    common_times = sorted(set(radar_df["time"]) & set(gps_df["time"]))
    radar_df = radar_df[radar_df["time"].isin(common_times)]
    gps_df = gps_df[gps_df["time"].isin(common_times)]
    return radar_df, gps_df

# ==============================
# 🔮 推理函数（融合输出）
# ==============================
def fuse_frame(radar_frame, gps_frame):
    x_radar = radar_frame.drop(columns=["time","id"]).to_numpy()
    x_gps = gps_frame.drop(columns=["time","id"]).to_numpy()
    x = np.stack([x_radar, x_gps], axis=1)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_norm = normalize_x(x)
    with torch.no_grad():
        pred, aux = model(x_norm)
    pred_np = denormalize_y(pred).cpu().numpy()[0]
    fused_df = pd.DataFrame([{
        "time": float(radar_frame["time"].iloc[0]),
        "id": "FUSED",
        "lat": pred_np[0],
        "lon": pred_np[1],
        "alt": pred_np[2],
        "vx": 0, "vy": 0,
        "confidence": aux["g"].mean().item(),
        "snr": 0, "rssi": 0, "delay": 0,
        "coverage": 1, "noiseVar": 0
    }])
    return fused_df

# ==============================
# 🎨 绘图函数
# ==============================
def plot_frame(fused_df, rep_df, t, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.title(f"Time = {t}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)

    # 上报轨迹 (红，待检)
    plt.scatter(rep_df["lon"], rep_df["lat"], color="red", label="Reported (to-check)", alpha=0.8, s=20)
    # 融合轨迹 (蓝，可信)
    plt.scatter(fused_df["lon"], fused_df["lat"], color="blue", label="Fusion (trusted)", alpha=0.9, marker="x", s=60)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{int(t)}.png", dpi=200)
    plt.close()

# ==============================
# 🧩 主循环：融合 + 匹配 + 可视化
# ==============================
if __name__ == "__main__":
    radar_df, gps_df, reported_df = load_modal_and_report_data()
    all_results = []
    os.makedirs("../plots", exist_ok=True)

    for t in sorted(set(radar_df["time"])):
        radar_frame    = radar_df[radar_df["time"] == t]
        gps_frame      = gps_df[gps_df["time"] == t]
        reported_frame = reported_df[reported_df["time"] == t]      # ← 上报帧

        fused_df = fuse_frame(radar_frame, gps_frame)                # det = 融合（可信）
        res = matcher.process(fused_df, reported_frame)              # rep = 上报（待校验）✅

        all_results.append(res)
        if PRINT_DETAIL:
            print(f"\n⏱ Time {t}")
            print("Matches:", res["matches"])
            print("Undetected (上报未被可信融合匹配):", res["undetected"])
            print("AbnormalQueue:", res["AbnormalQueue"])

        # 只在“上报异常”时画图：undetected / AbnormalQueue
        if len(res["undetected"]) > 0 or len(res["AbnormalQueue"]) > 0:
            plot_frame(fused_df, reported_frame, t)

    print("\n✅ 全部时间帧处理完成，累计异常目标：", matcher.AbnormalQueue)
    print("📁 图像已保存至 ./plots/")
