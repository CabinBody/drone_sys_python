import numpy as np
import torch
import matplotlib.pyplot as plt

# ==== 使用 serve.py 里已经实现好的：模型加载 + build_model_input ====
from serve import build_model_input, model, x_mean, x_std, y_mean, y_std, DEVICE
from dataset import to_enu_single_point


# =====================================================
# 1. 模拟与训练集一致的真实轨迹（Truth）
# =====================================================
def simulate_truth(T, lat0, lon0, alt0):
    """
    UAV 每帧大约移动 8~15 米（与你训练数据 BlueSky 分布相符）
    """

    step_e = np.random.uniform(8, 15, T).cumsum()
    step_n = np.random.uniform(8, 15, T).cumsum()

    east = step_e - step_e[0]
    north = step_n - step_n[0]
    up = alt0 + np.linspace(0, 5, T)

    # ENU → lat/lon
    lat = lat0 + north / 111000
    lon = lon0 + east / (111000 * np.cos(np.radians(lat0)))

    return lat, lon, up, east, north, up


# =====================================================
# 2. 模拟速度（符合真实 UAV 分布）
# =====================================================
def simulate_velocity(T):
    vx = np.random.uniform(5, 15, T)
    vy = np.random.uniform(5, 15, T)
    vz = np.random.uniform(-0.5, 0.5, T)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return vx, vy, vz, speed


# =====================================================
# 3. 模态噪声（与训练集一致）
# =====================================================
def simulate_modality(lat, lon, alt, noise_m, lat0, lon0):
    # 经纬度噪声（米 → 度）
    d_lat = noise_m / 111000
    d_lon = noise_m / (111000 * np.cos(np.radians(lat0)))

    noisy_lat = lat + np.random.randn(len(lat)) * d_lat
    noisy_lon = lon + np.random.randn(len(lon)) * d_lon
    noisy_alt = alt + np.random.randn(len(alt)) * (noise_m * 0.1)

    return noisy_lat, noisy_lon, noisy_alt


# =====================================================
# 4. 构造 MockFrame 数据结构（模拟真实传感器 JSON）
# =====================================================
class MockFrame:
    def __init__(self, ts, lat, lon, alt, vx, vy, vz, spd, conf):
        self.timestamp = ts
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.speed = spd
        self.source_conf = conf


def make_frames(lat, lon, alt, vx, vy, vz, speed, conf_list):
    return [
        MockFrame(
            i,
            lat[i], lon[i], alt[i],
            vx[i], vy[i], vz[i], speed[i],
            conf_list
        )
        for i in range(len(lat))
    ]


class Modal:
    def __init__(self, radar_f, fiveg_f, tdoa_f, lat0, lon0, alt0):
        self.radar = radar_f
        self.fiveg = fiveg_f
        self.tdoa  = tdoa_f
        self.meta = {"lat0": lat0, "lon0": lon0, "alt0": alt0}


# =====================================================
#                     MAIN
# =====================================================
def main():
    np.random.seed(42)

    T = 20
    lat0, lon0, alt0 = 39.90, 116.30, 50.0

    # --- Truth ---
    truth_lat, truth_lon, truth_alt, truth_e, truth_n, truth_u = simulate_truth(T, lat0, lon0, alt0)

    # --- Velocity ---
    vx, vy, vz, speed = simulate_velocity(T)

    # --- Modalities ---
    radar_lat, radar_lon, radar_alt = simulate_modality(truth_lat, truth_lon, truth_alt, 15, lat0, lon0)
    fiveg_lat, fiveg_lon, fiveg_alt = simulate_modality(truth_lat, truth_lon, truth_alt, 30, lat0, lon0)
    tdoa_lat,  tdoa_lon,  tdoa_alt  = simulate_modality(truth_lat, truth_lon, truth_alt, 60, lat0, lon0)

    # --- Convert to frames ---
    radar_f = make_frames(radar_lat, radar_lon, radar_alt, vx, vy, vz, speed, 0.9)
    fiveg_f = make_frames(fiveg_lat, fiveg_lon, fiveg_alt, vx, vy, vz, speed, 0.7)
    tdoa_f  = make_frames(tdoa_lat,  tdoa_lon,  tdoa_alt,  vx, vy, vz, speed, 0.5)

    mod = Modal(radar_f, fiveg_f, tdoa_f, lat0, lon0, alt0)

    # =====================================================
    # 5. 构造模型输入（使用 serve.py 的 build_model_input）
    # =====================================================
    X_np, M_np = build_model_input(mod, lat0, lon0, alt0)

    X = torch.tensor(X_np).unsqueeze(0).to(DEVICE)
    M = torch.tensor(M_np).unsqueeze(0).to(DEVICE)

    # 归一化
    X_norm = (X - x_mean.reshape(1,1,1,-1)) / x_std.reshape(1,1,1,-1)

    # 推理
    with torch.no_grad():
        pred_norm = model(X_norm, M)
        pred = pred_norm * y_std + y_mean

    fusion_enu = pred.squeeze(0).cpu().numpy()  # [20,3]


    # =====================================================
    # 6. 转换 ENU 模态以便绘图比较
    # =====================================================
    def llh_to_enu(lat, lon, alt):
        enu = []
        for i in range(T):
            e = to_enu_single_point(lat[i], lon[i], alt[i], lat0, lon0, alt0)
            enu.append(e)
        return np.array(enu)

    truth_enu = np.column_stack([truth_e, truth_n, truth_u])
    radar_enu = llh_to_enu(radar_lat, radar_lon, radar_alt)
    fiveg_enu = llh_to_enu(fiveg_lat, fiveg_lon, fiveg_alt)
    tdoa_enu  = llh_to_enu(tdoa_lat,  tdoa_lon,  tdoa_alt)


    # =====================================================
    # 7. 绘图
    # =====================================================
    plt.figure(figsize=(8, 7))

    plt.plot(truth_enu[:,0], truth_enu[:,1], 'k-', label="Truth", linewidth=2)
    plt.scatter(radar_enu[:,0], radar_enu[:,1], c='blue', s=35, label="Radar", alpha=0.8)
    plt.scatter(fiveg_enu[:,0], fiveg_enu[:,1], c='green', s=35, label="5G-A", alpha=0.8)
    plt.scatter(tdoa_enu[:,0],  tdoa_enu[:,1],  c='purple', s=35, label="TDOA", alpha=0.8)

    plt.plot(fusion_enu[:,0], fusion_enu[:,1], 'r--', linewidth=2, label="Fusion")

    plt.title("Fusion vs Truth (Mocked Multi-Source)", fontsize=13)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
