import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# 1️⃣ 手动配置要检查的 batch 路径
# ================================================================

BATCH_DIR = r"./dataset/scenario_multi_source_1000x120/batch02"

# 如果你的路径不同，直接改这一行即可

# ================================================================
# 2️⃣ 工具函数（与生成器严格一致）
# ================================================================

def calculate_bearing(lat_s, lon_s, lat_u, lon_u):
    """传感器到 UAV 的方位角（bearing）"""
    lat_s, lon_s = np.radians(lat_s), np.radians(lon_s)
    lat_u, lon_u = np.radians(lat_u), np.radians(lon_u)

    d_lon = lon_u - lon_s

    x = np.sin(d_lon) * np.cos(lat_u)
    y = np.cos(lat_s)*np.sin(lat_u) - np.sin(lat_s)*np.cos(lat_u)*np.cos(d_lon)

    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


def angle_diff(a, b):
    """最小角度差（-180° ~ 180°）"""
    return (a - b + 180) % 360 - 180


def latlon_dist_m(lat1, lon1, lat2, lon2):
    """经纬度差转米"""
    dx = (lon1 - lon2) * 111000
    dy = (lat1 - lat2) * 111000
    return np.sqrt(dx*dx + dy*dy)


# ================================================================
# 3️⃣ EM / Acoustic 方位角准确性验证
# ================================================================

def check_angle_alignment(df_obs, truth, title):
    print(f"\n===== Checking angle alignment for {title} =====")

    merged = df_obs.merge(truth, on=["timestamp","id"], suffixes=("_obs","_uav"))
    errors = []

    for _, row in merged.iterrows():
        angle_true = calculate_bearing(
            row.sensor_lat, row.sensor_lon,
            row.lat_uav, row.lon_uav
        )
        err = angle_diff(row.angle_deg, angle_true)
        errors.append(err)

    errors = np.array(errors)

    print(f"[{title}] count = {len(errors)}")
    print(f"MAE  = {np.mean(np.abs(errors)):.2f}°")
    print(f"RMSE = {np.sqrt(np.mean(errors**2)):.2f}°")
    print(f"MAX  = {np.max(np.abs(errors)):.2f}°")

    # 误差直方图
    plt.figure(figsize=(7,5))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title(f"{title} Angle Error Distribution")
    plt.xlabel("Angle Error (deg)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.show()


# ================================================================
# 4️⃣ 五个轨迹源置信度 vs 误差验证
# ================================================================

def check_conf_vs_error(df_obs, truth, title):
    print(f"\n===== Checking confidence vs error for {title} =====")

    merged = df_obs.merge(truth, on=["timestamp","id"], suffixes=("_obs","_truth"))

    errors = latlon_dist_m(
        merged.lat_obs, merged.lon_obs,
        merged.lat_truth, merged.lon_truth
    )
    conf = merged.source_conf.values

    # 输出统计
    corr = np.corrcoef(errors, conf)[0,1]
    print(f"[{title}] corr(error, conf) = {corr:.4f} (should < 0)")
    print(f"error mean = {errors.mean():.2f}m")
    print(f"conf  mean = {conf.mean():.3f}")

    # 绘制散点图：误差→置信度
    plt.figure(figsize=(7,5))
    plt.scatter(errors, conf, s=10, alpha=0.3)
    plt.title(f"{title}  Error vs Confidence")
    plt.xlabel("Error (m)")
    plt.ylabel("Confidence")
    plt.grid(alpha=0.3)
    plt.show()


# ================================================================
# 5️⃣ 主流程
# ================================================================

def main():

    print(f"\n[PATH] Checking batch dir:\n  {BATCH_DIR}")

    truth     = pd.read_csv(os.path.join(BATCH_DIR, "truth.csv"))
    gps       = pd.read_csv(os.path.join(BATCH_DIR, "gps.csv"))
    radar     = pd.read_csv(os.path.join(BATCH_DIR, "radar.csv"))
    fiveg     = pd.read_csv(os.path.join(BATCH_DIR, "5g_a.csv"))
    tdoa      = pd.read_csv(os.path.join(BATCH_DIR, "tdoa.csv"))
    acoustic  = pd.read_csv(os.path.join(BATCH_DIR, "acoustic.csv"))
    em        = pd.read_csv(os.path.join(BATCH_DIR, "em.csv"))

    print("\n[Loaded]")
    print(" truth   :", len(truth))
    print(" gps     :", len(gps))
    print(" radar   :", len(radar))
    print(" fiveg   :", len(fiveg))
    print(" tdoa    :", len(tdoa))
    print(" acoustic:", len(acoustic))
    print(" em      :", len(em))

    # =========================
    # 1️⃣ 方位角是否真实对应轨迹？
    # =========================
    check_angle_alignment(acoustic, truth, "Acoustic (5 sensors)")
    check_angle_alignment(em, truth, "EM (5 sensors)")

    # =========================
    # 2️⃣ 置信度是否随误差下降？
    # =========================
    check_conf_vs_error(gps,   truth, "GPS")
    check_conf_vs_error(radar, truth, "Radar")
    check_conf_vs_error(fiveg, truth, "5G-A")
    check_conf_vs_error(tdoa,  truth, "TDOA")


if __name__ == "__main__":
    main()
