import os
import numpy as np
import pandas as pd
from datetime import datetime
import bluesky as bs
from bluesky import stack

# ========== 1. 参数配置 ==========
SCENARIO_NAME = "scenario_test_200"
UAV_COUNT = 200
SIM_DURATION = 60          # 秒
STEP_SIZE = 1.0
OUTPUT_DIR = f"./dataset/{SCENARIO_NAME}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPS_NOISE = 0.00005
RADAR_NOISE = 0.0003

# ========== 2. 通信与置信度特征生成 ==========
def gen_comm_features(base_quality=1.0):
    """
    生成真实通信参数（具有物理相关性与波动）
    base_quality: 通信质量等级 (0.5~1.0)
    """
    snr = np.clip(np.random.normal(25 * base_quality, 3), 5, 45)
    rssi = np.clip(-90 + 40 * base_quality + np.random.normal(0, 2), -95, -45)
    delay = np.clip(200 * (1.1 - base_quality) + np.random.normal(0, 10), 10, 300)
    coverage = np.clip(0.6 + 0.4 * base_quality + np.random.normal(0, 0.03), 0.6, 1.0)
    noiseVar = np.clip(0.001 * (1.1 - base_quality) + np.random.normal(0, 0.0001), 0.00005, 0.002)
    confidence = np.clip(0.7 + 0.3 * base_quality + np.random.normal(0, 0.02), 0.7, 1.0)
    return confidence, snr, rssi, delay, coverage, noiseVar

# ========== 3. 初始化 BlueSky ==========
bs.init(guimode=False)
sim = bs.sim

# 初始化 UAV 随机参数
base_lat, base_lon = 45.0, 5.0
headings = np.random.uniform(0, 360, UAV_COUNT)
speeds = np.random.uniform(40, 90, UAV_COUNT)
altitudes = np.random.uniform(9500, 10500, UAV_COUNT)

for i in range(UAV_COUNT):
    lat = base_lat + np.random.uniform(-0.3, 0.3)
    lon = base_lon + np.random.uniform(-0.3, 0.3)
    ac_id = f"TEST{i+1}"

    bs.traf.cre(ac_id, "UAV", lat, lon, headings[i], altitudes[i], speeds[i])

    # 添加随机目标点（非直线路径）
    wp_lat = lat + np.random.uniform(0.2, 0.8) * np.cos(np.deg2rad(headings[i])) \
                   + 0.15 * np.sin(np.deg2rad(i * 25 % 360))
    wp_lon = lon + np.random.uniform(0.2, 0.8) * np.sin(np.deg2rad(headings[i])) \
                   + 0.15 * np.cos(np.deg2rad(i * 30 % 360))
    stack.stack(f"ADDWPT {ac_id} N{wp_lat:.3f} E{wp_lon:.3f} {altitudes[i]:.1f}")

print(f"[✓] 已初始化 {UAV_COUNT} 架测试无人机，开始仿真...")

# ========== 4. 仿真采样 ==========
records = []
base_timestamp = int(datetime.now().timestamp())  # 当前Unix时间

for step in range(int(SIM_DURATION / STEP_SIZE)):
    sim.step(STEP_SIZE)
    bs.traf.update()
    current_time = base_timestamp + step * STEP_SIZE  # 每步加1秒

    for idx in range(bs.traf.ntraf):
        # 不同 UAV 通信质量略有差异
        base_quality = np.clip(np.random.normal(0.85, 0.1), 0.5, 1.0)
        confidence, snr, rssi, delay, coverage, noiseVar = gen_comm_features(base_quality)

        records.append({
            "time": current_time,
            "id": bs.traf.id[idx],
            "lat": bs.traf.lat[idx],
            "lon": bs.traf.lon[idx],
            "alt": bs.traf.alt[idx],
            "vx": bs.traf.gs[idx] * np.cos(np.deg2rad(bs.traf.trk[idx])),
            "vy": bs.traf.gs[idx] * np.sin(np.deg2rad(bs.traf.trk[idx])),
            "confidence": confidence,
            "snr": snr,
            "rssi": rssi,
            "delay": delay,
            "coverage": coverage,
            "noi  seVar": noiseVar
        })

truth = pd.DataFrame(records)
truth.to_csv(os.path.join(OUTPUT_DIR, "truth.csv"), index=False)
print(f"[✓] 已保存真实轨迹 truth.csv，共 {len(truth)} 条数据")

# ========== 5. 生成 GPS / RADAR 模态数据 ==========
def add_noise(df, pos_std, mode):
    noisy = df.copy()
    noisy["lat"] += np.random.normal(0, pos_std, len(df))
    noisy["lon"] += np.random.normal(0, pos_std, len(df))
    factor = 0.8 if mode == "radar" else 1.0
    noisy["confidence"] *= factor
    noisy["snr"] *= factor
    noisy["rssi"] *= factor
    noisy["delay"] *= (2 - factor)
    noisy["coverage"] *= factor
    noisy["noiseVar"] *= (2 - factor)
    return noisy

gps = add_noise(truth, GPS_NOISE, "gps")
radar = add_noise(truth, RADAR_NOISE, "radar")

gps.to_csv(os.path.join(OUTPUT_DIR, "gps.csv"), index=False)
radar.to_csv(os.path.join(OUTPUT_DIR, "radar.csv"), index=False)
print("[✓] 已生成带通信特征的测试 GPS / RADAR 数据")

# ========== 6. 保存元数据 ==========
meta = {
    "scenario": SCENARIO_NAME,
    "uav_count": UAV_COUNT,
    "duration": SIM_DURATION,
    "step": STEP_SIZE,
    "gps_noise": GPS_NOISE,
    "radar_noise": RADAR_NOISE,
    "timestamp": datetime.now().isoformat(),
    "fields": [
        "time", "id", "lat", "lon", "alt", "vx", "vy",
        "confidence", "snr", "rssi", "delay", "coverage", "noiseVar"
    ]
}
pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "meta.json"))
print(f"[✓] 测试数据集生成完成：{OUTPUT_DIR}")
