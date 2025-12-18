
import os
import numpy as np
import pandas as pd
from datetime import datetime
import bluesky as bs
from bluesky import stack

# ---------------------------------------------------------------
# 1️⃣ 全局配置
# ---------------------------------------------------------------

SCENARIO_NAME = "scenario_multi_source_5000x120"
OUTPUT_ROOT   = f"./dataset/{SCENARIO_NAME}/"

UAV_COUNT     = 5000
SIM_DURATION  = 120
STEP_SIZE     = 1.0
BATCH_SIZE    = 100

# 随机折线路径参数
SEG_MIN = 4
SEG_MAX = 8
STEP_MIN = 0.05
STEP_MAX = 0.15

# ================================================================
# 🔧 可配置噪声控制（全局统一入口）
# ================================================================
NOISE_CONFIG = dict(
    gps=dict(pos_noise_m=5.0,  alt_noise_m=5.0,  vel_noise_m=0.3),
    radar=dict(pos_noise_m=18.0, alt_noise_m=7.0, vel_noise_m=0.5),
    fiveg=dict(pos_noise_m=30.0, alt_noise_m=12.0, vel_noise_m=0.6),
    tdoa=dict(pos_noise_m=60.0, alt_noise_m=20.0, vel_noise_m=0.8),
)

# 角度噪声
ACOUSTIC_ANGLE_STD = 10
EM_ANGLE_STD       = 16

# 模态基础置信度
BASE_CONF = dict(
    gps=0.92,
    radar=0.85,
    fiveg=0.78,
    tdoa=0.65,
    em=0.55,
    acoustic=0.50,
)

# 模态误差衰减尺度
CONF_SCALE = dict(
    gps=20.0,
    radar=30.0,
    fiveg=45.0,
    tdoa=60.0,
    em=70.0,
    acoustic=70.0,
)

# dropout
DROPOUT = dict(
    gps=0.00,
    radar=0.10,
    fiveg=0.15,
    tdoa=0.20,
    em=0.10,
    acoustic=0.12,
)

CONF_MIN = 0.05

base_lat, base_lon = 45.0, 5.0

# 传感器位置
EM_SENSORS = {
    "EM01": (base_lat + 0.10, base_lon + 0.05, 3.0),
    "EM02": (base_lat - 0.12, base_lon + 0.02, 3.0),
    "EM03": (base_lat + 0.08, base_lon - 0.06, 3.0),
    "EM04": (base_lat - 0.05, base_lon - 0.08, 3.0),
    "EM05": (base_lat + 0.03, base_lon + 0.12, 3.0),
}

ACOUSTIC_SENSORS = {
    "AC01": (base_lat + 0.06, base_lon + 0.03, 5.0),
    "AC02": (base_lat - 0.04, base_lon + 0.07, 5.0),
    "AC03": (base_lat + 0.02, base_lon - 0.05, 5.0),
    "AC04": (base_lat - 0.10, base_lon - 0.04, 5.0),
    "AC05": (base_lat + 0.12, base_lon + 0.01, 5.0),
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------------------------------
# 2️⃣ 工具函数
# ---------------------------------------------------------------

def apply_dropout(df, name):
    p = DROPOUT[name]
    if p <= 0:
        return df
    return df[np.random.rand(len(df)) > p].reset_index(drop=True)


# ⭐ 直接使用 BlueSky 内部速度，不再差分计算
def extract_truth_speed():
    """
    兼容旧版 BlueSky：使用 gs + trk + vs 计算三轴速度。
    """
    gs  = bs.traf.gs.copy()
    trk = bs.traf.trk.copy()
    vs  = bs.traf.vs.copy()   # 垂直速度（m/s）

    vx = gs * np.cos(np.deg2rad(trk))   # 东向速度
    vy = gs * np.sin(np.deg2rad(trk))   # 北向速度
    vz = vs                             # 上升速度

    return dict(
        vx=vx,
        vy=vy,
        vz=vz,
        speed=gs,
        track=trk,
    )



# bearing
def calculate_bearing(lat_s, lon_s, lat_u, lon_u):
    lat_s, lon_s = np.radians(lat_s), np.radians(lon_s)
    lat_u, lon_u = np.radians(lat_u), np.radians(lon_u)
    d_lon = lon_u - lon_s
    x = np.sin(d_lon) * np.cos(lat_u)
    y = np.cos(lat_s)*np.sin(lat_u) - np.sin(lat_s)*np.cos(lat_u)*np.cos(d_lon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


# 加噪声（位置+高度）
def add_noise_xyz(df, modality):
    cfg = NOISE_CONFIG[modality]
    pos_noise_m = cfg["pos_noise_m"]
    alt_noise_m = cfg["alt_noise_m"]

    noisy = df.copy()
    n = len(df)

    # 米 -> 度
    lat_noise = np.random.normal(0, pos_noise_m / 111000, n)
    lon_noise = np.random.normal(0, pos_noise_m / 111000, n)
    alt_noise = np.random.normal(0, alt_noise_m, n)

    noisy["lat"] += np.convolve(lat_noise, np.ones(3)/3, mode="same")
    noisy["lon"] += np.convolve(lon_noise, np.ones(3)/3, mode="same")
    noisy["alt"] += np.convolve(alt_noise, np.ones(3)/3, mode="same")

    return noisy


# ⭐ 给速度加噪声（真实 BlueSky vx/vy/vz 基础上）
def add_noise_velocity(df, modality):
    vel_std = NOISE_CONFIG[modality]["vel_noise_m"]
    df = df.copy()
    df["vx"] += np.random.normal(0, vel_std, len(df))
    df["vy"] += np.random.normal(0, vel_std, len(df))
    df["vz"] += np.random.normal(0, vel_std, len(df))
    df["speed"] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
    return df


# 模态置信度计算
def compute_conf_by_error(df_obs, df_truth, modality):
    merged = df_obs.merge(
        df_truth[["timestamp","id","lat","lon","alt"]],
        on=["timestamp","id"],
        suffixes=("", "_truth")
    )

    dlat = (merged.lat - merged.lat_truth) * 111000
    dlon = (merged.lon - merged.lon_truth) * 111000
    dalt = merged.alt - merged.alt_truth

    error = np.sqrt(dlat**2 + dlon**2 + dalt**2)

    base = BASE_CONF[modality]
    scale = CONF_SCALE[modality]

    conf = base * np.exp(-error / scale)
    conf = np.clip(conf, CONF_MIN, 1.0)

    merged["source_conf"] = conf

    return merged[[
        "timestamp","id","lat","lon","alt","source_conf",
        "vx","vy","vz","speed"
    ]]

# =========================================
# 🔧 UAV 高度模型：起飞 → 巡航 → 降落
# =========================================
def generate_realistic_altitude(n_steps, base_alt):
    """
    真实 UAV 高度变化模型：
      0–10%：爬升
      10–90%：巡航
      90–100%：下降
    """
    t = np.linspace(0, 1, n_steps)
    climb_frac = 0.10
    descend_frac = 0.10
    cruise_alt = base_alt + 100 + np.random.uniform(-20, 20)  # 巡航高度

    alt = np.zeros(n_steps)

    for i, tau in enumerate(t):
        if tau <= climb_frac:
            # 线性爬升
            rate = tau / climb_frac
            alt[i] = base_alt + (cruise_alt - base_alt) * rate

        elif tau >= 1 - descend_frac:
            # 下降
            tau2 = (tau - (1 - descend_frac)) / descend_frac
            alt[i] = cruise_alt - (cruise_alt - base_alt) * tau2

        else:
            # 巡航
            alt[i] = cruise_alt

    return alt

def apply_noise(lat, lon, alt, vx, vy, vz, noise_cfg):
    """
    将噪声从米转换到经纬度偏移
    """
    # 经纬度 1m 约等于：
    dlat = noise_cfg["pos_noise_m"] / 111320
    dlon = noise_cfg["pos_noise_m"] / (111320 * np.cos(np.radians(lat)))
    dalt = noise_cfg["alt_noise_m"]

    lat_n = lat + np.random.normal(0, dlat)
    lon_n = lon + np.random.normal(0, dlon)
    alt_n = alt + np.random.normal(0, dalt)

    vx_n = vx + np.random.normal(0, noise_cfg["vel_noise_m"])
    vy_n = vy + np.random.normal(0, noise_cfg["vel_noise_m"])
    vz_n = vz + np.random.normal(0, noise_cfg["vel_noise_m"])

    return lat_n, lon_n, alt_n, vx_n, vy_n, vz_n



# ---------------------------------------------------------------
# 3️⃣ BlueSky 模拟
# ---------------------------------------------------------------

REAL_ALT = {}
for i in range(UAV_COUNT):
    uav_id = f"UAV{i+1}"
    base_alt = np.random.uniform(80, 120)  # 起飞前地面高度
    REAL_ALT[uav_id] = generate_realistic_altitude(
        int(SIM_DURATION / STEP_SIZE),
        base_alt
    )

bs.init(guimode=False)
sim = bs.sim

print(f"[INIT] Creating {UAV_COUNT} UAVs...")


for i in range(UAV_COUNT):
    ac_id = f"UAV{i+1}"

    lat = base_lat + np.random.uniform(-0.3, 0.3)
    lon = base_lon + np.random.uniform(-0.3, 0.3)
    alt = np.random.uniform(80, 150)
    heading = np.random.uniform(0, 360)
    speed = np.random.uniform(12, 20)

    bs.traf.cre(ac_id, "UAV", lat, lon, heading, alt, speed)

    seg_count = np.random.randint(SEG_MIN, SEG_MAX+1)
    for _ in range(seg_count):
        turn = np.random.uniform(-120, 120)
        heading = (heading + turn) % 360
        dist = np.random.uniform(STEP_MIN, STEP_MAX)
        lat += dist * np.cos(np.deg2rad(heading))
        lon += dist * np.sin(np.deg2rad(heading))
        alt += np.random.uniform(-20, 20)
        stack.stack(f"ADDWPT {ac_id} N{lat:.4f} E{lon:.4f} {alt:.1f}")


# ---------------------------------------------------------------
# 4️⃣ 采集 truth（直接使用 BlueSky 内部速度）
# ---------------------------------------------------------------

print("[SIM] Running simulation...")

records = []
ts0 = int(datetime.now().timestamp())

for step in range(int(SIM_DURATION / STEP_SIZE)):
    sim.step(STEP_SIZE)
    bs.traf.update()
    ts = ts0 + step

    # ⭐ 提取所有 UAV 的速度
    vel = extract_truth_speed()

    for idx in range(bs.traf.ntraf):
        uav_id = bs.traf.id[idx]

        records.append({
            "timestamp": ts,
            "id": uav_id,
            "lat": bs.traf.lat[idx],
            "lon": bs.traf.lon[idx],
            "alt": REAL_ALT[uav_id][step],

            # ⭐ BlueSky 原生三轴速度
            "vx": vel["vx"][idx],
            "vy": vel["vy"][idx],
            "vz": vel["vz"][idx],
            "speed": vel["speed"][idx],
            "track": vel["track"][idx],
        })

truth = pd.DataFrame(records)

# ---------------------------------------------------------------
# 5️⃣ 导出 batch 文件
# ---------------------------------------------------------------

total_batches = (UAV_COUNT + BATCH_SIZE - 1) // BATCH_SIZE

for b in range(total_batches):
    ids = [f"UAV{i+1}" for i in range(b*BATCH_SIZE, min((b+1)*BATCH_SIZE, UAV_COUNT))]
    bt = truth[truth["id"].isin(ids)].reset_index(drop=True)

    path = os.path.join(OUTPUT_ROOT, f"batch{b+1:02d}")
    os.makedirs(path, exist_ok=True)

    # ⭐ 为每个模态加噪声（位置 + 速度）
    for key, fname in [
        ("gps", "gps.csv"),
        ("radar", "radar.csv"),
        ("fiveg", "5g_a.csv"),
        ("tdoa", "tdoa.csv"),
    ]:
        df = add_noise_xyz(bt, key)
        df = add_noise_velocity(df, key)

        # 加置信度
        conf_df = bt.rename(columns={
            "vx": "vx_truth",
            "vy": "vy_truth",
            "vz": "vz_truth",
            "speed": "speed_truth"
        })
        df = df.merge(conf_df[["timestamp", "id", "vx_truth", "vy_truth", "vz_truth", "speed_truth"]],
                      on=["timestamp", "id"], how="left")

        df = compute_conf_by_error(df, bt, key)
        df = apply_dropout(df, key)

        df.to_csv(os.path.join(path, fname), index=False)

    # ---------------------
    # Acoustic ×5
    # ---------------------
    ac_list = []
    for sid, (slat, slon, salt) in ACOUSTIC_SENSORS.items():
        tmp = bt.copy()
        tmp["sensor_id"] = sid
        tmp["sensor_lat"] = slat
        tmp["sensor_lon"] = slon
        tmp["sensor_alt"] = salt

        angle_obs = []
        angle_true = []

        for _, row in tmp.iterrows():
            at = calculate_bearing(slat, slon, row.lat, row.lon)
            ao = (at + np.random.normal(0, ACOUSTIC_ANGLE_STD)) % 360
            angle_true.append(at)
            angle_obs.append(ao)

        tmp["angle_deg"] = angle_obs

        # angle error
        ang_err = np.abs(np.array(angle_obs) - np.array(angle_true))
        ang_err = np.minimum(ang_err, 360 - ang_err)

        conf = BASE_CONF["acoustic"] * np.exp(-ang_err / CONF_SCALE["acoustic"])
        tmp["source_conf"] = np.clip(conf, CONF_MIN, 1.0)

        tmp = apply_dropout(tmp, "acoustic")

        # ⭐ 按要求输出字段
        tmp = tmp[
            [
                "timestamp", "id", "lat", "lon", "alt",
                "vx", "vy", "vz", "speed",
                "sensor_id", "sensor_lat", "sensor_lon", "sensor_alt",
                "angle_deg", "source_conf"
            ]
        ]

        ac_list.append(tmp)

    acoustic = pd.concat(ac_list, ignore_index=True)
    acoustic.to_csv(os.path.join(path, "acoustic.csv"), index=False)

    # ---------------------
    # EM ×5
    # ---------------------
    em_list = []
    for sid, (slat, slon, salt) in EM_SENSORS.items():
        tmp = bt.copy()
        tmp["sensor_id"] = sid
        tmp["sensor_lat"] = slat
        tmp["sensor_lon"] = slon
        tmp["sensor_alt"] = salt

        angle_obs = []
        angle_true = []

        for _, row in tmp.iterrows():
            at = calculate_bearing(slat, slon, row.lat, row.lon)
            ao = (at + np.random.normal(0, EM_ANGLE_STD)) % 360
            angle_true.append(at)
            angle_obs.append(ao)

        tmp["angle_deg"] = angle_obs

        ang_err = np.abs(np.array(angle_obs) - np.array(angle_true))
        ang_err = np.minimum(ang_err, 360 - ang_err)

        conf = BASE_CONF["em"] * np.exp(-ang_err / CONF_SCALE["em"])
        tmp["source_conf"] = np.clip(conf, CONF_MIN, 1.0)

        tmp = apply_dropout(tmp, "em")

        # ⭐ 输出字段必须一致
        tmp = tmp[
            [
                "timestamp", "id", "lat", "lon", "alt",
                "vx", "vy", "vz", "speed",
                "sensor_id", "sensor_lat", "sensor_lon", "sensor_alt",
                "angle_deg", "source_conf"
            ]
        ]

        em_list.append(tmp)

    em = pd.concat(em_list, ignore_index=True)
    em.to_csv(os.path.join(path, "em.csv"), index=False)

    bt.to_csv(os.path.join(path, "truth.csv"), index=False)
    print(f"[✓] Batch {b+1:02d} exported.")

print(f"\n🎉 Dataset saved to: {OUTPUT_ROOT}")
