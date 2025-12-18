import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ====== 配置 ======
BATCH_DIR = "./dataset/scenario_multi_source_5000x120/batch01"
uav_id = "UAV10"

# ====== 读取数据 ======
def load_source(name):
    df = pd.read_csv(os.path.join(BATCH_DIR, f"{name}.csv"))
    return df[df["id"] == uav_id]

truth = load_source("truth")
gps   = load_source("gps")
radar = load_source("radar")
fiveg = load_source("5g_a")
tdoa  = load_source("tdoa")

# ====== 3D 轨迹绘制 ======
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(truth["lon"], truth["lat"], truth["alt"], 'k-', label="Truth", linewidth=2)
ax.plot(gps["lon"], gps["lat"], gps["alt"], 'b--', alpha=0.6, label="GPS")
ax.plot(radar["lon"], radar["lat"], radar["alt"], 'r--', alpha=0.6, label="Radar")
ax.plot(fiveg["lon"], fiveg["lat"], fiveg["alt"], 'c--', alpha=0.6, label="5G-A")
ax.plot(tdoa["lon"], tdoa["lat"], tdoa["alt"], 'm--', alpha=0.6, label="TDOA")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude [m]")
ax.set_title(f"3D Trajectory Comparison for {uav_id}")
ax.legend()
plt.tight_layout()
plt.show()
