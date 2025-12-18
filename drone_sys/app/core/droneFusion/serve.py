import os
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from model import GraphFusionModel
from dataset import to_enu_single_point, NODE_FEAT_DIM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# 1. 加载模型 + 归一化
# ============================================================
MODEL_PATH = "graph_fusion_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
cfg = ckpt["config"]

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
model.eval()

x_mean = ckpt["x_mean"].to(DEVICE)
x_std  = ckpt["x_std"].to(DEVICE)
y_mean = ckpt["y_mean"].to(DEVICE)
y_std  = ckpt["y_std"].to(DEVICE)


# ============================================================
# 2. FastAPI 定义
# ============================================================
app = FastAPI(title="UAV Multi-Source Fusion API", version="1.0")


class Frame(BaseModel):
    timestamp: int
    lat: float
    lon: float
    alt: float
    vx: float
    vy: float
    vz: float
    speed: float
    source_conf: float


class InputModal(BaseModel):
    radar:  List[Frame]
    fiveg:  List[Frame]
    tdoa:   List[Frame]
    meta: Dict[str, float]   # { lat0, lon0, alt0 }


class RequestModel(BaseModel):
    data: InputModal


# ============================================================
# 3. 构造模型输入
# ============================================================
def build_model_input(mod, lat0, lon0, alt0):
    T = 20
    X = np.zeros((T, 3, NODE_FEAT_DIM), dtype=np.float32)
    M = np.ones((T, 3), dtype=np.float32)

    t_norm = np.linspace(0, 1, T)

    modalities = [mod.radar, mod.fiveg, mod.tdoa]

    for mi, frames in enumerate(modalities):
        for t in range(T):
            f = frames[t]

            # 经纬度 → ENU
            east, north, up = to_enu_single_point(
                f.lat, f.lon, f.alt,
                lat0, lon0, alt0
            )

            X[t, mi, 0] = east
            X[t, mi, 1] = north
            X[t, mi, 2] = up

            X[t, mi, 3] = f.vx
            X[t, mi, 4] = f.vy
            X[t, mi, 5] = f.vz
            X[t, mi, 6] = f.speed
            X[t, mi, 7] = f.source_conf
            X[t, mi, 8] = t_norm[t]

            # one-hot
            if mi == 0: X[t, mi, 9] = 1
            if mi == 1: X[t, mi,10] = 1
            if mi == 2: X[t, mi,11] = 1

    return X, M


# ============================================================
# 4. 主推理接口
# ============================================================
@app.post("/fusion")
def fuse(req: RequestModel):

    mod = req.data
    lat0 = mod.meta["lat0"]
    lon0 = mod.meta["lon0"]
    alt0 = mod.meta["alt0"]

    # ---- 构造模型输入 ----
    X_np, M_np = build_model_input(mod, lat0, lon0, alt0)

    X = torch.tensor(X_np).unsqueeze(0).to(DEVICE)
    M = torch.tensor(M_np).unsqueeze(0).to(DEVICE)

    # ---- 归一化 ----
    X_norm = (X - x_mean.reshape(1,1,1,-1)) / x_std.reshape(1,1,1,-1)

    # ---- 推理 ----
    with torch.no_grad():
        pred_norm = model(X_norm, M)        # [1,20,3]
        pred = pred_norm * y_std + y_mean   # 反归一化

    out = pred.squeeze(0).cpu().numpy()     # [20,3]

    # ---- 输出 JSON ----
    result = []
    for i, f in enumerate(mod.radar):
        result.append({
            "timestamp": f.timestamp,
            "east":  float(out[i,0]),
            "north": float(out[i,1]),
            "up":    float(out[i,2])
        })

    return {"fusion": result}
