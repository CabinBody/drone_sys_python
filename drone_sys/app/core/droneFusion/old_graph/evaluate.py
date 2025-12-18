import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from model import FusionModel
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# =========================
# ⚙ Configuration
# =========================
DATA_DIR   = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_test_80"
MODEL_PATH = "../pt_backup/fusion_model.pt"
NORM_PATH  = "../pt_backup/fusion_norm.pth"
SEQ_LEN    = 10
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 🔧 Load Model & Normalization Params
# =========================
norm = torch.load(NORM_PATH)
model = FusionModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = MultiModalDataset(DATA_DIR, seq_len=SEQ_LEN, normalize=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 📊 Inference
# =========================
preds, gps_last, radar_last, targets = [], [], [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, aux = model(x)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
        gps_last.append(x[:, -1, 0, :3].cpu().numpy())
        radar_last.append(x[:, -1, 1, :3].cpu().numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)
gps_last = np.concatenate(gps_last)
radar_last = np.concatenate(radar_last)

# Denormalize
y_mean = np.array(list(norm["y_mean"].values()))
y_std = np.array(list(norm["y_std"].values()))
preds = preds * y_std + y_mean
targets = targets * y_std + y_mean
gps_last = gps_last * y_std + y_mean
radar_last = radar_last * y_std + y_mean

# =========================
# 📈 Metrics
# =========================
def calc_metrics(p, t):
    mse = np.mean((p - t) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(p - t))
    r2 = r2_score(t, p)
    corr, _ = pearsonr(p.flatten(), t.flatten())
    euc = np.mean(np.linalg.norm(p - t, axis=1))
    return dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, PCC=corr, EUC=euc)

metrics_gps = calc_metrics(gps_last, targets)
metrics_radar = calc_metrics(radar_last, targets)
metrics_fusion = calc_metrics(preds, targets)

# =========================
# 📋 Print Results
# =========================
print("\n✅ Evaluation Results (Position Metrics)")
for name, m in zip(["GPS", "Radar", "Fusion"], [metrics_gps, metrics_radar, metrics_fusion]):
    print(f"\n{name} Modality:")
    for k, v in m.items():
        print(f"  {k:<6s}: {v:.6f}")

# =========================
# 📊 Visualization
# =========================

# 1️⃣ Scatter Plot (ENU plane)
plt.figure(figsize=(8,6))
plt.scatter(targets[:,0], targets[:,1], c='green', s=40, label="Truth")
plt.scatter(gps_last[:,0], gps_last[:,1], c='blue', alpha=0.4, label=f"GPS RMSE={metrics_gps['RMSE']:.2f}")
plt.scatter(radar_last[:,0], radar_last[:,1], c='orange', alpha=0.4, label=f"Radar RMSE={metrics_radar['RMSE']:.2f}")
plt.scatter(preds[:,0], preds[:,1], c='red', alpha=0.6, label=f"Fusion RMSE={metrics_fusion['RMSE']:.2f}")
plt.legend()
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Predicted vs Ground Truth (ENU plane)")
plt.grid(True)
plt.show()

# 2️⃣ Error Distribution Histogram
err_g = np.linalg.norm(gps_last - targets, axis=1)
err_r = np.linalg.norm(radar_last - targets, axis=1)
err_f = np.linalg.norm(preds - targets, axis=1)

plt.figure(figsize=(8,5))
plt.hist(err_g, bins=40, alpha=0.5, label="GPS")
plt.hist(err_r, bins=40, alpha=0.5, label="Radar")
plt.hist(err_f, bins=40, alpha=0.7, label="Fusion")
plt.xlabel("Position Error (m)")
plt.ylabel("Count")
plt.title("Error Distribution Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 3️⃣ Boxplot of Errors
plt.figure(figsize=(6,5))
plt.boxplot([err_g, err_r, err_f], labels=["GPS","Radar","Fusion"])
plt.ylabel("Error (m)")
plt.title("Error Distribution Boxplot")
plt.grid(True)
plt.show()

# 4️⃣ Temporal Error Trend
sample_idx = np.random.randint(0, len(dataset))
x, y = dataset[sample_idx]
x = x.unsqueeze(0).to(DEVICE)
pred, _ = model(x)
pred = pred.detach().cpu().numpy().squeeze() * y_std + y_mean
y = y.numpy() * y_std + y_mean

gps_seq = x[0, :, 0, :3].cpu().numpy() * y_std + y_mean
rad_seq = x[0, :, 1, :3].cpu().numpy() * y_std + y_mean
t_axis = np.arange(len(gps_seq))

plt.figure(figsize=(7,5))
plt.plot(t_axis, np.linalg.norm(gps_seq - y, axis=1), 'b--', label="GPS Error")
plt.plot(t_axis, np.linalg.norm(rad_seq - y, axis=1), 'orange', label="Radar Error")
plt.axhline(np.linalg.norm(pred - y), color='r', linestyle='-', label="Fusion Final Error")
plt.xlabel("Time Step")
plt.ylabel("Error (m)")
plt.title("Temporal Error Trend")
plt.legend()
plt.grid(True)
plt.show()
