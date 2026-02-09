import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import MODALITIES, MODALITY_TO_ID, NODE_FEAT_DIM, enu_to_llh, latlon_to_enu
from model import GraphFusionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================================================
# CONFIG
# ===========================================================
DATA_ROOT = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multi_source_test"
BATCH_DIR = os.path.join(DATA_ROOT, "batch01")
MODEL_PATH = "graph_fusion_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UID = 2

WINDOW_SIZE = 20
STRIDE = 5


# ===========================================================
# ERROR
# ===========================================================
def calc_err(pred, gt):
    if len(pred) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAX": np.nan}
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(dist ** 2))),
        "MAE": float(np.mean(np.abs(dist))),
        "MAX": float(np.max(dist)),
    }


def merge_windows(preds, starts, t_total, window=20):
    fusion = np.zeros((t_total, 3), dtype=np.float32)
    count = np.zeros(t_total, dtype=np.float32)

    for p, s in zip(preds, starts):
        valid = min(window, t_total - s)
        fusion[s : s + valid] += p[:valid]
        count[s : s + valid] += 1

    return fusion / (count[:, None] + 1e-6)


def modality_series_enu(df_truth_u, df_mod_u, lat0, lon0, alt0):
    merged = df_truth_u[["timestamp", "lat", "lon", "alt"]].merge(
        df_mod_u[["timestamp", "lat", "lon", "alt"]],
        on="timestamp",
        how="inner",
        suffixes=("_t", "_m"),
    )
    if len(merged) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    e_t, n_t, u_t = latlon_to_enu(
        merged["lat_t"].values, merged["lon_t"].values, merged["alt_t"].values, lat0, lon0, alt0
    )
    e_m, n_m, u_m = latlon_to_enu(
        merged["lat_m"].values, merged["lon_m"].values, merged["alt_m"].values, lat0, lon0, alt0
    )
    gt = np.stack([e_t, n_t, u_t], axis=1).astype(np.float32)
    obs = np.stack([e_m, n_m, u_m], axis=1).astype(np.float32)
    return gt, obs


def build_sparse_windows(df_truth_u, mod_frames, lat0, lon0, alt0, window_size, stride):
    ts_all = df_truth_u["timestamp"].values.astype(np.int64)
    vx_all = df_truth_u["vx"].values.astype(np.float32)
    vy_all = df_truth_u["vy"].values.astype(np.float32)
    vz_all = df_truth_u["vz"].values.astype(np.float32)
    spd_all = df_truth_u["speed"].values.astype(np.float32)

    mod_maps = {}
    for m in MODALITIES:
        df_m = mod_frames[m]
        df_m = df_m[df_m["id"] == df_truth_u.iloc[0]["id"]]
        mod_maps[m] = {
            int(r.timestamp): r
            for r in df_m[["timestamp", "lat", "lon", "alt", "source_conf"]].itertuples(index=False)
        }

    windows = []
    starts = []
    t_total = len(df_truth_u)
    for s in range(0, t_total - window_size + 1, stride):
        ts_win = ts_all[s : s + window_size]
        vx = vx_all[s : s + window_size]
        vy = vy_all[s : s + window_size]
        vz = vz_all[s : s + window_size]
        spd = spd_all[s : s + window_size]

        feats = []
        t_ids = []
        m_ids = []

        denom = max(window_size - 1, 1)
        for ti in range(window_size):
            ts = int(ts_win[ti])
            for m in MODALITIES:
                row = mod_maps[m].get(ts)
                if row is None:
                    continue

                east, north, up = latlon_to_enu(row.lat, row.lon, row.alt, lat0, lon0, alt0)
                feat = np.zeros((NODE_FEAT_DIM,), dtype=np.float32)
                feat[0:3] = np.array([east, north, up], dtype=np.float32)
                feat[3] = vx[ti]
                feat[4] = vy[ti]
                feat[5] = vz[ti]
                feat[6] = spd[ti]
                feat[7] = float(row.source_conf)
                feat[8] = float(ti / denom)
                feat[9 + MODALITY_TO_ID[m]] = 1.0

                feats.append(feat)
                t_ids.append(ti)
                m_ids.append(MODALITY_TO_ID[m])

        if len(feats) == 0:
            continue

        windows.append(
            {
                "node_feat": np.stack(feats).astype(np.float32),
                "node_t": np.array(t_ids, dtype=np.int64),
                "node_m": np.array(m_ids, dtype=np.int64),
            }
        )
        starts.append(s)

    return windows, starts


def main():
    truth = pd.read_csv(os.path.join(BATCH_DIR, "truth.csv"))
    uav = truth["id"].unique()[UID]
    print("UAV:", uav)

    df_t = truth[truth["id"] == uav].sort_values("timestamp").reset_index(drop=True)
    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]

    e_gt, n_gt, u_gt = latlon_to_enu(df_t["lat"].values, df_t["lon"].values, df_t["alt"].values, lat0, lon0, alt0)
    truth_full_enu = np.stack([e_gt, n_gt, u_gt], axis=-1).astype(np.float32)

    mod_frames = {
        "gps": pd.read_csv(os.path.join(BATCH_DIR, "gps.csv")),
        "radar": pd.read_csv(os.path.join(BATCH_DIR, "radar.csv")),
        "5g_a": pd.read_csv(os.path.join(BATCH_DIR, "5g_a.csv")),
        "tdoa": pd.read_csv(os.path.join(BATCH_DIR, "tdoa.csv")),
    }

    windows, starts = build_sparse_windows(df_t, mod_frames, lat0, lon0, alt0, WINDOW_SIZE, STRIDE)
    if len(windows) == 0:
        raise RuntimeError("No valid sparse windows for inference")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg = ckpt["config"]

    model = GraphFusionModel(
        in_dim=cfg["in_dim"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        window_size=cfg["window_size"],
        num_modalities=cfg.get("num_modalities", 4),
        knn_k=cfg.get("knn_k", 8),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = ckpt["x_mean"].to(DEVICE)
    x_std = ckpt["x_std"].to(DEVICE)
    y_mean = ckpt["y_mean"].to(DEVICE)
    y_std = ckpt["y_std"].to(DEVICE)

    preds = []
    for w in windows:
        node_feat = torch.tensor(w["node_feat"], dtype=torch.float32, device=DEVICE)
        node_t = torch.tensor(w["node_t"], dtype=torch.long, device=DEVICE)
        node_m = torch.tensor(w["node_m"], dtype=torch.long, device=DEVICE)

        node_feat = (node_feat - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        with torch.no_grad():
            pred_norm = model(
                node_feat=node_feat.unsqueeze(0),
                node_t=node_t.unsqueeze(0),
                node_m=node_m.unsqueeze(0),
                node_mask=torch.ones((1, node_feat.shape[0]), dtype=torch.float32, device=DEVICE),
                window_size=WINDOW_SIZE,
            )[0]
        pred = pred_norm * y_std + y_mean
        preds.append(pred.cpu().numpy())

    fusion_enu = merge_windows(np.array(preds), starts, t_total=len(truth_full_enu), window=WINDOW_SIZE)

    fusion_err = calc_err(fusion_enu, truth_full_enu)
    print("\nFusion Error:", fusion_err)

    for m in MODALITIES:
        gt_m, obs_m = modality_series_enu(df_t, mod_frames[m][mod_frames[m]["id"] == uav], lat0, lon0, alt0)
        print(f"{m} Error:", calc_err(obs_m, gt_m))

    pred_lat, pred_lon, _ = enu_to_llh(fusion_enu[:, 0], fusion_enu[:, 1], fusion_enu[:, 2], lat0, lon0, alt0)
    truth_lat = df_t["lat"].values
    truth_lon = df_t["lon"].values

    plt.figure(figsize=(10, 8))
    plt.scatter(truth_lon, truth_lat, s=15, c="black", label="Truth")
    plt.scatter(pred_lon, pred_lat, s=15, c="red", label="Fusion")

    colors = {"gps": "orange", "radar": "blue", "5g_a": "green", "tdoa": "purple"}
    for m in MODALITIES:
        d = mod_frames[m]
        d = d[d["id"] == uav]
        if len(d) == 0:
            continue
        plt.scatter(d["lon"].values, d["lat"].values, s=8, c=colors[m], label=m, alpha=0.35)

    plt.title("Truth / Fusion / Modalities")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
