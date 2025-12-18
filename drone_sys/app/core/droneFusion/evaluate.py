# evaluate.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataset import MultiSourceGraphDataset, latlon_to_enu
from model import GraphFusionModel, NODE_FEAT_DIM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================
# ⚙ CONFIG
# ==============================================================
DATA_ROOT = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multi_source_test"
MODEL_PATH = "graph_fusion_model.pt"
WINDOW_SIZE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_FIG = True


# ==============================================================
# 加载模型 + norm
# ==============================================================
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    cfg = ckpt["config"]

    model = GraphFusionModel(
        in_dim=cfg["in_dim"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        window_size=cfg["window_size"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return (
        model,
        ckpt["x_mean"].to(DEVICE),
        ckpt["x_std"].to(DEVICE),
        ckpt["y_mean"].to(DEVICE),
        ckpt["y_std"].to(DEVICE),
    )


# ==============================================================
# → 误差函数
# ==============================================================
def calc_err(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=1)
    return dict(
        RMSE=float(np.sqrt(np.mean(dist ** 2))),
        MAE=float(np.mean(dist)),
        MAX=float(np.max(dist)),
    )


def calc_z_err(pred_z, gt_z):
    diff = np.abs(pred_z - gt_z)
    return dict(
        RMSE=float(np.sqrt(np.mean(diff ** 2))),
        MAE=float(np.mean(diff)),
        MAX=float(np.max(diff)),
    )


# ==============================================================
# 单模态误差：Radar / 5G-A / TDOA
# ==============================================================
def modality_metrics(df_truth, df_mod, uav, lat0, lon0, alt0):
    df_t = df_truth[df_truth["id"] == uav].sort_values("timestamp")
    df_m = df_mod[df_mod["id"] == uav]

    merged = df_t.merge(df_m, on="timestamp", suffixes=("", "_m"), how="inner")
    if len(merged) < 5:
        return None

    # truth ENU
    e_gt, n_gt, u_gt = latlon_to_enu(
        merged["lat"], merged["lon"], merged["alt"], lat0, lon0, alt0
    )
    # modality ENU
    e_m, n_m, u_m = latlon_to_enu(
        merged["lat_m"], merged["lon_m"], merged["alt_m"], lat0, lon0, alt0
    )

    return calc_err(np.stack([e_m, n_m, u_m], 1), np.stack([e_gt, n_gt, u_gt], 1))


# ==============================================================
# Evaluate 1 UAV
# ==============================================================
def evaluate_uav(model, batch_dir, uav, x_mean, x_std, y_mean, y_std):

    truth = pd.read_csv(os.path.join(batch_dir, "truth.csv"))
    radar = pd.read_csv(os.path.join(batch_dir, "radar.csv"))
    fiveg = pd.read_csv(os.path.join(batch_dir, "5g_a.csv"))
    tdoa  = pd.read_csv(os.path.join(batch_dir, "tdoa.csv"))

    df_t = truth[truth["id"] == uav].sort_values("timestamp")
    if len(df_t) < WINDOW_SIZE:
        return None

    # ENU 基准点
    lat0, lon0, alt0 = df_t.iloc[0][["lat", "lon", "alt"]]

    # truth ENU
    e_gt, n_gt, u_gt = latlon_to_enu(df_t["lat"], df_t["lon"], df_t["alt"], lat0, lon0, alt0)
    GT = np.stack([e_gt, n_gt, u_gt], axis=1)

    # 单模态误差
    radar_err = modality_metrics(truth, radar, uav, lat0, lon0, alt0)
    fiveg_err = modality_metrics(truth, fiveg, uav, lat0, lon0, alt0)
    tdoa_err  = modality_metrics(truth, tdoa,  uav, lat0, lon0, alt0)

    # 使用 dataset 的对齐方法
    tmp_dataset = MultiSourceGraphDataset(DATA_ROOT)
    align = tmp_dataset._align_mod

    radar_m = align(df_t, radar, uav, lat0, lon0, alt0)
    fiveg_m = align(df_t, fiveg, uav, lat0, lon0, alt0)
    tdoa_m  = align(df_t, tdoa,  uav, lat0, lon0, alt0)

    # 构造模型输入
    T = len(df_t)
    T_use = min(T, WINDOW_SIZE)
    X = np.zeros((1, T_use, 3, NODE_FEAT_DIM), dtype=np.float32)
    M = np.zeros((1, T_use, 3), dtype=np.float32)
    t_norm = np.linspace(0, 1, T_use)

    mods = [radar_m, fiveg_m, tdoa_m]
    for mi, m in enumerate(mods):
        X[0, :, mi, 0] = m["east"][:T_use]
        X[0, :, mi, 1] = m["north"][:T_use]
        X[0, :, mi, 2] = m["up"][:T_use]
        X[0, :, mi, 7] = m["conf"][:T_use]
        X[0, :, mi, 8] = t_norm

        X[0, :, mi, 3] = df_t["vx"].values[:T_use]
        X[0, :, mi, 4] = df_t["vy"].values[:T_use]
        X[0, :, mi, 5] = df_t["vz"].values[:T_use]
        X[0, :, mi, 6] = df_t["speed"].values[:T_use]

        # one-hot
        if mi == 0: X[0, :, mi, 9] = 1
        if mi == 1: X[0, :, mi, 10] = 1
        if mi == 2: X[0, :, mi, 11] = 1

        M[0, :, mi] = m["mask"][:T_use]

    # ⇒ 标准化
    X_std = (X - x_mean.cpu().numpy()) / x_std.cpu().numpy()
    X_t = torch.tensor(X_std, dtype=torch.float32).to(DEVICE)
    M_t = torch.tensor(M, dtype=torch.float32).to(DEVICE)

    # ⇒ 推理
    with torch.no_grad():
        pred_norm = model(X_t, M_t).detach().cpu().numpy()[0]
    pred = pred_norm * y_std.cpu().numpy() + y_mean.cpu().numpy()

    GT_use = GT[:T_use]
    pred_use = pred[:T_use]

    # Fusion 误差
    fusion_err_xy = calc_err(pred_use[:, :2], GT_use[:, :2])  # 平面误差
    z_err = calc_z_err(pred_use[:, 2], GT_use[:, 2])          # 高度误差

    # 2D 图
    fig = plt.figure(figsize=(6, 5))
    plt.plot(GT_use[:, 0], GT_use[:, 1], 'k-', label='Truth')
    plt.plot(pred_use[:, 0], pred_use[:, 1], 'r--', label='Fusion')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title(f"{uav} - {os.path.basename(batch_dir)}")
    plt.legend()

    if SAVE_FIG:
        sp = os.path.join(OUTPUT_DIR, f"{os.path.basename(batch_dir)}_{uav}_2d.png")
        plt.savefig(sp)
    plt.close()

    return fusion_err_xy, z_err, radar_err, fiveg_err, tdoa_err


# ==============================================================
# *** 增加误差对比柱状图 ***
# ==============================================================
def plot_modality_bar(uav, fusion, radar, fiveg, tdoa):
    labels = ["Fusion", "Radar", "5G-A", "TDOA"]
    values = [
        fusion["RMSE"],
        radar["RMSE"] if radar else np.nan,
        fiveg["RMSE"] if fiveg else np.nan,
        tdoa["RMSE"] if tdoa else np.nan,
    ]

    plt.figure(figsize=(7,5))
    plt.bar(labels, values, color=["red", "blue", "green", "purple"])
    plt.ylabel("RMSE (m)")
    plt.title(f"RMSE Comparison - {uav}")
    sp = os.path.join(OUTPUT_DIR, f"{uav}_modality_compare.png")
    plt.savefig(sp)
    plt.close()


def plot_height_bar(uav, z_fusion, z_radar, z_5g, z_tdoa):
    labels = ["Fusion", "Radar", "5G-A", "TDOA"]
    values = [
        z_fusion["RMSE"],
        z_radar["RMSE"] if z_radar else np.nan,
        z_5g["RMSE"] if z_5g else np.nan,
        z_tdoa["RMSE"] if z_tdoa else np.nan,
    ]
    plt.figure(figsize=(7,5))
    plt.bar(labels, values, color=["red", "blue", "green", "purple"])
    plt.ylabel("Z-RMSE (m)")
    plt.title(f"Height Error Comparison - {uav}")
    sp = os.path.join(OUTPUT_DIR, f"{uav}_height_compare.png")
    plt.savefig(sp)
    plt.close()


def plot_height_error_curve(uav, pred_z, gt_z):
    """
    画高度误差的逐时间步折线图
    """
    err = np.abs(pred_z - gt_z)  # [T_use]

    plt.figure(figsize=(7, 5))
    plt.plot(err, color="red", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Height Error |pred_z - gt_z| (m)")
    plt.title(f"Fusion Z-Error Curve - {uav}")
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, f"{uav}_fusion_z_error_curve.png")
    plt.savefig(save_path)
    plt.close()

# ==============================================================
# MAIN
# ==============================================================
def main():
    torch.set_grad_enabled(False)

    model, x_mean, x_std, y_mean, y_std = load_model()

    results = []
    cnt = 0

    for batch in sorted(os.listdir(DATA_ROOT)):
        batch_dir = os.path.join(DATA_ROOT, batch)
        if not os.path.isdir(batch_dir): continue

        truth = pd.read_csv(os.path.join(batch_dir, "truth.csv"))

        for uav in truth["id"].unique():
            if cnt >= 20: break

            out = evaluate_uav(model, batch_dir, uav, x_mean, x_std, y_mean, y_std)
            if out is None:
                continue

            (fusion_err, z_err,
             radar_err, fiveg_err, tdoa_err) = out

            print(f"[✓] {batch} - {uav}: Fusion RMSE = {fusion_err['RMSE']:.3f} m")

            # 保存可视化：模态对比柱状图 + 高度误差图
            plot_modality_bar(uav, fusion_err, radar_err, fiveg_err, tdoa_err)
            plot_height_bar(uav, z_err, radar_err, fiveg_err, tdoa_err)

            # 结果写入 CSV
            results.append([
                uav, batch,
                fusion_err["RMSE"], fusion_err["MAE"], fusion_err["MAX"],
                z_err["RMSE"], z_err["MAE"], z_err["MAX"],
                radar_err["RMSE"] if radar_err else None,
                fiveg_err["RMSE"] if fiveg_err else None,
                tdoa_err["RMSE"] if tdoa_err else None,
            ])

            cnt += 1
        if cnt >= 20:
            break

    df = pd.DataFrame(results, columns=[
        "uav","batch",
        "fusion_rmse","fusion_mae","fusion_max",
        "z_rmse","z_mae","z_max",
        "radar_rmse","fiveg_rmse","tdoa_rmse"
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "fusion_eval.csv"), index=False)

    print("\n[✔ 完成] 前10个 UAV 评估完成！")
    print(df.head())




if __name__ == "__main__":
    main()
