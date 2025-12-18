# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    MultiSourceGraphDataset,
    DATA_ROOT,
    WINDOW_SIZE,
    STRIDE,
)
from model import GraphFusionModel, NODE_FEAT_DIM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OpenMP 冲突

# ==============================
# ⚙ 可配置参数
# ==============================
DATA_DIR      = DATA_ROOT
BATCH_SIZE    = 64
EPOCHS        = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

D_MODEL       = 128
NUM_HEADS     = 4
NUM_LAYERS    = 3
DIM_FF        = 256
DROPOUT       = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "graph_fusion_model.pt"
NORM_PATH  = "graph_fusion_norm.pth"   # 可选：单独保存归一化参数（这里直接用 dataset 里的）


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x, y, m = batch   # x: [B, T, 3, F], y: [B, T, 3], m: [B, T, 3]
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        optimizer.zero_grad()
        pred = model(x, m)       # [B, T, 3] (归一化空间)
        loss = loss_fn(pred, y)  # MSE

        loss.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def main():
    print(f"[Config] DATA_DIR={DATA_DIR}")
    print(f"[Config] DEVICE={DEVICE}")

    # 1) 构建 Dataset & DataLoader
    dataset = MultiSourceGraphDataset(
        data_root=DATA_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # Windows 建议先用 0，避免多进程问题
        drop_last=False,
    )

    # 2) 构建模型
    model = GraphFusionModel(
        in_dim=NODE_FEAT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
        window_size=WINDOW_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(model)
    print(f"[Train] Start training for {EPOCHS} epochs.")

    # 3) 训练循环
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, DEVICE)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d}  |  loss = {avg_loss:.6f}")

    # 4) 保存模型 + 归一化参数
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": dataset.x_mean,
            "x_std": dataset.x_std,
            "y_mean": dataset.y_mean,
            "y_std": dataset.y_std,
            "config": {
                "in_dim": NODE_FEAT_DIM,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dim_ff": DIM_FF,
                "dropout": DROPOUT,
                "window_size": WINDOW_SIZE,
            },
        },
        MODEL_PATH,
    )
    print(f"[Save] Model + norm stats saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
