import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from model import FusionModel

# ==============================
# ⚙️ Training Configuration
# ==============================
DATA_ROOT    = r"D:\MyCode\drone-fusion\datasetBuilder\dataset\scenario_multisource_1000x120"
SAVE_PATH    = "../pt_backup/fusion_model.pt"
NORM_PATH    = "../pt_backup/fusion_norm.pth"

EPOCHS_PER_BATCH = 3        # 每个批次训练轮数
BATCH_SIZE   = 64
SEQ_LEN      = 20
LEARNING_RATE= 4e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_INTERVAL = 20
WEIGHT_DECAY = 4e-3
GRAD_CLIP    = 1.0

# ==============================
# 🚀 One training epoch
# ==============================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, log_parts, steps = 0.0, {"task":0.0,"decor":0.0,"balance":0.0,"fuse":0.0}, 0

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred, aux = model(x)
        loss, parts = model.loss_fn(pred, y, aux)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        for k in log_parts: log_parts[k] += parts[k]
        steps += 1

        if step % PRINT_INTERVAL == 0:
            with torch.no_grad():
                g_mean = aux["g"].mean(dim=(0,1,3)).detach().cpu().numpy()
            print(f"  Step {step}/{len(loader)} | loss={loss.item():.4f} | "
                  f"task={parts['task']:.4f} decor={parts['decor']:.4f} "
                  f"balance={parts['balance']:.4f} fuse={parts['fuse']:.4f} | g_mean={g_mean}")

    avg = total_loss / max(1, steps)
    for k in log_parts: log_parts[k] /= max(1, steps)
    return avg, log_parts

# ==============================
# 🧠 Main training loop
# ==============================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 1️⃣ Collect all batch folders
    batch_dirs = sorted(
        [os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) if d.startswith("batch")],
        key=lambda x: x.lower()
    )
    print(f"🗂 Found {len(batch_dirs)} batches: {[os.path.basename(d) for d in batch_dirs]}")

    # 2️⃣ Initialize model
    model = FusionModel(
        input_dim=11, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
        tau=0.3, quality_idx=(6,7,8,9,10),
        ce_indices=(7,8,10),
        lambdas=(1e-2, 1e-3, 1e-2)
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3️⃣ If previous checkpoint exists, continue training
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print(f"🔄 Loaded existing weights from {SAVE_PATH}")

    # 4️⃣ Train sequentially on each batch folder
    for bi, batch_dir in enumerate(batch_dirs, 1):
        print(f"\n=== 🚀 Training on batch {bi}/{len(batch_dirs)}: {batch_dir} ===")

        dataset = MultiModalDataset(batch_dir, seq_len=SEQ_LEN, normalize=True)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

        for epoch in range(1, EPOCHS_PER_BATCH + 1):
            avg, parts = train_one_epoch(model, loader, optimizer, DEVICE)
            print(f"✅ Batch{bi} Epoch {epoch:02d} | avg_loss={avg:.4f} | "
                  f"task={parts['task']:.4f} decor={parts['decor']:.4f} "
                  f"balance={parts['balance']:.4f} fuse={parts['fuse']:.4f}")

        # 清理内存，释放 GPU
        del dataset, loader
        torch.cuda.empty_cache()

    # 5️⃣ Save final model and normalization stats
    final_dataset = MultiModalDataset(batch_dirs[0], seq_len=SEQ_LEN, normalize=True)
    torch.save(model.state_dict(), SAVE_PATH)
    torch.save({
        "y_mean": final_dataset.y_mean.to_dict(),
        "y_std":  final_dataset.y_std.to_dict(),
        "x_mean": final_dataset.x_mean.to_dict(),
        "x_std":  final_dataset.x_std.to_dict(),
    }, NORM_PATH)

    print(f"\n🎯 Training complete!")
    print(f"💾 Final model saved to {os.path.abspath(SAVE_PATH)}")
    print(f"💾 Normalization params saved to {os.path.abspath(NORM_PATH)}")
