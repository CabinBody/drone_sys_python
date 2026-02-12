import os
import glob
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    DATA_ROOT,
    MODALITIES,
    MultiSourceGraphDataset,
    sparse_collate_fn,
)
from model import GraphFusionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class DataConfig:
    data_dir: str = r"../datasetBuilder/dataset-processed/train-datasets"
    window_size: int = 20
    stride: int = 8
    truth_dt_s: float = 1.0
    align_tolerance_s: float = 0.55
    modalities: list = field(default_factory=lambda: list(MODALITIES))
    norm_stats_path: str = "graph_norm_stats_processed_sparse_enu.pth"
    rebuild_norm_stats: bool = False
    max_batches: int = 0  # 0 means no limit
    batch_prefix: str = "batch"
    dataset_verbose: bool = True
    dataset_log_every_uav: int = 20
    dataset_build_workers: int = 15
    dataset_build_use_multiprocessing: bool = True
    dataset_use_sample_cache: bool = True
    dataset_rebuild_sample_cache: bool = False
    dataset_sample_cache_dir: str = ".cache/graph_samples"


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.15
    knn_k: int = 6


@dataclass
class TrainConfig:
    batch_size: int = 12
    epochs: int = 15
    lr: float = 4e-4
    weight_decay: float = 5e-5
    grad_clip: float = 1.0
    num_workers: int = 16
    loader_persistent_workers: bool = True
    loader_prefetch_factor: int = 2
    loader_multiprocessing_context: str = "spawn"
    pin_memory: bool = True
    log_every_step: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "graph_fusion_model_processed.pt"
    resume_model_path: str = ""
    resume_if_model_exists: bool = True
    resume_strict: bool = True


DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is not a dict")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        state = payload["model_state_dict"]
    elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    elif len(payload) > 0 and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state = payload
    else:
        raise RuntimeError("checkpoint has no model_state_dict/state_dict")
    if len(state) > 0 and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _resolve_resume_path(train_cfg: TrainConfig):
    explicit = str(train_cfg.resume_model_path or "").strip()
    if explicit:
        return explicit
    if bool(train_cfg.resume_if_model_exists) and os.path.exists(train_cfg.model_path):
        return train_cfg.model_path
    return ""


def _maybe_restore_norm_stats_from_checkpoint(norm_stats_path: str, resume_path: str, rebuild_norm_stats: bool):
    if bool(rebuild_norm_stats):
        print("[Norm] rebuild_norm_stats=True, skip restoring norm stats from checkpoint")
        return
    if os.path.exists(norm_stats_path):
        print(f"[Norm] using existing norm stats: {norm_stats_path}")
        return
    if not resume_path:
        print("[Norm] no resume checkpoint and norm stats missing; will rebuild from current dataset")
        return

    payload = _safe_torch_load(resume_path)
    if not isinstance(payload, dict):
        print("[Norm] resume checkpoint is not dict; cannot restore norm stats")
        return
    keys = ("x_mean", "x_std", "y_mean", "y_std")
    if not all(k in payload for k in keys):
        print("[Norm] resume checkpoint has no x/y mean/std; cannot restore norm stats")
        return

    out_dir = os.path.dirname(os.path.abspath(norm_stats_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "x_mean": torch.as_tensor(payload["x_mean"]).cpu(),
            "x_std": torch.as_tensor(payload["x_std"]).cpu(),
            "y_mean": torch.as_tensor(payload["y_mean"]).cpu(),
            "y_std": torch.as_tensor(payload["y_std"]).cpu(),
        },
        norm_stats_path,
    )
    print(f"[Norm] restored norm stats from {resume_path} -> {norm_stats_path}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    epochs,
    log_every_step=True,
    grad_clip=1.0,
    phase_tag=""
):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0
    num_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        node_feat = batch["node_feat"].to(device)
        node_t = batch["node_t"].to(device)
        node_m = batch["node_m"].to(device)
        node_mask = batch["node_mask"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(
            node_feat=node_feat,
            node_t=node_t,
            node_m=node_m,
            node_mask=node_mask,
            window_size=y.shape[1],
        )
        loss = loss_fn(pred, y)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = node_feat.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        avg_loss = total_loss / max(total_samples, 1)

        if log_every_step:
            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"{phase_tag} "
                f"Step {step:04d}/{num_steps:04d} | "
                f"loss={loss.item():.6f} | avg={avg_loss:.6f}"
            )

    return total_loss / max(total_samples, 1)


def list_data_units(data_dir: str, batch_prefix: str, max_batches: int):
    batch_dirs = sorted(glob.glob(os.path.join(data_dir, f"{batch_prefix}*")))
    if len(batch_dirs) == 0:
        return [data_dir]
    if max_batches is not None and max_batches > 0:
        return batch_dirs[:max_batches]
    return batch_dirs


def build_loader_for_unit(unit_dir: str, cfg: DataConfig, train_cfg: TrainConfig, rebuild_norm_stats: bool):
    dataset = MultiSourceGraphDataset(
        data_root=unit_dir,
        window_size=cfg.window_size,
        stride=cfg.stride,
        modalities=cfg.modalities,
        truth_dt_s=cfg.truth_dt_s,
        align_tolerance_s=cfg.align_tolerance_s,
        norm_stats_path=cfg.norm_stats_path,
        rebuild_norm_stats=rebuild_norm_stats,
        max_batches=None,
        verbose=cfg.dataset_verbose,
        log_every_uav=cfg.dataset_log_every_uav,
        build_workers=cfg.dataset_build_workers,
        build_use_multiprocessing=cfg.dataset_build_use_multiprocessing,
        use_sample_cache=cfg.dataset_use_sample_cache,
        rebuild_sample_cache=cfg.dataset_rebuild_sample_cache,
        sample_cache_dir=cfg.dataset_sample_cache_dir,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=False,
        pin_memory=bool(train_cfg.pin_memory and str(train_cfg.device).startswith("cuda")),
        collate_fn=sparse_collate_fn,
    )
    if train_cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.loader_persistent_workers)
        loader_kwargs["prefetch_factor"] = int(train_cfg.loader_prefetch_factor)
        if train_cfg.loader_multiprocessing_context:
            loader_kwargs["multiprocessing_context"] = train_cfg.loader_multiprocessing_context
    loader = DataLoader(**loader_kwargs)
    return dataset, loader


def main():
    print("[Config] data:", asdict(DATA_CFG))
    print("[Config] model:", asdict(MODEL_CFG))
    print("[Config] train:", asdict(TRAIN_CFG))

    resume_path = _resolve_resume_path(TRAIN_CFG)
    if resume_path:
        print(f"[Resume] checkpoint: {resume_path}")
    else:
        print("[Resume] disabled (train from scratch)")

    _maybe_restore_norm_stats_from_checkpoint(
        norm_stats_path=DATA_CFG.norm_stats_path,
        resume_path=resume_path,
        rebuild_norm_stats=DATA_CFG.rebuild_norm_stats,
    )

    units = list_data_units(
        data_dir=DATA_CFG.data_dir,
        batch_prefix=DATA_CFG.batch_prefix,
        max_batches=DATA_CFG.max_batches,
    )
    print(f"[Data] training units: {len(units)}")

    # Build first unit to initialize model input dim + norm stats
    first_rebuild = DATA_CFG.rebuild_norm_stats
    first_ds, first_loader = build_loader_for_unit(
        unit_dir=units[0],
        cfg=DATA_CFG,
        train_cfg=TRAIN_CFG,
        rebuild_norm_stats=first_rebuild,
    )

    model = GraphFusionModel(
        in_dim=first_ds.node_feat_dim,
        d_model=MODEL_CFG.d_model,
        num_heads=MODEL_CFG.num_heads,
        num_layers=MODEL_CFG.num_layers,
        dim_ff=MODEL_CFG.dim_ff,
        dropout=MODEL_CFG.dropout,
        window_size=DATA_CFG.window_size,
        num_modalities=len(DATA_CFG.modalities),
        knn_k=MODEL_CFG.knn_k,
    ).to(TRAIN_CFG.device)

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_payload = _safe_torch_load(resume_path)
        resume_state = _extract_state_dict(resume_payload)
        load_ret = model.load_state_dict(resume_state, strict=bool(TRAIN_CFG.resume_strict))
        if bool(TRAIN_CFG.resume_strict):
            print("[Resume] model weights loaded (strict=True)")
        else:
            missing = len(getattr(load_ret, "missing_keys", []))
            unexpected = len(getattr(load_ret, "unexpected_keys", []))
            print(f"[Resume] model weights loaded (strict=False) | missing={missing} unexpected={unexpected}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CFG.lr,
        weight_decay=TRAIN_CFG.weight_decay,
    )

    print(model)
    print(f"[Train] start: epochs={TRAIN_CFG.epochs}, units/epoch={len(units)}")

    for epoch in range(1, TRAIN_CFG.epochs + 1):
        unit_losses = []
        for ui, unit_dir in enumerate(units, start=1):
            rebuild = False
            print(f"[Load] Epoch {epoch:02d} unit {ui:03d}/{len(units):03d}: {unit_dir}")
            if epoch == 1 and ui == 1:
                ds, loader = first_ds, first_loader
                print("[Load] reuse warmup loader for first unit")
            else:
                ds, loader = build_loader_for_unit(
                    unit_dir=unit_dir,
                    cfg=DATA_CFG,
                    train_cfg=TRAIN_CFG,
                    rebuild_norm_stats=rebuild,
                )
            if len(ds) == 0:
                print(f"[WARN] empty dataset in {unit_dir}, skip")
                continue
            unit_loss = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=TRAIN_CFG.device,
                epoch=epoch,
                epochs=TRAIN_CFG.epochs,
                log_every_step=TRAIN_CFG.log_every_step,
                grad_clip=TRAIN_CFG.grad_clip,
                phase_tag=f"[unit {ui:03d}/{len(units):03d}]",
            )
            unit_losses.append(unit_loss)
            print(f"[Unit] Epoch {epoch:02d} unit {ui:03d} done | avg_loss={unit_loss:.6f}")

        epoch_loss = float(sum(unit_losses) / max(len(unit_losses), 1))
        print(f"[Epoch {epoch:02d}] done | avg_loss={epoch_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": first_ds.x_mean,
            "x_std": first_ds.x_std,
            "y_mean": first_ds.y_mean,
            "y_std": first_ds.y_std,
            "config": {
                "data": asdict(DATA_CFG),
                "model": asdict(MODEL_CFG),
                "train": asdict(TRAIN_CFG),
                "in_dim": first_ds.node_feat_dim,
                "window_size": DATA_CFG.window_size,
                "num_modalities": len(DATA_CFG.modalities),
            },
        },
        TRAIN_CFG.model_path,
    )
    print(f"[Save] checkpoint -> {TRAIN_CFG.model_path}")


if __name__ == "__main__":
    main()
