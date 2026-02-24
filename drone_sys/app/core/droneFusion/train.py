import os
import glob
import random
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    MODALITIES,
    MultiSourceGraphDataset,
    sparse_collate_fn,
)
from model import GraphFusionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DataConfig:
    data_dir: str = os.path.normpath(os.path.join(BASE_DIR, "../datasetBuilder/dataset-processed/train-datasets-finetuning/"))
    window_size: int = 20
    stride: int = 8
    truth_dt_s: float = 1.0
    align_tolerance_s: float = 0.55
    modalities: list = field(default_factory=lambda: list(MODALITIES))
    norm_stats_path: str = os.path.normpath(os.path.join(BASE_DIR, "model_result/graph_norm_v2.pth"))
    rebuild_norm_stats: bool = False
    max_batches: int = 0  # 0 means no limit
    batch_prefix: str = "batch"
    dataset_verbose: bool = True
    dataset_log_every_uav: int = 20
    dataset_build_workers: int = 8
    dataset_build_use_multiprocessing: bool = True
    dataset_use_sample_cache: bool = True
    dataset_rebuild_sample_cache: bool = False
    dataset_sample_cache_dir: str = ".cache/graph_samples_v2.6/"


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
    batch_size: int = 16
    epochs: int = 1
    lr: float = 4e-5
    weight_decay: float = 5e-5
    grad_clip: float = 1.0
    num_workers: int = 2
    loader_persistent_workers: bool = False
    loader_prefetch_factor: int = 2
    loader_multiprocessing_context: str = "spawn"
    pin_memory: bool = True
    log_every_step: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = os.path.normpath(os.path.join(BASE_DIR, "model_result/graph_fusion_model_v2.6.pt"))
    resume_model_path: str = os.path.normpath(os.path.join(BASE_DIR, "model_result/graph_fusion_model_v2.5.pt"))
    resume_if_model_exists: bool = False
    resume_strict: bool = False
    shuffle_units_each_epoch: bool = True
    unit_shuffle_seed: int = 20260223
    use_coverage_weighted_loss: bool = True
    loss_weight_full_alpha: float = 1.2
    loss_weight_missing_alpha: float = 0.4
    loss_weight_power: float = 2.0
    loss_weight_transition_alpha: float = 0.6
    use_confidence_weight: bool = True
    loss_weight_conf_alpha: float = 1.2
    loss_weight_conf_power: float = 2.0
    loss_main: str = "huber"  # huber | mse
    huber_beta: float = 1.0
    loss_vel_alpha: float = 0.15
    loss_acc_alpha: float = 0.08
    loss_fde_alpha: float = 0.30


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
    phase_tag="",
    train_cfg=None,
    num_modalities=None,
):
    model.train()
    loss_main_mode = str(getattr(train_cfg, "loss_main", "mse")).strip().lower() if train_cfg is not None else "mse"
    if loss_main_mode == "huber":
        beta = float(getattr(train_cfg, "huber_beta", 1.0)) if train_cfg is not None else 1.0
        try:
            loss_fn = nn.SmoothL1Loss(beta=max(beta, 1e-6), reduction="none")
        except TypeError:
            loss_fn = nn.SmoothL1Loss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    total_loss = 0.0
    total_samples = 0
    num_steps = len(loader)

    def _coverage_loss_weights(node_feat, node_t, node_m, node_mask, t_len, m_count, obs_json_batch, sample_meta_batch):
        if train_cfg is None or (not bool(getattr(train_cfg, "use_coverage_weighted_loss", False))):
            bsz = node_feat.size(0)
            ones = torch.ones((bsz,), dtype=node_feat.dtype, device=node_feat.device)
            return ones, None, None, None, None

        bsz = node_feat.size(0)
        m_count = max(1, int(m_count or 1))
        coverage_ratio = None
        full_modal_ratio = None
        conf_quality_ratio = None
        transition_ratio = None

        if isinstance(sample_meta_batch, dict) and len(sample_meta_batch) > 0:
            def _meta_1d(name):
                v = sample_meta_batch.get(name)
                if v is None:
                    return None
                if torch.is_tensor(v):
                    return v.to(device).reshape(-1).to(torch.float32)
                return None

            coverage_ratio = _meta_1d("coverage_ratio")
            full_modal_ratio = _meta_1d("full_modal_ratio")
            conf_quality_ratio = _meta_1d("conf_quality_ratio")
            transition_ratio = _meta_1d("transition_ratio")

        if coverage_ratio is None or full_modal_ratio is None:
            # Fallback path when sample_meta is absent.
            obs_valid = node_feat[..., 10] > 0.5
            valid = node_mask > 0.5
            valid_obs = valid & obs_valid
            t_idx = node_t.clamp(min=0, max=max(int(t_len) - 1, 0))
            present = torch.zeros((bsz, int(t_len), m_count), dtype=torch.float32, device=node_feat.device)
            for mid in range(m_count):
                mask_m = valid_obs & (node_m == mid)
                counts = torch.zeros((bsz, int(t_len)), dtype=torch.float32, device=node_feat.device)
                counts.scatter_add_(1, t_idx, mask_m.to(torch.float32))
                present[:, :, mid] = (counts > 0).to(torch.float32)
            coverage_ratio = present.mean(dim=(1, 2))
            full_modal_ratio = (present.sum(dim=-1) == float(m_count)).to(torch.float32).mean(dim=1)
            if transition_ratio is None and present.size(1) > 1:
                transition_ratio = (present[:, 1:, :] - present[:, :-1, :]).abs().mean(dim=(1, 2))
            elif transition_ratio is None:
                transition_ratio = torch.zeros((bsz,), dtype=torch.float32, device=node_feat.device)

        p = max(float(getattr(train_cfg, "loss_weight_power", 2.0)), 1e-6)
        full_alpha = float(getattr(train_cfg, "loss_weight_full_alpha", 0.0))
        miss_alpha = float(getattr(train_cfg, "loss_weight_missing_alpha", 0.0))
        trans_alpha = float(getattr(train_cfg, "loss_weight_transition_alpha", 0.0))
        weights = (
            1.0
            + full_alpha * torch.pow(full_modal_ratio.clamp(0.0, 1.0), p)
            + miss_alpha * torch.pow((1.0 - coverage_ratio).clamp(0.0, 1.0), p)
        )
        if transition_ratio is not None and trans_alpha > 0:
            weights = weights + trans_alpha * torch.pow(transition_ratio.clamp(0.0, 1.0), p)

        if bool(getattr(train_cfg, "use_confidence_weight", False)):
            conf_alpha = float(getattr(train_cfg, "loss_weight_conf_alpha", 0.0))
            conf_power = max(float(getattr(train_cfg, "loss_weight_conf_power", 2.0)), 1e-6)
            if conf_quality_ratio is None and conf_alpha > 0 and isinstance(obs_json_batch, list):
                conf_quality_ratio = torch.zeros((bsz,), dtype=torch.float32, device=node_feat.device)
                for bi, obs_seq in enumerate(obs_json_batch):
                    if not isinstance(obs_seq, list) or len(obs_seq) == 0:
                        continue
                    conf_sum = 0.0
                    conf_cnt = 0
                    observed_slots = 0
                    total_slots = 0
                    for t_item in obs_seq:
                        if not isinstance(t_item, dict):
                            continue
                        total_slots += int(m_count)
                        for _, mod_item in t_item.items():
                            if not isinstance(mod_item, dict):
                                continue
                            try:
                                obs_ok = float(mod_item.get("obs_valid", 1.0)) > 0.5
                            except Exception:
                                obs_ok = True
                            if not obs_ok:
                                continue
                            observed_slots += 1
                            try:
                                conf_v = float(mod_item.get("confidence", 0.0))
                            except Exception:
                                conf_v = 0.0
                            conf_sum += max(0.0, min(1.0, conf_v))
                            conf_cnt += 1
                    if conf_cnt > 0:
                        mean_conf = conf_sum / float(conf_cnt)
                        slot_cov = (observed_slots / float(total_slots)) if total_slots > 0 else 0.0
                        conf_quality_ratio[bi] = float(max(0.0, min(1.0, mean_conf * slot_cov)))
            if conf_alpha > 0 and conf_quality_ratio is not None:
                weights = weights + conf_alpha * torch.pow(conf_quality_ratio.clamp(0.0, 1.0), conf_power)
        # Keep the effective learning rate stable after reweighting.
        weights = weights / (weights.mean().detach() + 1e-6)
        return weights.to(node_feat.dtype), coverage_ratio, full_modal_ratio, conf_quality_ratio, transition_ratio

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
        per_elem_loss = loss_fn(pred, y)
        per_sample_main = per_elem_loss.mean(dim=(1, 2))
        sample_weights, coverage_ratio, full_modal_ratio, conf_quality_ratio, transition_ratio = _coverage_loss_weights(
            node_feat=node_feat,
            node_t=node_t,
            node_m=node_m,
            node_mask=node_mask,
            t_len=int(y.shape[1]),
            m_count=int(num_modalities or 1),
            obs_json_batch=batch.get("obs_json"),
            sample_meta_batch=batch.get("sample_meta"),
        )

        # Trajectory-shape auxiliary losses improve transition robustness and tail stability.
        per_sample_loss = per_sample_main
        vel_alpha = float(getattr(train_cfg, "loss_vel_alpha", 0.0)) if train_cfg is not None else 0.0
        if vel_alpha > 0 and pred.shape[1] > 1:
            vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
            vel_gt = y[:, 1:, :] - y[:, :-1, :]
            per_sample_vel = loss_fn(vel_pred, vel_gt).mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + vel_alpha * per_sample_vel

        acc_alpha = float(getattr(train_cfg, "loss_acc_alpha", 0.0)) if train_cfg is not None else 0.0
        if acc_alpha > 0 and pred.shape[1] > 2:
            acc_pred = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
            acc_gt = y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]
            per_sample_acc = loss_fn(acc_pred, acc_gt).mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + acc_alpha * per_sample_acc

        fde_alpha = float(getattr(train_cfg, "loss_fde_alpha", 0.0)) if train_cfg is not None else 0.0
        if fde_alpha > 0 and pred.shape[1] > 0:
            per_sample_fde = loss_fn(pred[:, -1:, :], y[:, -1:, :]).mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + fde_alpha * per_sample_fde

        loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-6)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = node_feat.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        avg_loss = total_loss / max(total_samples, 1)

        if log_every_step:
            extra = ""
            if coverage_ratio is not None and full_modal_ratio is not None:
                extra = (
                    f" | cov={coverage_ratio.mean().item():.3f}"
                    f" | full={full_modal_ratio.mean().item():.3f}"
                    f" | w={sample_weights.mean().item():.3f}"
                )
                if conf_quality_ratio is not None:
                    extra += f" | confq={conf_quality_ratio.mean().item():.3f}"
                if transition_ratio is not None:
                    extra += f" | trans={transition_ratio.mean().item():.3f}"
            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"{phase_tag} "
                f"Step {step:04d}/{num_steps:04d} | "
                f"loss={loss.item():.6f} | avg={avg_loss:.6f}{extra}"
            )

    return total_loss / max(total_samples, 1)


def list_data_units(data_dir: str, batch_prefix: str, max_batches: int):
    batch_dirs = sorted(glob.glob(os.path.join(data_dir, f"{batch_prefix}*")))
    if len(batch_dirs) == 0:
        return [data_dir]
    if max_batches is not None and max_batches > 0:
        return batch_dirs[:max_batches]
    return batch_dirs


def get_epoch_units(units, epoch: int, train_cfg: TrainConfig):
    epoch_units = list(units)
    if len(epoch_units) <= 1:
        return epoch_units
    if not bool(getattr(train_cfg, "shuffle_units_each_epoch", False)):
        return epoch_units
    seed_base = int(getattr(train_cfg, "unit_shuffle_seed", 0))
    random.Random(seed_base + int(epoch)).shuffle(epoch_units)
    return epoch_units


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
    if len(units) == 0:
        raise RuntimeError(f"no training units found under: {DATA_CFG.data_dir}")
    epoch1_units = get_epoch_units(units, epoch=1, train_cfg=TRAIN_CFG)
    if bool(getattr(TRAIN_CFG, "shuffle_units_each_epoch", False)):
        print(
            f"[Data] epoch01 unit shuffle enabled | seed={int(getattr(TRAIN_CFG, 'unit_shuffle_seed', 0)) + 1}"
        )

    # Build first unit to initialize model input dim + norm stats
    first_rebuild = DATA_CFG.rebuild_norm_stats
    first_ds, first_loader = build_loader_for_unit(
        unit_dir=epoch1_units[0],
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
        epoch_units = get_epoch_units(units, epoch=epoch, train_cfg=TRAIN_CFG)
        if bool(getattr(TRAIN_CFG, "shuffle_units_each_epoch", False)):
            seed_used = int(getattr(TRAIN_CFG, "unit_shuffle_seed", 0)) + int(epoch)
            head = ", ".join(os.path.basename(x) for x in epoch_units[:5])
            print(f"[Epoch {epoch:02d}] shuffled units | seed={seed_used} | head=[{head}]")
        unit_losses = []
        for ui, unit_dir in enumerate(epoch_units, start=1):
            rebuild = False
            print(f"[Load] Epoch {epoch:02d} unit {ui:03d}/{len(epoch_units):03d}: {unit_dir}")
            if epoch == 1 and ui == 1 and unit_dir == epoch1_units[0]:
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
                phase_tag=f"[unit {ui:03d}/{len(epoch_units):03d}]",
                train_cfg=TRAIN_CFG,
                num_modalities=len(DATA_CFG.modalities),
            )
            unit_losses.append(unit_loss)
            print(f"[Unit] Epoch {epoch:02d} unit {ui:03d} done | avg_loss={unit_loss:.6f}")

        epoch_loss = float(sum(unit_losses) / max(len(unit_losses), 1))
        print(f"[Epoch {epoch:02d}] done | avg_loss={epoch_loss:.6f}")

    model_out_dir = os.path.dirname(os.path.abspath(TRAIN_CFG.model_path))
    if model_out_dir:
        os.makedirs(model_out_dir, exist_ok=True)

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
