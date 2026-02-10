import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import md5
from multiprocessing import get_context
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==============================================================
# CONFIG
# ==============================================================
DATA_ROOT = r"../datasetBuilder/dataset-processed/scenario_multi_source_5000x120/"

WINDOW_SIZE = 20
STRIDE = 5

MODALITIES = ["gps", "radar", "5g_a", "tdoa", "acoustic"]
MODALITY_TO_ID = {m: i for i, m in enumerate(MODALITIES)}
N_MODALITIES = len(MODALITIES)

R_EARTH = 6378137.0

# Node feature layout:
# 0:3   = east/north/up
# 3:6   = vx/vy/vz
# 6     = speed
# 7     = confidence
# 8     = t_norm
# 9     = pos_valid
# 10    = obs_valid (1 - missing_flag)
# 11    = detected_flag (for acoustic)
# 12    = spl_norm
# 13    = acoustic_energy
# 14:   = modality one-hot
BASE_FEAT_DIM = 14
NODE_FEAT_DIM = BASE_FEAT_DIM + N_MODALITIES
ONEHOT_OFFSET = BASE_FEAT_DIM

IDX_CONF = 7
IDX_TNORM = 8
IDX_POS_VALID = 9

NORM_STATS_PATH = "graph_norm_stats_processed_sparse_enu.pth"


@dataclass
class FusionDatasetConfig:
    data_root: str = DATA_ROOT
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    modalities: Optional[List[str]] = None
    truth_dt_s: float = 1.0
    align_tolerance_s: float = 0.55
    norm_stats_path: str = NORM_STATS_PATH
    rebuild_norm_stats: bool = False
    max_batches: Optional[int] = None
    verbose: bool = True
    log_every_uav: int = 25
    build_workers: int = 4
    build_use_multiprocessing: bool = True
    use_sample_cache: bool = True
    rebuild_sample_cache: bool = False
    sample_cache_dir: str = ".cache/graph_samples"


# ==============================================================
# 经纬度 <-> ENU
# ==============================================================
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east = dlon * np.cos(np.radians(lat0)) * R_EARTH
    north = dlat * R_EARTH
    up = alt - alt0
    return east, north, up


def to_enu_single_point(lat, lon, alt, lat0, lon0, alt0):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r
    east = dlon * np.cos(lat0_r) * R_EARTH
    north = dlat * R_EARTH
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float32)


def enu_to_llh(east, north, up, lat0, lon0, alt0):
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    lat = north / R_EARTH + lat0_r
    lon = east / (R_EARTH * np.cos(lat0_r)) + lon0_r
    alt = up + alt0
    return np.degrees(lat), np.degrees(lon), alt


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _nearest_truth_index(obs_ts: np.ndarray, truth_ts: np.ndarray):
    idx_hi = np.searchsorted(truth_ts, obs_ts, side="left")
    idx_lo = np.clip(idx_hi - 1, 0, len(truth_ts) - 1)
    idx_hi = np.clip(idx_hi, 0, len(truth_ts) - 1)

    d_lo = np.abs(obs_ts - truth_ts[idx_lo])
    d_hi = np.abs(obs_ts - truth_ts[idx_hi])
    choose_hi = d_hi < d_lo
    best_idx = np.where(choose_hi, idx_hi, idx_lo)
    best_diff = np.where(choose_hi, d_hi, d_lo)
    return best_idx, best_diff


def _normalize_spl(spl_db: float) -> float:
    # Keep same range as dataset generator defaults
    return float(np.clip((spl_db - 35.0) / 75.0, 0.0, 1.0))


def _build_node_feature(
    row: pd.Series,
    modality: str,
    ti_local: int,
    t_denom: int,
    lat0: float,
    lon0: float,
    alt0: float,
    n_modalities: int,
):
    feat = np.zeros((BASE_FEAT_DIM + n_modalities,), dtype=np.float32)

    lat = _safe_float(row.get("lat", np.nan), np.nan)
    lon = _safe_float(row.get("lon", np.nan), np.nan)
    alt = _safe_float(row.get("alt", np.nan), np.nan)
    has_pos = np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)

    if has_pos:
        e, n, u = latlon_to_enu(lat, lon, alt, lat0, lon0, alt0)
        feat[0:3] = np.array([e, n, u], dtype=np.float32)
        feat[IDX_POS_VALID] = 1.0
    else:
        feat[0:3] = 0.0
        feat[IDX_POS_VALID] = 0.0

    feat[3] = _safe_float(row.get("vx", 0.0), 0.0)
    feat[4] = _safe_float(row.get("vy", 0.0), 0.0)
    feat[5] = _safe_float(row.get("vz", 0.0), 0.0)
    feat[6] = _safe_float(row.get("speed", 0.0), 0.0)

    conf = row.get("confidence", row.get("source_conf", 0.0))
    feat[IDX_CONF] = float(np.clip(_safe_float(conf, 0.0), 0.0, 1.0))
    feat[IDX_TNORM] = float(ti_local / max(t_denom, 1))

    missing_flag = int(_safe_float(row.get("missing_flag", 0), 0))
    feat[10] = 0.0 if missing_flag > 0 else 1.0

    if modality == "acoustic":
        feat[11] = float(np.clip(_safe_float(row.get("detected_flag", 0), 0.0), 0.0, 1.0))
        spl_db = _safe_float(row.get("spl_db", np.nan), np.nan)
        feat[12] = _normalize_spl(spl_db) if np.isfinite(spl_db) else 0.0
        feat[13] = float(np.clip(_safe_float(row.get("acoustic_energy", 0.0), 0.0), 0.0, 1.0))

    feat[ONEHOT_OFFSET + MODALITY_TO_ID[modality]] = 1.0
    return feat


def build_input_feature(df_truth: pd.DataFrame, mod_frames: Dict[str, pd.DataFrame], modalities: Optional[List[str]] = None):
    """
    Build variable-node input for one UAV full sequence (without sliding windows).
    """
    cfg_modalities = modalities if modalities is not None else MODALITIES
    df_truth = df_truth.sort_values("timestamp").reset_index(drop=True)
    if len(df_truth) == 0:
        return {
            "node_feat": np.zeros((0, NODE_FEAT_DIM), dtype=np.float32),
            "node_t": np.zeros((0,), dtype=np.int64),
            "node_m": np.zeros((0,), dtype=np.int64),
            "y_true": np.zeros((0, 3), dtype=np.float32),
            "obs_json": [],
        }

    lat0, lon0, alt0 = df_truth.iloc[0][["lat", "lon", "alt"]]
    e_t, n_t, u_t = latlon_to_enu(df_truth["lat"].values, df_truth["lon"].values, df_truth["alt"].values, lat0, lon0, alt0)
    y_true = np.stack([e_t, n_t, u_t], axis=-1).astype(np.float32)

    ts_truth = pd.to_numeric(df_truth["timestamp"], errors="coerce").to_numpy(dtype=float)
    feats: List[np.ndarray] = []
    t_ids: List[int] = []
    m_ids: List[int] = []
    obs_json: List[Dict] = [dict() for _ in range(len(df_truth))]

    for m in cfg_modalities:
        df_m = mod_frames.get(m, pd.DataFrame())
        if df_m is None or len(df_m) == 0:
            continue
        if "timestamp" not in df_m.columns:
            continue

        ts_obs = pd.to_numeric(df_m["timestamp"], errors="coerce").to_numpy(dtype=float)
        valid_ts = np.isfinite(ts_obs)
        if valid_ts.sum() == 0:
            continue

        rows = df_m.loc[valid_ts].reset_index(drop=True)
        ts_obs = ts_obs[valid_ts]
        tidx, _ = _nearest_truth_index(ts_obs, ts_truth)

        for ri, row in rows.iterrows():
            ti = int(tidx[ri])
            feat = _build_node_feature(
                row=row,
                modality=m,
                ti_local=ti,
                t_denom=max(len(df_truth) - 1, 1),
                lat0=float(lat0),
                lon0=float(lon0),
                alt0=float(alt0),
                n_modalities=len(cfg_modalities),
            )
            feats.append(feat)
            t_ids.append(ti)
            m_ids.append(MODALITY_TO_ID[m])
            obs_json[ti][m] = {
                "confidence": float(feat[IDX_CONF]),
                "pos_valid": float(feat[IDX_POS_VALID]),
                "obs_valid": float(feat[10]),
            }

    if len(feats) == 0:
        node_feat = np.zeros((0, NODE_FEAT_DIM), dtype=np.float32)
        node_t = np.zeros((0,), dtype=np.int64)
        node_m = np.zeros((0,), dtype=np.int64)
    else:
        node_feat = np.stack(feats).astype(np.float32)
        node_t = np.array(t_ids, dtype=np.int64)
        node_m = np.array(m_ids, dtype=np.int64)

    return {
        "node_feat": node_feat,
        "node_t": node_t,
        "node_m": node_m,
        "y_true": y_true,
        "obs_json": obs_json,
    }


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_uav_samples_worker(payload):
    df_t, uav_mod_frames, modalities, window_size, stride, align_tolerance_s = payload

    if len(df_t) < window_size:
        return []

    lat0 = float(df_t.iloc[0]["lat"])
    lon0 = float(df_t.iloc[0]["lon"])
    alt0 = float(df_t.iloc[0]["alt"])

    e_t, n_t, u_t = latlon_to_enu(df_t["lat"].values, df_t["lon"].values, df_t["alt"].values, lat0, lon0, alt0)
    y_full = np.stack([e_t, n_t, u_t], axis=-1).astype(np.float32)
    ts_truth = pd.to_numeric(df_t["timestamp"], errors="coerce").to_numpy(dtype=float)

    aligned_by_t = {}
    for m in modalities:
        df_mu = uav_mod_frames.get(m, pd.DataFrame())
        if len(df_mu) == 0 or "timestamp" not in df_mu.columns:
            aligned_by_t[m] = {}
            continue

        ts_obs = pd.to_numeric(df_mu["timestamp"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ts_obs)
        df_mu = df_mu.loc[valid].copy().reset_index(drop=True)
        ts_obs = ts_obs[valid]
        if len(df_mu) == 0:
            aligned_by_t[m] = {}
            continue

        tidx, tdiff = _nearest_truth_index(ts_obs, ts_truth)
        keep = tdiff <= align_tolerance_s
        if keep.sum() == 0:
            aligned_by_t[m] = {}
            continue

        df_mu = df_mu.loc[keep].copy().reset_index(drop=True)
        df_mu["t_idx"] = tidx[keep].astype(np.int64)
        aligned_by_t[m] = {int(k): g for k, g in df_mu.groupby("t_idx")}

    t_total = len(df_t)
    t_denom = max(window_size - 1, 1)
    uav_samples = []
    for s in range(0, t_total - window_size + 1, stride):
        e = s + window_size
        y = y_full[s:e]
        feats = []
        t_ids = []
        m_ids = []
        obs_json = []

        for ti in range(window_size):
            t_global = s + ti
            item = {}
            for m in modalities:
                rows = aligned_by_t.get(m, {}).get(int(t_global))
                if rows is None or len(rows) == 0:
                    continue
                for _, row in rows.iterrows():
                    feat = _build_node_feature(
                        row=row,
                        modality=m,
                        ti_local=ti,
                        t_denom=t_denom,
                        lat0=lat0,
                        lon0=lon0,
                        alt0=alt0,
                        n_modalities=len(modalities),
                    )
                    feats.append(feat)
                    t_ids.append(ti)
                    m_ids.append(MODALITY_TO_ID[m])
                item[m] = int(len(rows))
            obs_json.append(item)

        if len(feats) == 0:
            continue

        uav_samples.append(
            {
                "node_feat": np.stack(feats).astype(np.float32),
                "node_t": np.array(t_ids, dtype=np.int64),
                "node_m": np.array(m_ids, dtype=np.int64),
                "y": y.astype(np.float32),
                "obs_json": obs_json,
            }
        )
    return uav_samples


class MultiSourceGraphDataset(Dataset):
    """
    Variable-modality sparse graph dataset built from dataset-processed.
    """

    def __init__(
        self,
        data_root=DATA_ROOT,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        modalities: Optional[List[str]] = None,
        truth_dt_s: float = 1.0,
        align_tolerance_s: float = 0.55,
        norm_stats_path: str = NORM_STATS_PATH,
        rebuild_norm_stats: bool = False,
        max_batches: Optional[int] = None,
        verbose: bool = True,
        log_every_uav: int = 25,
        build_workers: int = 4,
        build_use_multiprocessing: bool = True,
        use_sample_cache: bool = True,
        rebuild_sample_cache: bool = False,
        sample_cache_dir: str = ".cache/graph_samples",
    ):
        super().__init__()
        self.data_root = data_root
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.modalities = list(modalities) if modalities is not None else list(MODALITIES)
        self.truth_dt_s = float(truth_dt_s)
        self.align_tolerance_s = float(align_tolerance_s)
        self.norm_stats_path = norm_stats_path
        self.rebuild_norm_stats = bool(rebuild_norm_stats)
        self.max_batches = max_batches
        self.verbose = bool(verbose)
        self.log_every_uav = int(log_every_uav)
        self.build_workers = int(build_workers)
        self.build_use_multiprocessing = bool(build_use_multiprocessing)
        self.use_sample_cache = bool(use_sample_cache)
        self.rebuild_sample_cache = bool(rebuild_sample_cache)
        self.sample_cache_dir = str(sample_cache_dir)
        self.cache_version = 1

        t0 = time.time()
        if self.verbose:
            print(f"[Dataset] building samples from: {self.data_root}")
        samples = self._load_sample_cache()
        if samples is None:
            samples = self._build()
            self._save_sample_cache(samples)
        if len(samples) == 0:
            raise RuntimeError(f"No samples found in {data_root}")

        if (not self.rebuild_norm_stats) and os.path.exists(self.norm_stats_path):
            stats = _safe_torch_load(self.norm_stats_path)
            x_mean = stats["x_mean"].cpu().numpy().astype(np.float32)
            x_std = stats["x_std"].cpu().numpy().astype(np.float32)
            y_mean = stats["y_mean"].cpu().numpy().astype(np.float32)
            y_std = stats["y_std"].cpu().numpy().astype(np.float32)
        else:
            x_all = np.concatenate([s["node_feat"] for s in samples if len(s["node_feat"]) > 0], axis=0)
            x_mean = x_all.mean(0)
            x_std = x_all.std(0) + 1e-6

            # Do not standardize t_norm / flags / acoustic side channels / one-hot
            x_mean[IDX_TNORM:] = 0.0
            x_std[IDX_TNORM:] = 1.0

            y_all = np.concatenate([s["y"] for s in samples], axis=0)
            y_mean = y_all.mean(0)
            y_std = y_all.std(0) + 1e-6

            torch.save(
                {
                    "x_mean": torch.tensor(x_mean),
                    "x_std": torch.tensor(x_std),
                    "y_mean": torch.tensor(y_mean),
                    "y_std": torch.tensor(y_std),
                },
                self.norm_stats_path,
            )

        for s in samples:
            if len(s["node_feat"]) > 0:
                s["node_feat"] = (s["node_feat"] - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
            s["y"] = (s["y"] - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)

        self.samples = samples
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.tensor(x_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.tensor(y_std, dtype=torch.float32)
        self.node_feat_dim = NODE_FEAT_DIM

        if self.verbose:
            print(
                f"[Dataset] Loaded {len(self.samples)} samples from {self.data_root} "
                f"(build_time={time.time() - t0:.1f}s)"
            )


    def _data_signature(self):
        bdirs = self._list_batch_dirs()
        parts = []
        files = ["truth.csv"] + [f"{m}.csv" if m != "5g_a" else "5g_a.csv" for m in self.modalities]
        for bdir in bdirs:
            for fname in files:
                p = os.path.join(bdir, fname)
                if not os.path.exists(p):
                    continue
                st = os.stat(p)
                parts.append(f"{p}:{int(st.st_size)}:{int(st.st_mtime)}")
        return md5("|".join(sorted(parts)).encode("utf-8")).hexdigest()

    def _sample_cache_meta(self):
        return {
            "cache_version": int(self.cache_version),
            "data_root": os.path.abspath(self.data_root),
            "window_size": int(self.window_size),
            "stride": int(self.stride),
            "modalities": list(self.modalities),
            "align_tolerance_s": float(self.align_tolerance_s),
            "data_signature": self._data_signature(),
        }

    def _sample_cache_path(self):
        unit_name = os.path.basename(os.path.normpath(self.data_root)) or "data_root"
        key_raw = "|".join([
            os.path.abspath(self.data_root),
            str(self.window_size),
            str(self.stride),
            ",".join(self.modalities),
            f"{self.align_tolerance_s:.6f}",
            self._data_signature(),
        ]).encode("utf-8")
        key = md5(key_raw).hexdigest()[:16]
        return os.path.join(self.sample_cache_dir, f"graph_samples_{unit_name}_{key}.pth")

    def _load_sample_cache(self):
        if (not self.use_sample_cache) or self.rebuild_sample_cache:
            return None
        cache_path = self._sample_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            payload = _safe_torch_load(cache_path)
            if payload.get("meta", {}) != self._sample_cache_meta():
                return None
            samples = payload.get("samples")
            if samples is None:
                return None
            if self.verbose:
                print(f"[Dataset] sample cache hit: {cache_path}")
            return samples
        except Exception as ex:
            if self.verbose:
                print(f"[Dataset] sample cache load failed, fallback rebuild: {ex}")
            return None

    def _save_sample_cache(self, samples):
        if not self.use_sample_cache:
            return
        cache_path = self._sample_cache_path()
        try:
            os.makedirs(self.sample_cache_dir, exist_ok=True)
            torch.save({"meta": self._sample_cache_meta(), "samples": samples}, cache_path)
            if self.verbose:
                print(f"[Dataset] sample cache saved: {cache_path}")
        except Exception as ex:
            if self.verbose:
                print(f"[Dataset] sample cache save failed (ignored): {ex}")

    def _list_batch_dirs(self):
        bdirs = sorted(glob.glob(os.path.join(self.data_root, "batch*")))
        if len(bdirs) == 0:
            bdirs = [self.data_root]
        if self.max_batches is not None:
            bdirs = bdirs[: int(self.max_batches)]
        return bdirs

    def _read_mod_csv(self, bdir, fname):
        path = os.path.join(bdir, fname)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "id" in df.columns and "uav_id" not in df.columns:
            df = df.rename(columns={"id": "uav_id"})
        if "source_conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"source_conf": "confidence"})
        return df

    def _build_uav_samples(self, df_t: pd.DataFrame, uav_mod_frames: Dict[str, pd.DataFrame]):
        payload = (
            df_t,
            uav_mod_frames,
            self.modalities,
            self.window_size,
            self.stride,
            self.align_tolerance_s,
        )
        return _build_uav_samples_worker(payload)

    def _build(self):
        samples = []
        bdirs = self._list_batch_dirs()
        if self.verbose:
            print(f"[Dataset] units found: {len(bdirs)} | build_workers={self.build_workers}")

        for bi, bdir in enumerate(bdirs, start=1):
            truth_path = os.path.join(bdir, "truth.csv")
            if not os.path.exists(truth_path):
                if self.verbose:
                    print(f"[Dataset] skip unit {bi}: truth.csv missing in {bdir}")
                continue

            truth = pd.read_csv(truth_path)
            if "id" in truth.columns and "uav_id" not in truth.columns:
                truth = truth.rename(columns={"id": "uav_id"})
            if "uav_id" not in truth.columns:
                if self.verbose:
                    print(f"[Dataset] skip unit {bi}: no uav_id in truth")
                continue

            dfs = {m: self._read_mod_csv(bdir, f"{m}.csv" if m != "5g_a" else "5g_a.csv") for m in self.modalities}
            if self.verbose:
                mod_rows = ", ".join([f"{m}:{len(dfs[m])}" for m in self.modalities])
                print(
                    f"[Dataset] unit {bi}/{len(bdirs)}: {os.path.basename(bdir)} | "
                    f"truth_rows={len(truth)} | uavs={truth['uav_id'].nunique()} | {mod_rows}"
                )

            unit_samples_before = len(samples)
            uavs = sorted(truth["uav_id"].unique())
            truth_by_uav = {u: g.sort_values("timestamp").reset_index(drop=True) for u, g in truth.groupby("uav_id", sort=False)}
            mod_by_uav = {}
            for m in self.modalities:
                dfm = dfs.get(m, pd.DataFrame())
                if len(dfm) == 0 or "uav_id" not in dfm.columns:
                    mod_by_uav[m] = {}
                    continue
                mod_by_uav[m] = {u: g.reset_index(drop=True) for u, g in dfm.groupby("uav_id", sort=False)}

            worker_count = max(1, self.build_workers)
            if worker_count == 1 or (not self.build_use_multiprocessing):
                for ui, uav in enumerate(uavs, start=1):
                    df_t = truth_by_uav.get(uav)
                    if df_t is None:
                        continue
                    uav_frames = {m: mod_by_uav.get(m, {}).get(uav, pd.DataFrame()) for m in self.modalities}
                    uav_samples = self._build_uav_samples(df_t, uav_frames)
                    samples.extend(uav_samples)
                    if self.verbose and (ui % max(self.log_every_uav, 1) == 0 or ui == len(uavs)):
                        print(
                            f"[Dataset] unit {bi}/{len(bdirs)} | uav {ui}/{len(uavs)} ({uav}) "
                            f"| windows_added={len(uav_samples)} | total_samples={len(samples)}"
                        )
            else:
                if self.verbose:
                    print(f"[Dataset] unit {bi}: parallel build with {worker_count} processes")
                mp_ctx = get_context("spawn")
                with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as ex:
                    fut_map = {}
                    for uav in uavs:
                        df_t = truth_by_uav.get(uav)
                        if df_t is None:
                            continue
                        uav_frames = {m: mod_by_uav.get(m, {}).get(uav, pd.DataFrame()) for m in self.modalities}
                        payload = (
                            df_t,
                            uav_frames,
                            self.modalities,
                            self.window_size,
                            self.stride,
                            self.align_tolerance_s,
                        )
                        fut = ex.submit(_build_uav_samples_worker, payload)
                        fut_map[fut] = uav

                    done = 0
                    for fut in as_completed(fut_map):
                        uav = fut_map[fut]
                        uav_samples = fut.result()
                        samples.extend(uav_samples)
                        done += 1
                        if self.verbose and (done % max(self.log_every_uav, 1) == 0 or done == len(fut_map)):
                            print(
                                f"[Dataset] unit {bi}/{len(bdirs)} | done_uav {done}/{len(fut_map)} ({uav}) "
                                f"| windows_added={len(uav_samples)} | total_samples={len(samples)}"
                            )

            if self.verbose:
                print(
                    f"[Dataset] unit {bi}/{len(bdirs)} done | "
                    f"new_samples={len(samples) - unit_samples_before} | total_samples={len(samples)}"
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "node_feat": torch.tensor(s["node_feat"], dtype=torch.float32),
            "node_t": torch.tensor(s["node_t"], dtype=torch.long),
            "node_m": torch.tensor(s["node_m"], dtype=torch.long),
            "y": torch.tensor(s["y"], dtype=torch.float32),
            "obs_json": s["obs_json"],
        }


def sparse_collate_fn(batch):
    """
    Pad variable node counts to dense batch tensor.
    """
    bsz = len(batch)
    feat_dim = batch[0]["node_feat"].shape[-1]
    t_len = batch[0]["y"].shape[0]
    lmax = max(x["node_feat"].shape[0] for x in batch)

    node_feat = torch.zeros((bsz, lmax, feat_dim), dtype=torch.float32)
    node_t = torch.full((bsz, lmax), -1, dtype=torch.long)
    node_m = torch.full((bsz, lmax), -1, dtype=torch.long)
    node_mask = torch.zeros((bsz, lmax), dtype=torch.float32)
    y = torch.zeros((bsz, t_len, 3), dtype=torch.float32)

    obs_json = []
    for i, item in enumerate(batch):
        n = item["node_feat"].shape[0]
        node_feat[i, :n] = item["node_feat"]
        node_t[i, :n] = item["node_t"]
        node_m[i, :n] = item["node_m"]
        node_mask[i, :n] = 1.0
        y[i] = item["y"]
        obs_json.append(item["obs_json"])

    return {
        "node_feat": node_feat,
        "node_t": node_t,
        "node_m": node_m,
        "node_mask": node_mask,
        "y": y,
        "obs_json": obs_json,
    }
