import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

# Keep aligned with dataset.py
NODE_FEAT_DIM = 19
N_MODALITIES = 5
IDX_CONF = 7
IDX_TNORM = 8
IDX_POS_VALID = 9
IDX_OBS_VALID = 10


class GraphTransformerLayer(nn.Module):
    """
    Sparse-attention transformer layer used as node-level context encoder.
    """

    def __init__(self, d_model=128, num_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h, attn_bias=None, node_mask=None, sparse_mask=None):
        bsz, lsz, dsz = h.shape
        heads = self.num_heads

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, lsz, heads, self.d_k).transpose(1, 2)
        k = k.view(bsz, lsz, heads, self.d_k).transpose(1, 2)
        v = v.view(bsz, lsz, heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_bias is not None:
            scores = scores + attn_bias

        if sparse_mask is not None:
            scores = scores.masked_fill(~sparse_mask[:, None, :, :], -1e9)

        if node_mask is not None:
            key_mask = node_mask[:, None, None, :] > 0
            scores = scores.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)

        if node_mask is not None:
            query_mask = (node_mask[:, None, :, None] > 0).to(attn.dtype)
            attn = attn * query_mask

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, lsz, dsz)
        out = self.out_proj(out)

        h = self.norm1(h + self.dropout(out))
        ff = self.ffn(h)
        h = self.norm2(h + self.dropout(ff))

        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)
        return h


class GraphFusionModel(nn.Module):
    """
    Adaptive multimodal fusion model for time-varying modality availability.

    Inputs:
      node_feat: [B, L, F]
      node_t:    [B, L]  (0..T-1, pad=-1)
      node_m:    [B, L]  (0..M-1, pad=-1)
      node_mask: [B, L]  (1=valid, 0=pad)
    Output:
      pred: [B, T, 3]
    """

    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        num_modalities: int = N_MODALITIES,
        window_size: int = 20,
        knn_k: int = 8,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_modalities = int(num_modalities)
        self.window_size = int(window_size)
        self.knn_k = int(knn_k)

        # Keep legacy backbone names so old checkpoints can be partially reused.
        self.node_encoder = nn.Linear(self.in_dim, self.d_model)

        # edge_feat = [dist, dt, same_mod, same_time, dconf, pair_pos_valid]
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_heads),
        )
        # Extra confidence-aware edge bias (new head, backward-compatible add-on).
        self.edge_conf_mlp = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.num_heads),
        )

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.node_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Slot = (time, modality). Build adaptive modality fusion at each time step.
        self.mod_emb = nn.Embedding(self.num_modalities, self.d_model)
        self.slot_aux_proj = nn.Sequential(
            nn.Linear(8, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.slot_gate = nn.Sequential(
            nn.Linear(self.d_model + 8, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.gate_conf_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_avail_bias = nn.Parameter(torch.tensor(0.25))

        # Time sequence modeling after per-time multimodal fusion.
        self.time_in_proj = nn.Sequential(
            nn.Linear(self.d_model + 6, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)

        # Keep legacy head name for checkpoint compatibility.
        self.out_head = nn.Linear(self.d_model, 3)

    def _build_sparse_mask(self, dist, dt, node_mask):
        """
        KNN sparse graph + later in forward we add guaranteed local-time edges.
        dist, dt: [B,L,L]
        node_mask: [B,L]
        """
        bsz, lsz, _ = dist.shape
        valid_pair = (node_mask[:, :, None] > 0) & (node_mask[:, None, :] > 0)

        metric = dist + dt
        metric = metric.masked_fill(~valid_pair, 1e9)

        eye = torch.eye(lsz, device=dist.device, dtype=torch.bool).unsqueeze(0)
        valid_self = eye & (node_mask[:, :, None] > 0)
        metric = metric.masked_fill(valid_self, 0.0)

        k_eff = max(1, min(self.knn_k, lsz))
        knn_idx = torch.topk(metric, k=k_eff, dim=-1, largest=False).indices

        sparse_mask = torch.zeros((bsz, lsz, lsz), dtype=torch.bool, device=dist.device)
        sparse_mask.scatter_(2, knn_idx, True)

        sparse_mask = (sparse_mask | sparse_mask.transpose(1, 2)) & valid_pair
        sparse_mask = sparse_mask | valid_self
        return sparse_mask

    @staticmethod
    def _safe_conf01(conf_raw: torch.Tensor) -> torch.Tensor:
        in_range = (conf_raw >= 0.0) & (conf_raw <= 1.0)
        conf_sig = torch.sigmoid(conf_raw)
        conf01 = torch.where(in_range, conf_raw, conf_sig)
        return conf01.clamp(0.0, 1.0)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        mask_bool = mask > 0
        masked_logits = logits.masked_fill(~mask_bool, -1e9)
        probs = torch.softmax(masked_logits, dim=dim)
        probs = probs * mask_bool.to(probs.dtype)
        denom = probs.sum(dim=dim, keepdim=True)
        return probs / (denom + 1e-6)

    def _slot_age(self, slot_avail: torch.Tensor) -> torch.Tensor:
        # slot_avail: [B,T,M] bool
        bsz, t_len, msz = slot_avail.shape
        age = torch.zeros((bsz, t_len, msz), dtype=torch.float32, device=slot_avail.device)
        if t_len <= 0:
            return age
        for m in range(msz):
            last = torch.full((bsz,), -1, dtype=torch.long, device=slot_avail.device)
            for t in range(t_len):
                cur = slot_avail[:, t, m]
                cur_age = torch.where(
                    cur,
                    torch.zeros_like(last, dtype=torch.float32),
                    torch.where(last >= 0, (t - last).float(), torch.full_like(last, float(t + 1), dtype=torch.float32)),
                )
                age[:, t, m] = cur_age
                last = torch.where(cur, torch.full_like(last, int(t)), last)
        return age / float(max(t_len - 1, 1))

    def _pairwise_spread(self, slot_pos: torch.Tensor, slot_avail: torch.Tensor) -> torch.Tensor:
        # slot_pos: [B,T,M,3], slot_avail: [B,T,M]
        bsz, t_len, msz, _ = slot_pos.shape
        if msz <= 1:
            return torch.zeros((bsz, t_len), dtype=slot_pos.dtype, device=slot_pos.device)
        pi = slot_pos[:, :, :, None, :]
        pj = slot_pos[:, :, None, :, :]
        d = torch.norm(pi - pj, dim=-1)
        valid = (slot_avail[:, :, :, None] > 0) & (slot_avail[:, :, None, :] > 0)
        eye = torch.eye(msz, dtype=torch.bool, device=slot_pos.device).view(1, 1, msz, msz)
        valid = valid & (~eye)
        num = (d * valid.to(d.dtype)).sum(dim=(-1, -2))
        den = valid.to(d.dtype).sum(dim=(-1, -2))
        return num / (den + 1e-6)

    def _aggregate_slots(self, h, node_feat, node_t, node_m, node_mask, t_len):
        bsz, lsz, dsz = h.shape
        msz = self.num_modalities
        device = h.device
        dtype = h.dtype

        t_idx = node_t.clamp(min=0, max=max(int(t_len) - 1, 0))
        m_idx = node_m.clamp(min=0, max=max(msz - 1, 0))
        flat_idx = t_idx * msz + m_idx
        slot_n = int(t_len) * int(msz)

        valid = (node_mask > 0.5)
        conf_raw = node_feat[..., IDX_CONF] if node_feat.shape[-1] > IDX_CONF else torch.zeros_like(node_mask)
        conf01 = self._safe_conf01(conf_raw)
        pos_valid = node_feat[..., IDX_POS_VALID] if node_feat.shape[-1] > IDX_POS_VALID else torch.ones_like(node_mask)
        obs_valid = node_feat[..., IDX_OBS_VALID] if node_feat.shape[-1] > IDX_OBS_VALID else torch.ones_like(node_mask)
        pos_valid = (pos_valid > 0.5).to(dtype)
        obs_valid = (obs_valid > 0.5).to(dtype)
        valid_f = valid.to(dtype)

        # Weighted node contribution prefers high-confidence valid observations but never fully drops nodes.
        node_w = valid_f * (0.15 + 0.85 * conf01) * (0.5 + 0.5 * obs_valid)
        slot_idx_expand = flat_idx.unsqueeze(-1)

        slot_h_sum = torch.zeros((bsz, slot_n, dsz), dtype=dtype, device=device)
        slot_h_w = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)
        slot_conf_sum = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)
        slot_pos_sum = torch.zeros((bsz, slot_n, 3), dtype=dtype, device=device)
        slot_pos_w = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)
        slot_count = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)
        slot_obs_valid_sum = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)
        slot_pos_valid_sum = torch.zeros((bsz, slot_n, 1), dtype=dtype, device=device)

        h_w = h * node_w.unsqueeze(-1)
        slot_h_sum.scatter_add_(1, slot_idx_expand.expand(-1, -1, dsz), h_w)
        slot_h_w.scatter_add_(1, slot_idx_expand, node_w.unsqueeze(-1))
        slot_conf_sum.scatter_add_(1, slot_idx_expand, (conf01 * valid_f).unsqueeze(-1))
        if node_feat.shape[-1] >= 3:
            slot_pos_sum.scatter_add_(1, slot_idx_expand.expand(-1, -1, 3), node_feat[..., 0:3] * (pos_valid * valid_f).unsqueeze(-1))
            slot_pos_w.scatter_add_(1, slot_idx_expand, (pos_valid * valid_f).unsqueeze(-1))
        slot_count.scatter_add_(1, slot_idx_expand, valid_f.unsqueeze(-1))
        slot_obs_valid_sum.scatter_add_(1, slot_idx_expand, (obs_valid * valid_f).unsqueeze(-1))
        slot_pos_valid_sum.scatter_add_(1, slot_idx_expand, (pos_valid * valid_f).unsqueeze(-1))

        slot_h = slot_h_sum / (slot_h_w + 1e-6)
        slot_conf_mean = slot_conf_sum / (slot_count + 1e-6)
        slot_pos = slot_pos_sum / (slot_pos_w + 1e-6)
        slot_obs_valid_mean = slot_obs_valid_sum / (slot_count + 1e-6)
        slot_pos_valid_mean = slot_pos_valid_sum / (slot_count + 1e-6)
        slot_avail = (slot_count.squeeze(-1) > 0.0)

        slot_h = slot_h.view(bsz, int(t_len), msz, dsz)
        slot_conf_mean = slot_conf_mean.view(bsz, int(t_len), msz)
        slot_pos = slot_pos.view(bsz, int(t_len), msz, 3)
        slot_obs_valid_mean = slot_obs_valid_mean.view(bsz, int(t_len), msz)
        slot_pos_valid_mean = slot_pos_valid_mean.view(bsz, int(t_len), msz)
        slot_count = slot_count.view(bsz, int(t_len), msz)
        slot_avail_f = slot_avail.view(bsz, int(t_len), msz).to(dtype)

        age = self._slot_age(slot_avail.view(bsz, int(t_len), msz))
        spread_t = self._pairwise_spread(slot_pos, slot_avail.view(bsz, int(t_len), msz))
        cov_t = slot_avail_f.mean(dim=-1)
        full_t = (slot_avail_f.sum(dim=-1) >= float(msz)).to(dtype)
        count_norm = torch.tanh(torch.log1p(slot_count))

        spread_broadcast = spread_t.unsqueeze(-1).expand(-1, -1, msz)
        cov_broadcast = cov_t.unsqueeze(-1).expand(-1, -1, msz)
        full_broadcast = full_t.unsqueeze(-1).expand(-1, -1, msz)
        slot_aux = torch.stack(
            [
                slot_conf_mean.clamp(0.0, 1.0),
                slot_obs_valid_mean.clamp(0.0, 1.0),
                slot_pos_valid_mean.clamp(0.0, 1.0),
                count_norm.squeeze(-1).clamp(0.0, 1.0),
                age.clamp(0.0, 1.0),
                cov_broadcast.clamp(0.0, 1.0),
                full_broadcast.clamp(0.0, 1.0),
                torch.tanh(spread_broadcast),
            ],
            dim=-1,
        )

        # Time-level diagnostics for the temporal model.
        avail_count_t = slot_avail_f.sum(dim=-1)
        conf_mean_t = (slot_conf_mean * slot_avail_f).sum(dim=-1) / (avail_count_t + 1e-6)
        spread_t = torch.tanh(spread_t)
        cov_delta = torch.zeros_like(cov_t)
        conf_delta = torch.zeros_like(conf_mean_t)
        if int(t_len) > 1:
            cov_delta[:, 1:] = (cov_t[:, 1:] - cov_t[:, :-1]).abs()
            conf_delta[:, 1:] = (conf_mean_t[:, 1:] - conf_mean_t[:, :-1]).abs()
        time_aux = torch.stack(
            [
                cov_t.clamp(0.0, 1.0),
                full_t.clamp(0.0, 1.0),
                conf_mean_t.clamp(0.0, 1.0),
                spread_t,
                cov_delta.clamp(0.0, 1.0),
                conf_delta.clamp(0.0, 1.0),
            ],
            dim=-1,
        )

        return slot_h, slot_aux, slot_avail_f, slot_conf_mean, time_aux

    def forward(self, node_feat, node_t, node_m, node_mask, window_size=None):
        bsz, lsz, _ = node_feat.shape
        t_len = int(window_size or self.window_size)

        h = self.node_encoder(node_feat)

        t_clamped = node_t.clamp(min=0)
        m_clamped = node_m.clamp(min=0)

        pos = node_feat[..., 0:3]
        conf_raw = node_feat[..., IDX_CONF : IDX_CONF + 1] if node_feat.shape[-1] > IDX_CONF else torch.zeros((bsz, lsz, 1), device=node_feat.device, dtype=node_feat.dtype)
        conf = self._safe_conf01(conf_raw)
        pos_valid = node_feat[..., IDX_POS_VALID : IDX_POS_VALID + 1] if node_feat.shape[-1] > IDX_POS_VALID else torch.ones((bsz, lsz, 1), device=node_feat.device, dtype=node_feat.dtype)

        t_i = t_clamped[:, :, None]
        t_j = t_clamped[:, None, :]
        dt = (t_i - t_j).abs().float() / max(t_len - 1, 1)

        m_i = m_clamped[:, :, None]
        m_j = m_clamped[:, None, :]
        same_mod = (m_i == m_j).float()
        same_time = (t_i == t_j).float()

        pi = pos[:, :, None, :]
        pj = pos[:, None, :, :]
        dist = torch.norm(pi - pj, dim=-1)
        pv_i = pos_valid[:, :, None, :]
        pv_j = pos_valid[:, None, :, :]
        pair_pos_valid = (pv_i > 0.5) & (pv_j > 0.5)
        dist = torch.where(pair_pos_valid.squeeze(-1), dist, torch.full_like(dist, 50.0))

        ci = conf[:, :, None, :]
        cj = conf[:, None, :, :]
        dconf = (ci - cj).abs().squeeze(-1)

        edge_feat = torch.stack(
            [dist, dt, same_mod, same_time, dconf, pair_pos_valid.squeeze(-1).float()],
            dim=-1,
        )

        valid_pair = (node_mask[:, :, None] > 0) & (node_mask[:, None, :] > 0)
        edge_feat = edge_feat * valid_pair.unsqueeze(-1).to(edge_feat.dtype)

        attn_bias = self.edge_mlp(edge_feat)
        ci_full = ci.expand(-1, -1, node_feat.shape[1], -1).squeeze(-1)
        cj_full = cj.expand(-1, node_feat.shape[1], -1, -1).squeeze(-1)
        cprod = (ci * cj).expand(-1, -1, node_feat.shape[1], -1).squeeze(-1)
        edge_conf_feat = torch.stack([ci_full, cj_full, cprod, dconf], dim=-1)
        edge_conf_bias = self.edge_conf_mlp(edge_conf_feat)
        attn_bias = attn_bias + edge_conf_bias
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()

        sparse_mask = self._build_sparse_mask(dist, dt, node_mask)
        # Guarantee local temporal edges so dropout/recovery transitions are propagated.
        temporal_step = 1.0 / max(t_len - 1, 1)
        local_time_edges = (dt <= (temporal_step + 1e-6)) & valid_pair
        sparse_mask = sparse_mask | local_time_edges

        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias, node_mask=node_mask, sparse_mask=sparse_mask)

        h = h + self.node_refine(h)
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)

        slot_h, slot_aux, slot_avail_f, slot_conf_mean, time_aux = self._aggregate_slots(
            h=h,
            node_feat=node_feat,
            node_t=node_t,
            node_m=node_m,
            node_mask=node_mask,
            t_len=t_len,
        )

        mod_ids = torch.arange(self.num_modalities, device=h.device, dtype=torch.long)
        mod_emb = self.mod_emb(mod_ids).view(1, 1, self.num_modalities, self.d_model)
        slot_emb = slot_h + mod_emb + self.slot_aux_proj(slot_aux)

        gate_logits = self.slot_gate(torch.cat([slot_emb, slot_aux], dim=-1)).squeeze(-1)
        gate_logits = gate_logits + self.gate_conf_scale * slot_conf_mean + self.gate_avail_bias * slot_avail_f
        gate_w = self._masked_softmax(gate_logits, slot_avail_f > 0, dim=-1)
        time_fused = (slot_emb * gate_w.unsqueeze(-1)).sum(dim=2)

        time_h = self.time_in_proj(torch.cat([time_fused, time_aux], dim=-1))
        gru_out, _ = self.temporal_gru(time_h)
        time_h = self.temporal_norm(time_fused + self.temporal_proj(gru_out))

        pred = self.out_head(time_h)
        return pred
