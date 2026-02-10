import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

# 与 dataset.py 保持一致
NODE_FEAT_DIM = 19
N_MODALITIES = 5
IDX_CONF = 7
IDX_POS_VALID = 9


class GraphTransformerLayer(nn.Module):
    """
    带边偏置与稀疏掩码的 Transformer Encoder 层。
    h: [B, L, D]
    attn_bias: [B, H, L, L]
    node_mask: [B, L] (1=valid, 0=pad)
    sparse_mask: [B, L, L] (True=允许注意力边)
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

        # 无效 query 节点不参与输出，避免 padding 传播噪声
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
    可变模态 + 稀疏图融合模型。

    输入:
      node_feat: [B, L, F]
      node_t:    [B, L]  (0..T-1, pad为-1)
      node_m:    [B, L]  (0..M-1, pad为-1)
      node_mask: [B, L]  (1=valid, 0=pad)
    输出:
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
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_modalities = num_modalities
        self.window_size = window_size
        self.knn_k = knn_k

        self.node_encoder = nn.Linear(in_dim, d_model)

        # edge_feat = [dist, dt, same_mod, same_time, dconf, pair_pos_valid]
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_heads),
        )

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_head = nn.Linear(d_model, 3)

    def _build_sparse_mask(self, dist, dt, node_mask):
        """
        基于 KNN 的稀疏邻接。
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

        # 双向化并裁掉 padding
        sparse_mask = (sparse_mask | sparse_mask.transpose(1, 2)) & valid_pair
        sparse_mask = sparse_mask | valid_self
        return sparse_mask

    def forward(self, node_feat, node_t, node_m, node_mask, window_size=None):
        bsz, lsz, _ = node_feat.shape
        t_len = int(window_size or self.window_size)

        h = self.node_encoder(node_feat)

        t_clamped = node_t.clamp(min=0)
        m_clamped = node_m.clamp(min=0)

        pos = node_feat[..., 0:3]
        conf = node_feat[..., IDX_CONF : IDX_CONF + 1]
        pos_valid = node_feat[..., IDX_POS_VALID : IDX_POS_VALID + 1]

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

        edge_feat = torch.stack([dist, dt, same_mod, same_time, dconf, pair_pos_valid.squeeze(-1).float()], dim=-1)

        # padding 边置零，避免无效特征影响 edge_mlp
        valid_pair = (node_mask[:, :, None] > 0) & (node_mask[:, None, :] > 0)
        edge_feat = edge_feat * valid_pair.unsqueeze(-1).to(edge_feat.dtype)

        attn_bias = self.edge_mlp(edge_feat)
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()

        sparse_mask = self._build_sparse_mask(dist, dt, node_mask)

        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias, node_mask=node_mask, sparse_mask=sparse_mask)

        t_clamped = node_t.clamp(min=0, max=t_len - 1)
        time_onehot = Fnn.one_hot(t_clamped, num_classes=t_len).to(h.dtype)
        time_onehot = time_onehot * node_mask.unsqueeze(-1)

        weights = time_onehot.sum(dim=1)
        fused = torch.einsum("bld,blt->btd", h, time_onehot) / (weights.unsqueeze(-1) + 1e-6)

        pred = self.out_head(fused)
        return pred
