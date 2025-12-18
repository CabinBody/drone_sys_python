# model.py
import math
import torch
import torch.nn as nn

# ==============================
# ⚙ 可配置参数（与 dataset 对应）
# ==============================
NODE_FEAT_DIM = 12   # 输入特征维度（dataset 里定义）
N_MODALITIES  = 3    # radar / 5g_a / tdoa


class GraphTransformerLayer(nn.Module):
    """
    标准 Transformer Encoder + edge bias
    h: [B, L, D]
    attn_bias: [B, H, L, L]
    mask: [B, L] (1=valid, 0=padding)
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

    def forward(self, h, attn_bias=None, mask=None):
        # h: [B, L, D]
        B, L, D = h.shape
        H = self.num_heads

        qkv = self.qkv(h)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        k = k.view(B, L, H, self.d_k).transpose(1, 2)
        v = v.view(B, L, H, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]

        if attn_bias is not None:
            # attn_bias: [B, H, L, L]
            scores = scores + attn_bias

        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            attn_mask = (mask[:, None, None, :] > 0).to(scores.dtype)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, L, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)

        h = self.norm1(h + self.dropout(out))

        ff = self.ffn(h)
        h = self.norm2(h + self.dropout(ff))
        return h


class GraphFusionModel(nn.Module):
    """
    在一个时间窗口内，将 (T × 3) 个“源观测”看作图上的节点：
    - 节点特征: 位置(lat/lon/alt)、速度(vx/vy/vz)、speed、conf、时间t_norm、模态one-hot
    - 边特征: 归一化空间距离、时间差、是否同模态、是否同时间、置信度差
    使用 Graph Transformer 进行全局建模，然后按时间聚合三个模态，输出融合后的
    truth 估计（lat/lon/alt），形状为 [B, T, 3]。
    """
    def __init__(self,
                 in_dim: int = NODE_FEAT_DIM,
                 d_model: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dim_ff: int = 256,
                 dropout: float = 0.1,
                 num_modalities: int = N_MODALITIES,
                 window_size: int = 20):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_modalities = num_modalities
        self.window_size = window_size

        # 节点特征编码
        self.node_encoder = nn.Linear(in_dim, d_model)

        # 边特征编码 -> attention bias (每个 head 一个 bias)
        # edge_feat = [dist, dt, same_mod, same_time, dconf]
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_heads),
        )

        # 多层 Graph Transformer
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 输出头：将每个时间步聚合后的表示映射到 (lat, lon, alt)
        self.out_head = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        """
        x:    [B, T, M, F]  已归一化的节点特征
        mask: [B, T, M]     每个节点是否有观测 (1/0)
        返回:
        pred: [B, T, 3]     预测的 (lat, lon, alt)，同样是归一化空间
        """
        B, T, M, F = x.shape
        assert M == self.num_modalities, "num_modalities mismatch."

        L = T * M  # 窗口内总节点数
        device = x.device

        # 节点编码
        h = self.node_encoder(x)         # [B, T, M, D]
        h = h.view(B, L, self.d_model)   # [B, L, D]

        # node mask
        node_mask = mask.view(B, L)      # [B, L]

        # 从 x 中取出位置和置信度 (已经是归一化后的空间，但相对关系仍然有意义)
        pos = x[..., 0:3].view(B, L, 3)   # [B, L, 3]   (lat, lon, alt)
        conf = x[..., 7].view(B, L, 1)    # [B, L, 1]   (source_conf)

        # 时间 & 模态索引 (根据展平顺序构造)
        t_idx = torch.arange(T, device=device).repeat_interleave(M)  # [L]
        m_idx = torch.arange(M, device=device).repeat(T)             # [L]

        # pair-wise 时间特征
        t_i = t_idx[None, :, None]    # [1, L, 1]
        t_j = t_idx[None, None, :]    # [1, 1, L]
        dt = (t_i - t_j).abs().float() / max(T - 1, 1)  # [1, L, L] 归一化时间差

        # pair-wise 模态关系
        m_i = m_idx[None, :, None]    # [1, L, 1]
        m_j = m_idx[None, None, :]    # [1, 1, L]
        same_mod  = (m_i == m_j).float()  # [1, L, L]
        same_time = (t_i == t_j).float()  # [1, L, L]

        # 扩展到 batch
        dt        = dt.expand(B, -1, -1)         # [B, L, L]
        same_mod  = same_mod.expand(B, -1, -1)   # [B, L, L]
        same_time = same_time.expand(B, -1, -1)  # [B, L, L]

        # pair-wise 空间距离 (在归一化空间中计算，保持单调性即可)
        pi = pos[:, :, None, :]  # [B, L, 1, 3]
        pj = pos[:, None, :, :]  # [B, 1, L, 3]
        dist = torch.norm(pi - pj, dim=-1)  # [B, L, L]

        # 置信度差
        ci = conf[:, :, None, :]  # [B, L, 1, 1]
        cj = conf[:, None, :, :]  # [B, 1, L, 1]
        dconf = (ci - cj).abs().squeeze(-1)  # [B, L, L]

        # 拼接边特征: [dist, dt, same_mod, same_time, dconf]
        edge_feat = torch.stack([dist, dt, same_mod, same_time, dconf], dim=-1)  # [B, L, L, 5]

        # 编码为 attention bias
        attn_bias = self.edge_mlp(edge_feat)  # [B, L, L, H]
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]

        # 多层 Graph Transformer
        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias, mask=node_mask)

        # 重塑回 [B, T, M, D]
        h = h.view(B, T, M, self.d_model)

        # 按时间聚合三个模态 -> conf / mask 加权平均
        node_mask_exp = node_mask.view(B, T, M, 1)  # [B, T, M, 1]
        weights = node_mask_exp  # 这里只用 mask，当作权重；conf 已经嵌入特征中
        fused = (h * weights).sum(dim=2) / (weights.sum(dim=2) + 1e-6)  # [B, T, D]

        # 输出预测的 (lat, lon, alt) （在归一化空间）
        pred = self.out_head(fused)  # [B, T, 3]
        return pred
