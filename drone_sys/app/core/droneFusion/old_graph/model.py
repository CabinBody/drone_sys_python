# ================================================================
# model.py —— Stable Graph Transformer for Multi-Source Trajectory Fusion
# 输入:
#   traj_obs:  (T, 3, 7)  # (radar, 5g, tdoa) × [pos(3)+vel(3)+speed]
#   traj_conf: (T, 3)
# 输出:
#   (T, 3)  # 融合后 xyz
# ================================================================

import math
import torch
import torch.nn as nn

# ==========================
# ★ Model Config
# ==========================
NODE_BASE_DIM   = 8    # [pos(3)+vel(3)+speed(1)+conf(1)]
EDGE_INPUT_DIM  = 9    # [dpos(3)+dist(1)+dvel(3)+dconf(1)+dt(1)]
HIDDEN_DIM      = 128
NUM_LAYERS      = 3
NUM_HEADS       = 4
DROPOUT         = 0.1
OUTPUT_DIM      = 3
MAX_TIME_STEPS  = 256  # 支持最多 256 帧的时间编码
NUM_SOURCES     = 3    # radar / 5g / tdoa


# ================================================================
# Graph Builder —— 用 torch 构图（单个 UAV）
# ================================================================
class GraphBuilder:
    def __init__(self):
        pass

    def build_graph(self, traj_obs: torch.Tensor, traj_conf: torch.Tensor):
        """
        traj_obs : (T, 3, 7)
        traj_conf: (T, 3)
        返回:
          node_feat_raw: (N, 8)    # [pos, vel, speed, conf]
          edge_attr    : (N, N, 9) # [dpos, dist, dvel, dconf, dt]
          time_ids     : (N,)      # 0..T-1
          source_ids   : (N,)      # 0,1,2
          T, S
        """
        device = traj_obs.device
        T, S, D = traj_obs.shape   # S=3 sources, D=7

        # ----- 拆成 pos / vel / speed / conf -----
        pos   = traj_obs[..., 0:3]    # (T,3,3)
        vel   = traj_obs[..., 3:6]    # (T,3,3)
        speed = traj_obs[..., 6:7]    # (T,3,1)
        conf  = traj_conf.unsqueeze(-1)  # (T,3,1)

        # ----- 展平时间+模态 -----
        # N = T * 3
        N = T * S
        pos_f   = pos.reshape(N, 3)
        vel_f   = vel.reshape(N, 3)
        speed_f = speed.reshape(N, 1)
        conf_f  = conf.reshape(N, 1)

        # 基础节点特征: 8维
        node_feat_raw = torch.cat([pos_f, vel_f, speed_f, conf_f], dim=-1)  # (N,8)

        # ----- 时间 id & 源 id -----
        # time_ids: [0,0,0, 1,1,1, ..., T-1,T-1,T-1]
        t_ids = torch.arange(T, device=device).unsqueeze(1).repeat(1, S).reshape(-1)  # (N,)
        # source_ids: [0,1,2, 0,1,2, ...]
        s_ids = torch.arange(S, device=device).unsqueeze(0).repeat(T, 1).reshape(-1)  # (N,)

        # ----- 边特征 (N,N,9) -----
        pos_i = pos_f.unsqueeze(1)          # (N,1,3)
        pos_j = pos_f.unsqueeze(0)          # (1,N,3)
        dpos  = pos_i - pos_j               # (N,N,3)
        dist  = torch.norm(dpos, dim=-1, keepdim=True)  # (N,N,1)

        vel_i = vel_f.unsqueeze(1)
        vel_j = vel_f.unsqueeze(0)
        dvel  = vel_i - vel_j               # (N,N,3)

        conf_i = conf_f.unsqueeze(1)        # (N,1,1)
        conf_j = conf_f.unsqueeze(0)        # (1,N,1)
        dconf  = conf_i - conf_j            # (N,N,1)

        ti = t_ids.unsqueeze(1)             # (N,1)
        tj = t_ids.unsqueeze(0)             # (1,N)
        dt = (ti - tj).abs().float() / max(T, 1)  # 归一化时间差 [0,1]
        dt = dt.unsqueeze(-1)               # (N,N,1)

        edge_attr = torch.cat([dpos, dist, dvel, dconf, dt], dim=-1)  # (N,N,9)

        return node_feat_raw, edge_attr, t_ids, s_ids, T, S


# ================================================================
# Graph Transformer Layer（dense attention + edge bias）
# ================================================================
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dropout=DROPOUT):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # edge_attr -> attention bias for each head
        self.edge_mlp = nn.Linear(EDGE_INPUT_DIM, heads)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, edge_attr):
        """
        x        : (N, dim)
        edge_attr: (N, N, EDGE_INPUT_DIM)
        """
        N, dim = x.shape
        device = x.device

        # ----- Multi-head Attention -----
        Q = self.Wq(x).view(N, self.heads, self.dk)  # (N,H,dk)
        K = self.Wk(x).view(N, self.heads, self.dk)
        V = self.Wv(x).view(N, self.heads, self.dk)

        # 变成 (H,N,dk)
        Qh = Q.permute(1, 0, 2)   # (H,N,dk)
        Kh = K.permute(1, 0, 2)
        Vh = V.permute(1, 0, 2)

        # 基础注意力分数: (H,N,N)
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.dk)

        # 边特征 → 注意力 bias
        # edge_attr: (N,N,9) -> (N,N,H) -> (H,N,N)
        edge_bias = self.edge_mlp(edge_attr)          # (N,N,H)
        edge_bias = edge_bias.permute(2, 0, 1)        # (H,N,N)

        attn_scores = attn_scores + edge_bias

        attn = torch.softmax(attn_scores, dim=-1)     # (H,N,N)
        attn = self.dropout(attn)

        # 聚合消息
        out_h = torch.matmul(attn, Vh)  # (H,N,dk)
        out = out_h.permute(1, 0, 2).reshape(N, dim)  # (N,dim)

        # 残差 + Norm
        x = self.norm1(x + self.dropout(out))

        # FFN
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))

        return x


# ================================================================
# 主模型：GraphFusionModel
# ================================================================
class GraphFusionModel(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        layers=NUM_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads

        # 节点基础特征映射到 hidden
        self.node_proj = nn.Linear(NODE_BASE_DIM, hidden_dim)

        # 时间 & 源 编码
        self.time_emb = nn.Embedding(MAX_TIME_STEPS, hidden_dim)
        self.source_emb = nn.Embedding(NUM_SOURCES, hidden_dim)

        # 多层 Graph Transformer
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout=dropout)
            for _ in range(layers)
        ])

        # 输出: 每个时间融合 3 源之后 -> 预测 xyz
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

        self.graph_builder = GraphBuilder()

    # ------------------------------------------------------------------
    # traj_obs:  (T,3,7)
    # traj_conf: (T,3)
    # ------------------------------------------------------------------
    def forward(self, traj_obs, traj_conf):
        device = traj_obs.device

        # 1. 构图
        node_raw, edge_attr, time_ids, src_ids, T, S = \
            self.graph_builder.build_graph(traj_obs, traj_conf)
        # node_raw: (N,8), edge_attr: (N,N,9), time_ids: (N,), src_ids: (N,)
        N = node_raw.size(0)

        # 2. 节点特征 + 时间/源 编码
        x = self.node_proj(node_raw)                    # (N,H)
        t_emb = self.time_emb(time_ids.clamp(max=MAX_TIME_STEPS-1))   # (N,H)
        s_emb = self.source_emb(src_ids)               # (N,H)

        x = x + t_emb + s_emb

        # 3. 多层 Graph Transformer
        for layer in self.layers:
            x = layer(x, edge_attr)

        # 4. 还原为 (T,3,H)
        x = x.view(T, S, self.hidden_dim)   # (T,3,H)

        # 5. 对 3 个源做简单平均（你后面可以换成 source-attention）
        fused = x.mean(dim=1)              # (T,H)

        # 6. 输出 xyz
        out = self.out_mlp(fused)          # (T,3)

        return out
