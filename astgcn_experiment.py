# astgcn_experiment.py
#
# AST-GCN (Attention based Spatio-Temporal GCN)
# - Multi-task: [TotalTraffic, Speed]
# - Multi-stream: X_h (recent), X_d (daily), X_w (weekly)
#
# [핵심 변경/요구 반영]
# 1) Spatial Attention SAtt를 (B,N,N) dense로 만들지 않고, edge-level만 계산: S_edge (B,E) (row-softmax)
# 2) Adaptive Adjacency도 dense (N,N) 만들지 않고, edge-level만 학습: A_adp_edge (E,) (row-softmax)
# 3) 메시지 패싱 가중치:
#    msg_edge = ((1-beta)A_fixed_edge + beta*A_adp_edge) * S_edge
#    msg_edge를 row-wise로 다시 normalize
# 4) GraphConv는 sparse mm 대신 edge scatter_add로 수행 (batched sparse mm 회피)
#
# [중요 버그 수정]
# - "one of the variables needed for gradient computation has been modified by an inplace operation"
#   원인: row_ptr 루프 + 슬라이스(view)에 대한 in-place 대입(out[:, s:e] = ...)
#   해결: row-softmax / row-renorm 모두 scatter 기반으로 구현해서 in-place 제거
#
# 전제:
#   res/ 아래 파일들 존재:
#     - Xh_samples.npy  : (S, N, T_h, F)
#     - Xd_samples.npy  : (S, N, T_d, F)   (periodic=True)
#     - Xw_samples.npy  : (S, N, T_w, F)   (periodic=True)
#     - Y_samples_ast.npy: (S, N, T_out, 2)
#     - adjacency_corr_norm.npy (권장) 또는 adjacency_norm.npy

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# 0. 공통 설정
# =========================

data_dir = Path("/mnt/c/새 폴더/res")

A_corr_path = data_dir / "adjacency_corr_norm.npy"
A_phys_path = data_dir / "adjacency_norm.npy"
A_path = A_corr_path if A_corr_path.exists() else A_phys_path
print("[AST-GCN] Using adjacency:", A_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Adaptive A 스케줄 -----
USE_ADAPTIVE_ADJ = True
ADJ_EMB_DIM = 16
ADAPTIVE_WARMUP_EPOCHS = 10
BETA_FINAL = 0.3
BETA_RAMP_EPOCHS = 10


# =========================
# 1. Dataset / DataLoader
# =========================

class ASTGCNDataset(Dataset):
    def __init__(self, Xh, Y, Xd=None, Xw=None):
        assert Xh.shape[0] == Y.shape[0], "샘플 수 불일치"
        self.Xh = torch.from_numpy(Xh).float()
        self.Y  = torch.from_numpy(Y).float()
        self.Xd = torch.from_numpy(Xd).float() if Xd is not None else None
        self.Xw = torch.from_numpy(Xw).float() if Xw is not None else None

    def __len__(self):
        return self.Xh.shape[0]

    def __getitem__(self, idx):
        xh = self.Xh[idx]
        y  = self.Y[idx]
        if (self.Xd is not None) and (self.Xw is not None):
            return xh, self.Xd[idx], self.Xw[idx], y
        return xh, y


def get_dataloaders_ast(
    use_periodic: bool,
    batch_size: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    Xh = np.load(data_dir / "Xh_samples.npy")
    Y  = np.load(data_dir / "Y_samples_ast.npy")

    if use_periodic:
        Xd = np.load(data_dir / "Xd_samples.npy")
        Xw = np.load(data_dir / "Xw_samples.npy")
        dataset = ASTGCNDataset(Xh, Y, Xd, Xw)
        print("[AST-GCN] mode = full-periodic (X_h, X_d, X_w)")
    else:
        dataset = ASTGCNDataset(Xh, Y)
        print("[AST-GCN] mode = recent-only (X_h)")

    S, N, T_h, F = Xh.shape
    _, _, T_out, num_targets = Y.shape

    if use_periodic:
        _, _, T_d, _ = Xd.shape
        _, _, T_w, _ = Xw.shape
    else:
        T_d = 0
        T_w = 0

    print(f"[Data] Xh: {Xh.shape}, Y: {Y.shape}")
    if use_periodic:
        print(f"[Data] Xd: {Xd.shape}, Xw: {Xw.shape}")

    rng = np.random.RandomState(seed)
    indices = np.arange(S)
    rng.shuffle(indices)

    n_test = int(S * test_ratio)
    n_val  = int(S * val_ratio)
    n_train = S - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    meta = dict(
        N=N, F=F,
        T_h=T_h, T_d=T_d, T_w=T_w,
        T_out=T_out,
        num_targets=num_targets,
        use_periodic=use_periodic,
    )
    return train_loader, val_loader, test_loader, meta


# =========================
# 2. Edge list 구축
# =========================

def build_edge_index_from_A(A_np: np.ndarray):
    """
    A_np에서 mask(A>0) + self-loop를 사용해 directed edge list 생성.
    반환:
      edge_row, edge_col: (E,)
      row_ptr: (N+1,)  (CSR)  (호환용)
      A_edge: (E,)  edge별 A값
    """
    N = A_np.shape[0]
    mask = (A_np > 0)
    np.fill_diagonal(mask, True)

    rows, cols = np.where(mask)
    order = np.lexsort((cols, rows))
    rows = rows[order].astype(np.int64)
    cols = cols[order].astype(np.int64)

    A_edge = A_np[rows, cols].astype(np.float32)

    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(row_ptr, rows + 1, 1)
    row_ptr = np.cumsum(row_ptr)

    return rows, cols, row_ptr, A_edge


def _has_scatter_reduce() -> bool:
    x = torch.empty(1)
    return hasattr(x, "scatter_reduce_")


def row_softmax_edges_batch(scores_be: torch.Tensor, edge_row_e: torch.Tensor, num_nodes: int, eps: float = 1e-12):
    """
    scores_be : (B,E)
    edge_row_e: (E,)
    return    : (B,E) row-wise softmax (scatter_reduce 기반)
    """
    if not _has_scatter_reduce():
        # (구버전 torch fallback) 정확한 row-softmax는 못하지만 일단 터지진 않게
        return torch.softmax(scores_be, dim=-1)

    B, E = scores_be.shape
    dev = scores_be.device
    dtype = scores_be.dtype
    edge_row = edge_row_e.to(dev, torch.long)

    b_ids = torch.arange(B, device=dev).view(B, 1)                       # (B,1)
    row_id = (b_ids * num_nodes + edge_row.view(1, E)).reshape(-1)       # (B*E,)
    s_flat = scores_be.reshape(-1)                                       # (B*E,)

    max_row = torch.full((B * num_nodes,), -1e15, device=dev, dtype=dtype)
    max_row.scatter_reduce_(0, row_id, s_flat, reduce="amax", include_self=True)

    exp = torch.exp(s_flat - max_row[row_id])
    sum_row = torch.zeros((B * num_nodes,), device=dev, dtype=dtype)
    sum_row.scatter_add_(0, row_id, exp)

    out = exp / (sum_row[row_id] + eps)
    return out.view(B, E)


def row_softmax_edges_1d(scores_e: torch.Tensor, edge_row_e: torch.Tensor, num_nodes: int, eps: float = 1e-12):
    """
    scores_e  : (E,)
    edge_row_e: (E,)
    return    : (E,) row-wise softmax
    """
    if not _has_scatter_reduce():
        return torch.softmax(scores_e, dim=-1)

    dev = scores_e.device
    dtype = scores_e.dtype
    edge_row = edge_row_e.to(dev, torch.long)

    max_row = torch.full((num_nodes,), -1e15, device=dev, dtype=dtype)
    max_row.scatter_reduce_(0, edge_row, scores_e, reduce="amax", include_self=True)

    exp = torch.exp(scores_e - max_row[edge_row])
    sum_row = torch.zeros((num_nodes,), device=dev, dtype=dtype)
    sum_row.scatter_add_(0, edge_row, exp)

    return exp / (sum_row[edge_row] + eps)


def row_renorm_edges_batch(weights_be: torch.Tensor, edge_row_e: torch.Tensor, num_nodes: int, eps: float = 1e-12):
    """
    weights_be: (B,E)
    return    : (B,E) row별 sum=1 재정규화 (scatter_add 기반, in-place 없음)
    """
    B, E = weights_be.shape
    dev = weights_be.device
    dtype = weights_be.dtype
    edge_row = edge_row_e.to(dev, torch.long)

    b_ids = torch.arange(B, device=dev).view(B, 1)
    row_id = (b_ids * num_nodes + edge_row.view(1, E)).reshape(-1)     # (B*E,)
    w_flat = weights_be.reshape(-1)                                     # (B*E,)

    sum_row = torch.zeros((B * num_nodes,), device=dev, dtype=dtype)
    sum_row.scatter_add_(0, row_id, w_flat)

    out = w_flat / (sum_row[row_id] + eps)
    return out.view(B, E)


# =========================
# 3. Edge-level Spatial Attention / Temporal Attention
# =========================

class SpatialAttentionEdges(nn.Module):
    """
    x: (B,N,T,C)
    - node feature flatten -> (B,N,T*C)
    - Q,K 만든 뒤 edge(i,j) score = <Q_i, K_j>/sqrt(d)
    - row(edge_row=i) 기준 row-softmax -> S_edge (B,E)

    row_ptr는 호환용으로 '받기만' 함.
    """
    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        edge_row: torch.Tensor,
        edge_col: torch.Tensor,
        row_ptr: torch.Tensor = None,   # ✅ 여기 때문에 예전 호출(row_ptr=)이 터지면 안 됨
        d_k: int = 32,
    ):
        super().__init__()
        self.in_dim = in_channels * time_steps
        self.d_k = d_k
        self.W_q = nn.Linear(self.in_dim, d_k)
        self.W_k = nn.Linear(self.in_dim, d_k)

        self.register_buffer("edge_row", edge_row.long())
        self.register_buffer("edge_col", edge_col.long())
        # row_ptr는 쓰진 않지만 buffer로 유지(호환)
        if row_ptr is None:
            self.register_buffer("row_ptr", torch.empty(0, dtype=torch.long))
        else:
            self.register_buffer("row_ptr", row_ptr.long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.shape
        x_flat = x.reshape(B, N, T * C)

        Q = self.W_q(x_flat)  # (B,N,d)
        K = self.W_k(x_flat)  # (B,N,d)

        qi = Q[:, self.edge_row, :]  # (B,E,d)
        kj = K[:, self.edge_col, :]  # (B,E,d)

        scores = (qi * kj).sum(dim=-1) / (self.d_k ** 0.5)  # (B,E)
        return row_softmax_edges_batch(scores, self.edge_row, num_nodes=N)


class TemporalAttention(nn.Module):
    """
    x: (B,N,T,C)
    return: (B,T,T)
    """
    def __init__(self, in_channels: int, num_nodes: int, d_k: int = 32):
        super().__init__()
        self.in_dim = in_channels * num_nodes
        self.d_k = d_k
        self.W_q = nn.Linear(self.in_dim, d_k)
        self.W_k = nn.Linear(self.in_dim, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, N * C)
        Q = self.W_q(x_flat)
        K = self.W_k(x_flat)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        return torch.softmax(scores, dim=-1)


# =========================
# 4. Adaptive Adjacency (edge-only, row-softmax vectorized)
# =========================

class AdaptiveAdjacencyMaskedEdges(nn.Module):
    """
    edge row/col 위치만 허용하는 adaptive adjacency를 edge weight로 생성.
    - score_e = ReLU(E_i · E_j)
    - row-softmax => A_adp_edge: (E,)
    """
    def __init__(
        self,
        num_nodes: int,
        edge_row: torch.Tensor,
        edge_col: torch.Tensor,
        row_ptr: torch.Tensor,      # 호환용
        emb_dim: int = 16,
    ):
        super().__init__()
        self.N = int(num_nodes)
        self.Eemb = nn.Parameter(torch.randn(self.N, emb_dim) * 0.1)

        self.register_buffer("edge_row", edge_row.long())
        self.register_buffer("edge_col", edge_col.long())
        self.register_buffer("row_ptr", row_ptr.long())

    def forward(self) -> torch.Tensor:
        score_e = torch.relu((self.Eemb[self.edge_row] * self.Eemb[self.edge_col]).sum(dim=-1))  # (E,)
        return row_softmax_edges_1d(score_e, self.edge_row, num_nodes=self.N)                    # (E,)


# =========================
# 5. Edge GraphConv (scatter_add)
# =========================

class GraphConvEdge(nn.Module):
    """
    x: (B,C_in,T,N)
    msg_edge: (B,E)
    out: (B,C_out,T,N)
    """
    def __init__(self, in_channels: int, out_channels: int, edge_row: torch.Tensor, edge_col: torch.Tensor):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.register_buffer("edge_row", edge_row.long())  # dst
        self.register_buffer("edge_col", edge_col.long())  # src

    def forward(self, x: torch.Tensor, msg_edge: torch.Tensor) -> torch.Tensor:
        x = self.theta(x)  # (B,C_out,T,N)
        B, C, T, N = x.shape
        E = self.edge_row.numel()
        CT = C * T

        out = torch.zeros((B, C, T, N), device=x.device, dtype=x.dtype)
        dst = self.edge_row
        src = self.edge_col

        for b in range(B):
            X = x[b].reshape(CT, N)              # (CT,N)
            X_src = X[:, src]                    # (CT,E)
            W = msg_edge[b].unsqueeze(0)         # (1,E)
            weighted = X_src * W                 # (CT,E)

            Y = torch.zeros((CT, N), device=x.device, dtype=x.dtype)
            Y.scatter_add_(1, dst.expand(CT, E), weighted)
            out[b] = Y.view(C, T, N)

        return out


# =========================
# 6. AST-GCN Block / Encoder / MultiTask
# =========================

class ASTGCNBlockEdge(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        time_steps: int,
        in_channels: int,
        out_channels: int,
        edge_row: torch.Tensor,
        edge_col: torch.Tensor,
        row_ptr: torch.Tensor,
        kernel_size: int = 3,
        dropout: float = 0.1,
        att_dim: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.time_steps = int(time_steps)
        self.use_residual = bool(use_residual)

        padding = kernel_size // 2

        self.t_att = TemporalAttention(in_channels, num_nodes, d_k=att_dim)
        self.s_att_edge = SpatialAttentionEdges(
            in_channels=in_channels,
            time_steps=time_steps,
            edge_row=edge_row,
            edge_col=edge_col,
            row_ptr=row_ptr,          # ✅ 넘겨도 안전
            d_k=att_dim,
        )

        self.temporal1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0))
        self.graph_conv = GraphConvEdge(out_channels, out_channels, edge_row=edge_row, edge_col=edge_col)
        self.temporal2 = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0))

        if self.use_residual and in_channels != out_channels:
            self.res_proj = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.res_proj = None

        self.register_buffer("edge_row", edge_row.long())

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, A_base_edge: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,T,C_in)
        A_base_edge: (E,) = (1-beta)A_fixed_edge + beta*A_adp_edge
        """
        B, N, T, C = x.shape
        assert N == self.num_nodes and T == self.time_steps

        # 1) Temporal attention
        T_att = self.t_att(x)  # (B,T,T)
        x_t = torch.einsum("bts, bnsf -> bntf", T_att, x)  # (B,N,T,C)

        # 2) Spatial attention edge-level
        S_edge = self.s_att_edge(x_t)  # (B,E) row-softmax

        # 3) msg_edge = A_base_edge * S_edge, then row-renorm (scatter)
        msg_edge = S_edge * A_base_edge.unsqueeze(0)              # (B,E)
        msg_edge = row_renorm_edges_batch(msg_edge, self.edge_row, num_nodes=N)

        # 4) Conv + Edge GraphConv
        x_conv_in = x_t.permute(0, 3, 2, 1)  # (B,C,T,N)
        identity = x_conv_in

        out = self.relu(self.temporal1(x_conv_in))
        out = self.relu(self.graph_conv(out, msg_edge))
        out = self.temporal2(out)

        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        out = self.dropout(self.relu(out))
        return out.permute(0, 3, 2, 1)  # (B,N,T,C_out)


class ASTGCNEncoderEdge(nn.Module):
    def __init__(
        self,
        N_nodes: int,
        T_in: int,
        F_in: int,
        edge_row: torch.Tensor,
        edge_col: torch.Tensor,
        row_ptr: torch.Tensor,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        att_dim: int = 32,
    ):
        super().__init__()
        blocks = []
        in_c = F_in
        for _ in range(num_blocks):
            blocks.append(
                ASTGCNBlockEdge(
                    num_nodes=N_nodes,
                    time_steps=T_in,
                    in_channels=in_c,
                    out_channels=hidden_channels,
                    edge_row=edge_row,
                    edge_col=edge_col,
                    row_ptr=row_ptr,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    att_dim=att_dim,
                    use_residual=True,
                )
            )
            in_c = hidden_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, A_base_edge: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, A_base_edge)
        return x[:, :, -1, :]  # (B,N,H)


class ASTGCNMultiTaskEdge(nn.Module):
    def __init__(
        self,
        meta: dict,
        A_fixed_np: np.ndarray,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        att_dim: int = 32,
        use_adaptive_adj: bool = True,
        adj_emb_dim: int = 16,
        adaptive_warmup_epochs: int = 10,
        beta_final: float = 0.3,
        beta_ramp_epochs: int = 10,
    ):
        super().__init__()
        N = meta["N"]
        F = meta["F"]
        T_h = meta["T_h"]
        T_d = meta["T_d"]
        T_w = meta["T_w"]
        T_out = meta["T_out"]
        num_targets = meta["num_targets"]
        use_periodic = meta["use_periodic"]

        self.use_periodic = bool(use_periodic)
        self.T_out = int(T_out)
        self.num_targets = int(num_targets)

        # edge list 구축
        rows, cols, row_ptr, A_edge = build_edge_index_from_A(A_fixed_np)
        edge_row = torch.from_numpy(rows)
        edge_col = torch.from_numpy(cols)
        row_ptr_t = torch.from_numpy(row_ptr)
        A_fixed_edge = torch.from_numpy(A_edge)

        self.register_buffer("edge_row", edge_row.long())
        self.register_buffer("edge_col", edge_col.long())
        self.register_buffer("row_ptr",  row_ptr_t.long())          # 호환용
        self.register_buffer("A_fixed_edge", A_fixed_edge.float())  # (E,)

        # adaptive schedule
        self.use_adaptive_adj = bool(use_adaptive_adj)
        self.adaptive_warmup_epochs = int(adaptive_warmup_epochs)
        self.beta_final = float(beta_final)
        self.beta_ramp_epochs = int(beta_ramp_epochs)
        self.current_epoch = 0
        self.cur_beta = 0.0

        if self.use_adaptive_adj:
            self.adp_adj = AdaptiveAdjacencyMaskedEdges(
                num_nodes=N,
                edge_row=self.edge_row,
                edge_col=self.edge_col,
                row_ptr=self.row_ptr,
                emb_dim=adj_emb_dim,
            )
        else:
            self.adp_adj = None

        # encoders
        self.enc_h = ASTGCNEncoderEdge(
            N_nodes=N, T_in=T_h, F_in=F,
            edge_row=self.edge_row, edge_col=self.edge_col, row_ptr=self.row_ptr,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            att_dim=att_dim,
        )

        if self.use_periodic and T_d > 0:
            self.enc_d = ASTGCNEncoderEdge(
                N_nodes=N, T_in=T_d, F_in=F,
                edge_row=self.edge_row, edge_col=self.edge_col, row_ptr=self.row_ptr,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dropout=dropout,
                att_dim=att_dim,
            )
        else:
            self.enc_d = None

        if self.use_periodic and T_w > 0:
            self.enc_w = ASTGCNEncoderEdge(
                N_nodes=N, T_in=T_w, F_in=F,
                edge_row=self.edge_row, edge_col=self.edge_col, row_ptr=self.row_ptr,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dropout=dropout,
                att_dim=att_dim,
            )
        else:
            self.enc_w = None

        fusion_in = hidden_channels
        if self.enc_d is not None:
            fusion_in += hidden_channels
        if self.enc_w is not None:
            fusion_in += hidden_channels

        self.fc_fusion = nn.Linear(fusion_in, hidden_channels)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_channels, T_out * num_targets)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        if self.current_epoch <= self.adaptive_warmup_epochs:
            self.cur_beta = 0.0
        else:
            t = (self.current_epoch - self.adaptive_warmup_epochs) / max(1, self.beta_ramp_epochs)
            t = min(1.0, max(0.0, t))
            self.cur_beta = self.beta_final * t

    def _get_A_base_edge(self) -> torch.Tensor:
        if (not self.use_adaptive_adj) or (self.adp_adj is None) or (self.cur_beta <= 0.0):
            return self.A_fixed_edge
        A_adp_edge = self.adp_adj()  # (E,)
        return (1.0 - self.cur_beta) * self.A_fixed_edge + self.cur_beta * A_adp_edge

    def forward(self, xh: torch.Tensor, xd=None, xw=None) -> torch.Tensor:
        A_base_edge = self._get_A_base_edge()

        h_h = self.enc_h(xh, A_base_edge)
        feats = [h_h]

        if self.enc_d is not None and xd is not None:
            feats.append(self.enc_d(xd, A_base_edge))

        if self.enc_w is not None and xw is not None:
            feats.append(self.enc_w(xw, A_base_edge))

        if len(feats) == 1:
            h = feats[0]
        else:
            h_cat = torch.cat(feats, dim=-1)
            h = self.relu(self.fc_fusion(h_cat))

        y_flat = self.fc_out(h)
        B, N, _ = y_flat.shape
        return y_flat.view(B, N, self.T_out, self.num_targets)


# =========================
# 7. 학습 / 평가
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_periodic: bool,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = total_mae_all = total_mae_tr = total_mae_sp = 0.0
    total_count = 0

    for batch in loader:
        if use_periodic:
            xh, xd, xw, yb = batch
        else:
            xh, yb = batch
            xd = xw = None

        xh = xh.to(device)
        yb = yb.to(device)
        if xd is not None:
            xd = xd.to(device)
        if xw is not None:
            xw = xw.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(xh, xd, xw)
        loss = criterion(y_hat, yb)
        loss.backward()
        optimizer.step()

        B = xh.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()
        mae_all = diff.mean().item()
        mae_tr = diff[..., 0].mean().item()
        mae_sp = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_tr  += mae_tr  * B
        total_mae_sp  += mae_sp  * B
        total_count   += B

    return (
        total_loss / total_count,
        total_mae_all / total_count,
        total_mae_tr  / total_count,
        total_mae_sp  / total_count,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    use_periodic: bool,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = total_mae_all = total_mae_tr = total_mae_sp = 0.0
    total_count = 0

    for batch in loader:
        if use_periodic:
            xh, xd, xw, yb = batch
        else:
            xh, yb = batch
            xd = xw = None

        xh = xh.to(device)
        yb = yb.to(device)
        if xd is not None:
            xd = xd.to(device)
        if xw is not None:
            xw = xw.to(device)

        y_hat = model(xh, xd, xw)
        loss = criterion(y_hat, yb)

        B = xh.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()
        mae_all = diff.mean().item()
        mae_tr = diff[..., 0].mean().item()
        mae_sp = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_tr  += mae_tr  * B
        total_mae_sp  += mae_sp  * B
        total_count   += B

    return (
        total_loss / total_count,
        total_mae_all / total_count,
        total_mae_tr  / total_count,
        total_mae_sp  / total_count,
    )


# =========================
# 8. 메인
# =========================

def main(use_periodic: bool):
    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=2,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    print("Meta:", meta)

    A_fixed_np = np.load(A_path).astype(np.float32)
    assert A_fixed_np.shape == (meta["N"], meta["N"]), "adjacency shape mismatch"

    model = ASTGCNMultiTaskEdge(
        meta=meta,
        A_fixed_np=A_fixed_np,
        hidden_channels=32,
        num_blocks=1,
        kernel_size=3,
        dropout=0.1,
        att_dim=32,
        use_adaptive_adj=USE_ADAPTIVE_ADJ,
        adj_emb_dim=ADJ_EMB_DIM,
        adaptive_warmup_epochs=ADAPTIVE_WARMUP_EPOCHS,
        beta_final=BETA_FINAL,
        beta_ramp_epochs=BETA_RAMP_EPOCHS,
    ).to(device)

    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    num_epochs = 100
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.set_epoch(epoch)

        tr_loss, tr_mae_all, tr_mae_tr, tr_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        print(
            f"[Epoch {epoch:03d}] beta={model.cur_beta:.3f} | "
            f"Train Loss: {tr_loss:.4f}, MAE(all): {tr_mae_all:.4f}, "
            f"MAE(tr): {tr_mae_tr:.4f}, MAE(sp): {tr_mae_sp:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
            f"MAE(tr): {val_mae_tr:.4f}, MAE(sp): {val_mae_sp:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion, use_periodic
    )
    print(
        f"[Test] Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(tr): {test_mae_tr:.4f}, MAE(sp): {test_mae_sp:.4f}"
    )

    tag = "full" if use_periodic else "recent"
    save_path = data_dir / f"astgcn_multitask_{tag}_edgeatt_adaptive_best.pth"
    torch.save(
        {
            "model_type": "astgcn_multitask_edgeatt_adaptive",
            "use_periodic": use_periodic,
            "state_dict": model.state_dict(),
            "meta": meta,
            "config": {
                "A_path_used": str(A_path),
                "use_adaptive_adj": USE_ADAPTIVE_ADJ,
                "adj_emb_dim": ADJ_EMB_DIM,
                "adaptive_warmup_epochs": ADAPTIVE_WARMUP_EPOCHS,
                "beta_final": BETA_FINAL,
                "beta_ramp_epochs": BETA_RAMP_EPOCHS,
            },
            "test_metrics": {
                "loss": float(test_loss),
                "mae_all": float(test_mae_all),
                "mae_traffic": float(test_mae_tr),
                "mae_speed": float(test_mae_sp),
            },
        },
        save_path,
    )
    print("Saved best model to:", save_path)


if __name__ == "__main__":
    # main(use_periodic=False)   # recent-only
    main(use_periodic=True)      # full-periodic
