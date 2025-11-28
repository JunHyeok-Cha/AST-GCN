# astgcn_experiment.py
#
# AST-GCN (Attention based Spatio-Temporal Graph Convolutional Network)
# - Multi-task: [TotalTraffic, Speed]
# - Multi-stream: X_h (recent), X_d (daily), X_w (weekly)
# - Spatio-Temporal Attention:
#     * Spatial Attention SAtt(X): (B, N, N)
#     * Temporal Attention TAtt(X): (B, T, T)
#     * X_att = SAtt ⊗ TAtt ⊗ X  (노드/시간 모두 가중합)
# - Graph Convolution + Temporal Convolution (+ Residual)
#
# 전제:
#   res 디렉토리에 다음 파일이 존재한다고 가정한다.
#     - Xh_samples.npy  : (S, N, T_h, F)
#     - Xd_samples.npy  : (S, N, T_d, F)   # full-periodic 모드일 때만 필요
#     - Xw_samples.npy  : (S, N, T_w, F)   # full-periodic 모드일 때만 필요
#     - Y_samples_ast.npy: (S, N, T_out, 2)
#     - adjacency_norm.npy: (N, N)

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# 0. 공통 설정
# =========================

data_dir = Path("/mnt/c/Source/python/AST-GCN/res")
A_path   = data_dir / "adjacency_norm.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 1. Dataset / DataLoader
# =========================

class ASTGCNDataset(Dataset):
    """
    AST-GCN용 Dataset.

    - recent-only 모드: X_h, Y 만 사용
    - full-periodic 모드: X_h, X_d, X_w, Y 모두 사용
    """
    def __init__(self, Xh, Y, Xd=None, Xw=None):
        assert Xh.shape[0] == Y.shape[0], "샘플 수 불일치"
        self.Xh = torch.from_numpy(Xh).float()  # (S, N, T_h, F)
        self.Y  = torch.from_numpy(Y).float()   # (S, N, T_out, 2)

        if Xd is not None:
            assert Xd.shape[0] == Xh.shape[0], "Xd 샘플 수 불일치"
            self.Xd = torch.from_numpy(Xd).float()  # (S, N, T_d, F)
        else:
            self.Xd = None

        if Xw is not None:
            assert Xw.shape[0] == Xh.shape[0], "Xw 샘플 수 불일치"
            self.Xw = torch.from_numpy(Xw).float()  # (S, N, T_w, F)
        else:
            self.Xw = None

    def __len__(self):
        return self.Xh.shape[0]

    def __getitem__(self, idx):
        xh = self.Xh[idx]  # (N, T_h, F)
        y  = self.Y[idx]   # (N, T_out, 2)
        if (self.Xd is not None) and (self.Xw is not None):
            xd = self.Xd[idx]  # (N, T_d, F)
            xw = self.Xw[idx]  # (N, T_w, F)
            return xh, xd, xw, y
        else:
            # recent-only
            return xh, y


def get_dataloaders_ast(
    use_periodic: bool,
    batch_size: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    use_periodic:
      False → recent-only AST-GCN (X_h만 사용)
      True  → full AST-GCN (X_h, X_d, X_w 모두 사용)
    """
    Xh = np.load(data_dir / "Xh_samples.npy")        # (S, N, T_h, F)
    Y  = np.load(data_dir / "Y_samples_ast.npy")     # (S, N, T_out, 2)

    if use_periodic:
        Xd = np.load(data_dir / "Xd_samples.npy")    # (S, N, T_d, F)
        Xw = np.load(data_dir / "Xw_samples.npy")    # (S, N, T_w, F)
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

    # ----- Train / Val / Test split -----
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
        N=N,
        F=F,
        T_h=T_h,
        T_d=T_d,
        T_w=T_w,
        T_out=T_out,
        num_targets=num_targets,
        use_periodic=use_periodic,
    )
    return train_loader, val_loader, test_loader, meta


# =========================
# 2. Spatio-Temporal Attention + GCN 블록
# =========================

class SpatialAttention(nn.Module):
    """
    Spatial Attention SAtt(X).
    입력: x (B, N, T, C)
    출력: S (B, N, N)  - 각 노드 쌍의 중요도

    구현 아이디어:
      - 시간(T)과 채널(C)을 flatten → (B, N, T*C)
      - Linear → query/key 로 변환
      - Q * K^T / sqrt(d_k) → softmax → attention matrix
    """
    def __init__(self, in_channels: int, time_steps: int, d_k: int = 32):
        super().__init__()
        self.in_dim = in_channels * time_steps
        self.d_k = d_k
        self.W_q = nn.Linear(self.in_dim, d_k)
        self.W_k = nn.Linear(self.in_dim, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, C)
        B, N, T, C = x.shape
        x_flat = x.reshape(B, N, T * C)  # (B, N, T*C)

        Q = self.W_q(x_flat)            # (B, N, d_k)
        K = self.W_k(x_flat)            # (B, N, d_k)

        # attention scores: (B, N, N)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        S = torch.softmax(scores, dim=-1)
        return S


class TemporalAttention(nn.Module):
    """
    Temporal Attention TAtt(X).
    입력: x (B, N, T, C)
    출력: T (B, T, T)  - 각 시간 스텝 쌍의 중요도

    구현 아이디어:
      - 노드(N)와 채널(C)을 flatten → (B, T, N*C)
      - Linear → query/key
      - Q * K^T / sqrt(d_k) → softmax
    """
    def __init__(self, in_channels: int, num_nodes: int, d_k: int = 32):
        super().__init__()
        self.in_dim = in_channels * num_nodes
        self.d_k = d_k
        self.W_q = nn.Linear(self.in_dim, d_k)
        self.W_k = nn.Linear(self.in_dim, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, C)
        B, N, T, C = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, N * C)  # (B, T, N*C)

        Q = self.W_q(x_flat)             # (B, T, d_k)
        K = self.W_k(x_flat)             # (B, T, d_k)

        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)  # (B, T, T)
        T_att = torch.softmax(scores, dim=-1)
        return T_att


class GraphConv(nn.Module):
    """
    간단한 Graph Convolution (A는 정규화된 adjacency).

    입력:
      x: (B, C_in, T, N)
      A: (N, N)

    출력:
      y: (B, C_out, T, N)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 1) 채널 변환
        x = self.theta(x)  # (B, C_out, T, N)
        # 2) 인접행렬 기반 메시지 전달
        x = torch.einsum("ij, bctj -> bcti", A, x)
        return x


class ASTGCNBlock(nn.Module):
    """
    하나의 AST-GCN 블록:
      - Spatial Attention: SAtt(X)
      - Temporal Attention: TAtt(X)
      - Attended X → GraphConv(A_hat) + TemporalConv
      - Residual Connection + Dropout

    입력/출력:
      x: (B, N, T, C_in)
      out: (B, N, T, C_out)
    """
    def __init__(
        self,
        num_nodes: int,
        time_steps: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_residual: bool = True,
        dropout: float = 0.1,
        att_dim: int = 32,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual

        padding = kernel_size // 2

        # Spatio-Temporal Attention
        self.s_att = SpatialAttention(in_channels, time_steps, d_k=att_dim)
        self.t_att = TemporalAttention(in_channels, num_nodes, d_k=att_dim)

        # GCN + Temporal Convolution (Conv2d on (C, T, N))
        self.temporal1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )
        self.graph_conv = GraphConv(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.temporal2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )

        # Residual projection (채널 수 변경되는 경우)
        if use_residual and in_channels != out_channels:
            self.res_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            )
        else:
            self.res_proj = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, C_in)
        A: (N, N)
        return: (B, N, T, C_out)
        """
        B, N, T, C_in = x.shape
        assert N == self.num_nodes
        assert T == self.time_steps

        # ----- Spatio-Temporal Attention -----
        S = self.s_att(x)  # (B, N, N)
        T_att = self.t_att(x)  # (B, T, T)

        # Temporal Attention: (B, N, T, C) -> (B, N, T, C)
        # y[b, n, t, c] = sum_s T_att[b, t, s] * x[b, n, s, c]
        x_t = torch.einsum("bts, bnsf -> bntf", T_att, x)

        # Spatial Attention: (B, N, T, C) -> (B, N, T, C)
        # y[b, n, t, c] = sum_m S[b, n, m] * x_t[b, m, t, c]
        x_st = torch.einsum("bnm, bmtf -> bntf", S, x_t)

        # ----- GCN + Temporal Convolution -----
        # Conv2d 입력 형식으로 변환: (B, C_in, T, N)
        x_conv_in = x_st.permute(0, 3, 2, 1)  # (B, C_in, T, N)
        identity = x_conv_in

        out = self.temporal1(x_conv_in)
        out = self.relu(out)

        out = self.graph_conv(out, A)
        out = self.relu(out)

        out = self.temporal2(out)

        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        out = self.relu(out)
        out = self.dropout(out)

        # 다시 (B, N, T, C_out) 로
        out = out.permute(0, 3, 2, 1)
        return out


class ASTGCNEncoder(nn.Module):
    """
    한 스트림(X_h / X_d / X_w)을 처리하는 인코더.
    입력: x (B, N, T_in, F)
    출력: h (B, N, H)  - 마지막 시간 스텝의 hidden feature
    """
    def __init__(
        self,
        N_nodes: int,
        T_in: int,
        F_in: int,
        A_norm: np.ndarray,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        att_dim: int = 32,
    ):
        super().__init__()
        self.N_nodes = N_nodes
        self.T_in = T_in
        self.F_in = F_in

        A = torch.tensor(A_norm, dtype=torch.float32)
        self.register_buffer("A", A)

        blocks = []
        in_c = F_in
        for _ in range(num_blocks):
            block = ASTGCNBlock(
                num_nodes=N_nodes,
                time_steps=T_in,
                in_channels=in_c,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                use_residual=True,
                dropout=dropout,
                att_dim=att_dim,
            )
            blocks.append(block)
            in_c = hidden_channels

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F_in)
        return: (B, N, hidden_channels)
        """
        B, N, T, F = x.shape
        assert N == self.N_nodes

        for block in self.blocks:
            x = block(x, self.A)  # (B, N, T, hidden)

        # 마지막 시간 스텝의 hidden 사용
        h_last = x[:, :, -1, :]  # (B, N, hidden)
        return h_last


class ASTGCNMultiTask(nn.Module):
    """
    AST-GCN 멀티태스크 모델.

    - recent-only 모드: encoder_h 만 사용
    - full-periodic 모드: encoder_h, encoder_d, encoder_w 모두 사용 후 fusion
    """
    def __init__(
        self,
        meta: dict,
        A_norm: np.ndarray,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        att_dim: int = 32,
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

        self.use_periodic = use_periodic
        self.T_out = T_out
        self.num_targets = num_targets
        self.hidden_channels = hidden_channels

        # recent encoder (항상 존재)
        self.enc_h = ASTGCNEncoder(
            N_nodes=N,
            T_in=T_h,
            F_in=F,
            A_norm=A_norm,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            att_dim=att_dim,
        )

        # daily / weekly encoder (옵션)
        if use_periodic and T_d > 0:
            self.enc_d = ASTGCNEncoder(
                N_nodes=N,
                T_in=T_d,
                F_in=F,
                A_norm=A_norm,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dropout=dropout,
                att_dim=att_dim,
            )
        else:
            self.enc_d = None

        if use_periodic and T_w > 0:
            self.enc_w = ASTGCNEncoder(
                N_nodes=N,
                T_in=T_w,
                F_in=F,
                A_norm=A_norm,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dropout=dropout,
                att_dim=att_dim,
            )
        else:
            self.enc_w = None

        # fusion: [h_h, h_d, h_w] concat 후 Linear
        fusion_in_dim = hidden_channels
        if self.enc_d is not None:
            fusion_in_dim += hidden_channels
        if self.enc_w is not None:
            fusion_in_dim += hidden_channels

        self.fc_fusion = nn.Linear(fusion_in_dim, hidden_channels)
        self.relu = nn.ReLU()

        # 최종 출력: T_out * num_targets
        self.fc_out = nn.Linear(hidden_channels, T_out * num_targets)

    def forward(self, xh: torch.Tensor, xd=None, xw=None) -> torch.Tensor:
        """
        xh: (B, N, T_h, F)
        xd: (B, N, T_d, F) or None
        xw: (B, N, T_w, F) or None
        return: y_hat (B, N, T_out, num_targets)
        """
        h_h = self.enc_h(xh)  # (B, N, H)
        feats = [h_h]

        if self.enc_d is not None and xd is not None:
            h_d = self.enc_d(xd)  # (B, N, H)
            feats.append(h_d)

        if self.enc_w is not None and xw is not None:
            h_w = self.enc_w(xw)  # (B, N, H)
            feats.append(h_w)

        if len(feats) == 1:
            h = feats[0]
        else:
            h_cat = torch.cat(feats, dim=-1)      # (B, N, H * (#streams))
            h = self.relu(self.fc_fusion(h_cat))  # (B, N, H)

        y_flat = self.fc_out(h)  # (B, N, T_out * num_targets)
        B, N, _ = y_flat.shape
        y = y_flat.view(B, N, self.T_out, self.num_targets)
        return y


# =========================
# 3. 학습 / 평가 루프
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_periodic: bool,
) -> Tuple[float, float, float, float]:
    """
    한 epoch 학습.
    반환: loss, MAE(all), MAE(traffic), MAE(speed)
    """
    model.train()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_tr = 0.0
    total_mae_sp = 0.0
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

        optimizer.zero_grad()
        y_hat = model(xh, xd, xw)
        loss = criterion(y_hat, yb)
        loss.backward()
        optimizer.step()

        B = xh.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()  # (B, N, T_out, 2)
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
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_tr = 0.0
    total_mae_sp = 0.0
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
# 4. 메인 실행
# =========================

def main(use_periodic: bool):
    # 1) DataLoader 준비
    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=2,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    print("Meta:", meta)

    # 2) 인접행렬 로드
    A_norm = np.load(A_path)
    assert A_norm.shape == (meta["N"], meta["N"]), "adjacency_norm.npy의 N이 샘플과 다름"

    # 3) AST-GCN 모델 생성
    model = ASTGCNMultiTask(
        meta=meta,
        A_norm=A_norm,
        hidden_channels=32,
        num_blocks=1,
        kernel_size=3,
        dropout=0.1,
        att_dim=32,
    ).to(device)

    print(model)

    # 4) 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5,
    )

    num_epochs = 100
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, MAE(all): {train_mae_all:.4f}, "
            f"MAE(tr): {train_mae_tr:.4f}, MAE(sp): {train_mae_sp:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
            f"MAE(tr): {val_mae_tr:.4f}, MAE(sp): {val_mae_sp:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # 5) best 모델로 Test 평가
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion, use_periodic
    )
    print(
        f"[Test] Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(tr): {test_mae_tr:.4f}, MAE(sp): {test_mae_sp:.4f}"
    )

    # 6) 모델 저장
    tag = "full" if use_periodic else "recent"
    save_path = data_dir / f"astgcn_multitask_{tag}_best.pth"
    torch.save(
        {
            "model_type": "astgcn_multitask",
            "use_periodic": use_periodic,
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        save_path,
    )
    print("Saved best model to:", save_path)


if __name__ == "__main__":
    # 1) recent-only AST-GCN (X_h만 사용하는 버전)
    main(use_periodic=False)

    # 2) X_h + X_d + X_w 모두 사용하는 full AST-GCN
    # main(use_periodic=True)
