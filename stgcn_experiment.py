# stgcn_experiment.py (multi-task + 튜닝/Residual 버전)
#
# 목적:
#   - X_samples.npy, Y_samples.npy, adjacency_norm.npy 를 사용한
#     "멀티태스크 ST-GCN" 모델 학습/평가.
#   - 출력: TotalTraffic(t+1), Speed(t+1) 동시 예측
#
# 변경 사항 (이전 버전 대비):
#   - learning rate 기본값: 1e-3 → 5e-4 (config에서 쉽게 수정 가능)
#   - hidden_channels 기본값: 64 → 32 (config에서 쉽게 수정 가능)
#   - num_blocks 기본값: 2 → 1 (over-smoothing 완화 목적)
#   - STGCNBlock 에 residual connection 추가
#       - in_channels != out_channels 인 경우 1x1 conv 로 projection
#   - block 내부에 dropout 약간 추가 (과적합/폭주 완화)

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# 0. 공통 설정 + 하이퍼파라미터
# =========================

data_dir = Path("/mnt/c/Source/python/AST-GCN/res")

X_path = data_dir / "X_samples.npy"      # (S, N, T_in, F)
Y_path = data_dir / "Y_samples.npy"      # (S, N, T_out, 2)
A_path = data_dir / "adjacency_norm.npy" # (N, N)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔧 튜닝하기 쉽게 위에 모아둠
HIDDEN_CHANNELS = 32       # 32 / 64 / 128 등 바꿔가면서 실험
NUM_BLOCKS      = 1        # 1 or 2 (over-smoothing 피하려면 1부터)
LR              = 5e-4     # 1e-3 → 5e-4 / 1e-4 등 시도
DROPOUT_P       = 0.1      # block 내부 dropout 비율


# =========================
# 1. Dataset / DataLoader
# =========================

class TrafficSamplesDataset(Dataset):
    """
    X_samples, Y_samples 를 래핑하는 Dataset.

    X: (S, N, T_in, F)
    Y: (S, N, T_out, 2)  ← [TotalTraffic, Speed]
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0], "샘플 개수(S)가 서로 다름"
        self.X = torch.from_numpy(X).float()  # (S, N, T_in, F)
        self.Y = torch.from_numpy(Y).float()  # (S, N, T_out, 2)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    X_path: Path,
    Y_path: Path,
    batch_size: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    X = np.load(X_path)  # (S, N, T_in, F)
    Y = np.load(Y_path)  # (S, N, T_out, 2)

    S, N, T_in, F = X.shape
    S2, N2, T_out, num_targets = Y.shape
    assert S == S2 and N == N2, "X, Y의 샘플 개수 / 노드 수가 다름"
    assert num_targets == 2, "Y 마지막 축은 2 (TotalTraffic, Speed) 이어야 함"

    print(f"[Data] X: {X.shape}, Y: {Y.shape}")
    print(f"[Data] N={N}, T_in={T_in}, F={F}, T_out={T_out}, num_targets={num_targets}")

    dataset = TrafficSamplesDataset(X, Y)

    rng = np.random.RandomState(seed)
    indices = np.arange(S)
    rng.shuffle(indices)

    n_test = int(S * test_ratio)
    n_val = int(S * val_ratio)
    n_train = S - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[Split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    meta = dict(N=N, T_in=T_in, F=F, T_out=T_out, num_targets=num_targets)
    return train_loader, val_loader, test_loader, meta


# =========================
# 2. ST-GCN 관련 모듈 (Residual + Dropout 추가)
# =========================

class GraphConv(nn.Module):
    """
    (아주 단순한) 그래프 컨볼루션 레이어.

    입력:
      x: (B, C_in, T, N)
      A: (N, N)

    동작:
      1) 채널 방향 1x1 Conv (theta) 로 피처 변환
      2) 인접행렬 A 를 사용해 노드 방향으로 메시지 전달:
         y[b, c, t, i] = sum_j A[i, j] * x_theta[b, c, t, j]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.theta(x)  # (B, C_out, T, N)
        x = torch.einsum("ij, bctj -> bcti", A, x)  # (B, C_out, T, N)
        return x


class STGCNBlock(nn.Module):
    """
    하나의 ST-GCN 블록 (Residual + Dropout 포함 버전).

    구조:
      input x -> TemporalConv1 -> ReLU
              -> GraphConv     -> ReLU
              -> TemporalConv2
              -> (Residual Add) -> ReLU
              -> Dropout

    - in_channels != out_channels 인 경우,
      residual 연결 전에 1x1 Conv 로 projection 진행.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.use_residual = use_residual
        self.dropout_p = dropout

        # 시간축에만 conv
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

        # residual projection (채널 수가 다를 때만 사용)
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
        x: (B, C_in, T, N)
        A: (N, N)
        return: (B, C_out, T, N)
        """
        identity = x

        # 1) Temporal Conv -> ReLU
        out = self.temporal1(x)
        out = self.relu(out)

        # 2) Graph Conv -> ReLU
        out = self.graph_conv(out, A)
        out = self.relu(out)

        # 3) 다시 Temporal Conv
        out = self.temporal2(out)

        # 4) Residual 연결
        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        # 5) ReLU + Dropout
        out = self.relu(out)
        out = self.dropout(out)

        return out


class STGCNMultiTask(nn.Module):
    """
    전체 ST-GCN 멀티태스크 모델.

    입력:
      x: (B, N, T_in, F)

    출력:
      y_hat: (B, N, T_out, num_targets=2)
    """

    def __init__(
        self,
        N_nodes: int,
        T_in: int,
        F_in: int,
        T_out: int,
        num_targets: int,
        A_norm: np.ndarray,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.N_nodes = N_nodes
        self.T_in = T_in
        self.F_in = F_in
        self.T_out = T_out
        self.num_targets = num_targets

        A = torch.tensor(A_norm, dtype=torch.float32)
        self.register_buffer("A", A)

        blocks = []
        in_c = F_in
        for b in range(num_blocks):
            block = STGCNBlock(
                in_channels=in_c,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                use_residual=use_residual,
                dropout=dropout,
            )
            blocks.append(block)
            in_c = hidden_channels

        self.blocks = nn.ModuleList(blocks)

        # 마지막 hidden feature(C) → T_out * num_targets
        self.fc_out = nn.Linear(hidden_channels, T_out * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F) → y: (B, N, T_out, num_targets)
        """
        B, N, T_in, F = x.shape
        assert N == self.N_nodes, "노드 수 N이 인접행렬과 맞지 않습니다"
        assert T_in == self.T_in and F == self.F_in

        # (B, N, T, F) -> (B, F, T, N)
        x = x.permute(0, 3, 2, 1)  # (B, F, T_in, N)

        for block in self.blocks:
            x = block(x, self.A)  # (B, hidden, T_in, N)

        # 마지막 시간 스텝만 사용
        h_last = x[:, :, -1, :]    # (B, hidden, N)
        h_last = h_last.permute(0, 2, 1)  # (B, N, hidden)

        y_flat = self.fc_out(h_last)  # (B, N, T_out * num_targets)
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
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_tr = 0.0
    total_mae_sp = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        y_hat = model(xb)
        loss = criterion(y_hat, yb)

        loss.backward()
        optimizer.step()

        B = xb.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()
        mae_all = diff.mean().item()
        mae_tr = diff[..., 0].mean().item()
        mae_sp = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_tr += mae_tr * B
        total_mae_sp += mae_sp * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae_all = total_mae_all / total_count
    avg_mae_tr = total_mae_tr / total_count
    avg_mae_sp = total_mae_sp / total_count
    return avg_loss, avg_mae_all, avg_mae_tr, avg_mae_sp


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_tr = 0.0
    total_mae_sp = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = criterion(y_hat, yb)

        B = xb.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()
        mae_all = diff.mean().item()
        mae_tr = diff[..., 0].mean().item()
        mae_sp = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_tr += mae_tr * B
        total_mae_sp += mae_sp * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae_all = total_mae_all / total_count
    avg_mae_tr = total_mae_tr / total_count
    avg_mae_sp = total_mae_sp / total_count
    return avg_loss, avg_mae_all, avg_mae_tr, avg_mae_sp


# =========================
# 4. 메인 실행
# =========================

def main():
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=2,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    N = meta["N"]
    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]
    print("Meta:", meta)

    A_norm = np.load(A_path)  # (N, N)
    assert A_norm.shape == (N, N), "A_norm shape이 X의 N과 다릅니다"
    print("[Adjacency] Loaded A_norm:", A_norm.shape)

    model = STGCNMultiTask(
        N_nodes=N,
        T_in=T_in,
        F_in=F,
        T_out=T_out,
        num_targets=num_targets,
        A_norm=A_norm,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS,
        kernel_size=3,
        dropout=DROPOUT_P,
        use_residual=True,
    ).to(device)

    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )

    num_epochs = 100
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, MAE(all): {train_mae_all:.4f}, "
            f"MAE(traffic): {train_mae_tr:.4f}, MAE(speed): {train_mae_sp:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
            f"MAE(traffic): {val_mae_tr:.4f}, MAE(speed): {val_mae_sp:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion
    )
    print(
        f"[Test] Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, MAE(speed): {test_mae_sp:.4f}"
    )

    save_path = data_dir / "stgcn_multitask_tuned_best.pth"
    torch.save(
        {
            "model_type": "stgcn_multitask_tuned",
            "state_dict": model.state_dict(),
            "meta": meta,
            "config": {
                "hidden_channels": HIDDEN_CHANNELS,
                "num_blocks": NUM_BLOCKS,
                "lr": LR,
                "dropout_p": DROPOUT_P,
            },
        },
        save_path,
    )
    print("Saved best model to:", save_path)


if __name__ == "__main__":
    main()
