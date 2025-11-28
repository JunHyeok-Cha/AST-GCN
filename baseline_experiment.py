# baseline_experiment.py (multi-task 버전)
#
# 1) X_samples.npy, Y_samples.npy 로부터 DataLoader 생성
# 2) No-Graph Baseline 모델(MLP / LSTM) 정의
#    - 출력: TotalTraffic(t+1), Speed(t+1) 동시 예측 (multi-task)
# 3) 학습 루프 실행 및 성능 출력
#
# ⚠ 전제: 전처리 단계에서 이미
#   - T_in, T_out
#   - X_samples.npy: (num_samples, N, T_in, F)
#   - Y_samples.npy: (num_samples, N, T_out, 2)
#     여기서 마지막 축 2개 채널은 [TotalTraffic, Speed]
#   이 저장되어 있다고 가정한다.

from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# 0. 공통 설정
# =========================

data_dir = Path("/mnt/c/Source/python/AST-GCN/res")

# 🔥 multi-task용 X/Y 파일 (TotalTraffic + Speed 동시 예측)
X_path = data_dir / "X_samples.npy"
Y_path = data_dir / "Y_samples.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 1. Dataset 정의
# =========================

class TrafficSamplesDataset(Dataset):
    """
    X_samples, Y_samples 를 래핑하는 Dataset.
    
    X: (S, N, T_in, F)
    Y: (S, N, T_out, 2)  ← [TotalTraffic, Speed]
    
    한 "샘플"은 하나의 "시계열 윈도우 (T_in 시간)" 이고,
    그 안에 N개의 노드가 모두 포함되어 있다.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0], "샘플 개수 불일치"
        self.X = torch.from_numpy(X).float()  # (S, N, T_in, F)
        self.Y = torch.from_numpy(Y).float()  # (S, N, T_out, 2)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # 반환 형태: (N, T_in, F), (N, T_out, 2)
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    X_path: Path,
    Y_path: Path,
    batch_size: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    X_samples.npy, Y_samples.npy 로부터
    Train / Val / Test DataLoader 를 생성한다.
    """

    X = np.load(X_path)  # (S, N, T_in, F)
    Y = np.load(Y_path)  # (S, N, T_out, 2)

    S, N, T_in, F = X.shape
    S2, N2, T_out, num_targets = Y.shape
    assert S == S2 and N == N2, "X, Y의 샘플 개수 / 노드 수가 다름"
    assert num_targets == 2, "Y 마지막 축은 2 (TotalTraffic, Speed) 이어야 함"

    print(f"[Data] X: {X.shape}, Y: {Y.shape}")
    print(f"[Data] N={N}, T_in={T_in}, F={F}, T_out={T_out}, num_targets={num_targets}")

    dataset = TrafficSamplesDataset(X, Y)

    # ---------- 인덱스 셔플 후 Train/Val/Test 나누기 ----------
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
# 2. No-Graph Multi-task Baseline 모델들
# =========================

class MLPBaselineMultiTask(nn.Module):
    """
    그래프 정보를 전혀 사용하지 않는 Multi-task Baseline (MLP).
    
    - 입력: x (B, N, T_in, F)
    - 내부: 각 노드 시퀀스 (T_in * F) 를 벡터로 펼쳐서
            공유 MLP 에 넣는다 (노드별 독립 처리, 파라미터 공유).
    - 출력: y_hat (B, N, T_out, 2)
            → [TotalTraffic(t+1), Speed(t+1)]
    """
    def __init__(
        self,
        T_in: int,
        F: int,
        T_out: int = 1,
        num_targets: int = 2,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T_in = T_in
        self.F = F
        self.T_out = T_out
        self.num_targets = num_targets

        in_dim = T_in * F
        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # 출력층: T_out * num_targets (예: 1 * 2 = 2)
        out_dim = T_out * num_targets
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F)
        return: (B, N, T_out, num_targets)
        """
        B, N, T_in, F = x.shape
        assert T_in == self.T_in and F == self.F

        # (B, N, T_in, F) -> (B * N, T_in * F)
        x_flat = x.view(B * N, T_in * F)

        # (B*N, T_out*num_targets)
        y_flat = self.mlp(x_flat)

        # (B, N, T_out, num_targets)
        y = y_flat.view(B, N, self.T_out, self.num_targets)
        return y


class LSTMBaselineMultiTask(nn.Module):
    """
    그래프 정보를 전혀 쓰지 않는 Multi-task LSTM Baseline.
    
    - 입력: x (B, N, T_in, F)
    - 내부: (B * N, T_in, F) 시퀀스로 변환 후
            공유 LSTM 적용 → 마지막 타임스텝 hidden 사용
    - 출력: y_hat (B, N, T_out, 2)
    """
    def __init__(
        self,
        F: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        T_out: int = 1,
        num_targets: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.F = F
        self.T_out = T_out
        self.num_targets = num_targets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 입력 (B*N, T_in, F)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # 마지막 hidden → T_out * num_targets 로 매핑
        self.fc = nn.Linear(lstm_out_dim, T_out * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F)
        return: (B, N, T_out, num_targets)
        """
        B, N, T_in, F = x.shape
        assert F == self.F

        # (B, N, T_in, F) -> (B*N, T_in, F)
        x_seq = x.view(B * N, T_in, F)

        out, (h_n, c_n) = self.lstm(x_seq)
        # out: (B*N, T_in, hidden)
        # 마지막 타임스텝의 출력 사용
        h_last = out[:, -1, :]  # (B*N, hidden*dir)

        y_flat = self.fc(h_last)  # (B*N, T_out*num_targets)
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
    """
    한 epoch 동안 train_loader 에 대해 학습하고,
    평균 loss / MAE(all) / MAE(traffic) / MAE(speed) 를 반환.
    """
    model.train()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_traffic = 0.0
    total_mae_speed = 0.0
    total_count = 0

    for xb, yb in loader:
        # xb: (B, N, T_in, F)
        # yb: (B, N, T_out, 2)
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        y_hat = model(xb)  # (B, N, T_out, 2)
        loss = criterion(y_hat, yb)

        loss.backward()
        optimizer.step()

        B = xb.size(0)
        total_loss += loss.item() * B

        diff = (y_hat - yb).abs()  # (B, N, T_out, 2)
        mae_all = diff.mean().item()
        mae_traffic = diff[..., 0].mean().item()
        mae_speed   = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_traffic += mae_traffic * B
        total_mae_speed   += mae_speed * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae_all = total_mae_all / total_count
    avg_mae_traffic = total_mae_traffic / total_count
    avg_mae_speed = total_mae_speed / total_count
    return avg_loss, avg_mae_all, avg_mae_traffic, avg_mae_speed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float, float]:
    """
    Val/Test 용 평가 함수.
    평균 loss / MAE(all) / MAE(traffic) / MAE(speed) 반환.
    """
    model.eval()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_traffic = 0.0
    total_mae_speed = 0.0
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
        mae_traffic = diff[..., 0].mean().item()
        mae_speed   = diff[..., 1].mean().item()

        total_mae_all += mae_all * B
        total_mae_traffic += mae_traffic * B
        total_mae_speed   += mae_speed * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae_all = total_mae_all / total_count
    avg_mae_traffic = total_mae_traffic / total_count
    avg_mae_speed = total_mae_speed / total_count
    return avg_loss, avg_mae_all, avg_mae_traffic, avg_mae_speed


# =========================
# 4. 메인: MLP / LSTM 중 하나 골라서 학습
# =========================

def main(model_type: str = "mlp"):
    # 1) DataLoader 준비
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path, Y_path,
        batch_size=2,   # N이 커서 batch_size는 작게
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

    # 2) 모델 선택
    if model_type == "mlp":
        model = MLPBaselineMultiTask(
            T_in=T_in,
            F=F,
            T_out=T_out,
            num_targets=num_targets,
            hidden_dims=[64, 64],
            dropout=0.1,
        )
    elif model_type == "lstm":
        model = LSTMBaselineMultiTask(
            F=F,
            hidden_size=64,
            num_layers=1,
            T_out=T_out,
            num_targets=num_targets,
            dropout=0.0,
            bidirectional=False,
        )
    else:
        raise ValueError("model_type 은 'mlp' 또는 'lstm' 이어야 합니다.")

    model = model.to(device)
    print(model)

    # 3) 학습
    criterion = nn.MSELoss()  # 모든 타깃(traffic, speed)에 대해 평균 MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

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

    # 4) best 모델로 Test 평가
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion
    )
    print(
        f"[Test] Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, MAE(speed): {test_mae_sp:.4f}"
    )

    # 5) 모델 저장
    save_path = data_dir / f"baseline_{model_type}_multitask_best.pth"
    torch.save(
        {
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        save_path,
    )
    print("Saved best model to:", save_path)


if __name__ == "__main__":
    # main(model_type="mlp")
    main(model_type="lstm")
