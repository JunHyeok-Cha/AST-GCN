# baseline_experiment.py
#
# 1) X_samples.npy, Y_samples.npy 로부터 DataLoader 생성
# 2) No-Graph Baseline 모델(MLP / LSTM) 정의
# 3) 학습 루프 실행 및 성능 출력
#
# ⚠ 전제: 전처리 단계에서 이미
#   - T_in, T_out
#   - X_samples.npy: (num_samples, N, T_in, F)
#   - Y_samples.npy: (num_samples, N, T_out)
#   이 저장되어 있다고 가정한다.

import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List


# =========================
# 0. 공통 설정
# =========================

data_dir = Path("/mnt/c/Source/python/AST-GCN/res")

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
    
    X: (num_samples, N, T_in, F)
    Y: (num_samples, N, T_out)
    
    한 "샘플"은 하나의 "하루 시계열 윈도우 (T_in 시간)" 이고,
    그 안에 N개의 노드가 모두 포함되어 있다.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0], "샘플 개수 불일치"
        self.X = torch.from_numpy(X).float()  # (S, N, T_in, F)
        self.Y = torch.from_numpy(Y).float()  # (S, N, T_out)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # 반환 형태: (N, T_in, F), (N, T_out)
        return self.X[idx], self.Y[idx]

def get_dataloaders(
    X_path: Path,
    Y_path: Path,
    batch_size: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    use_speed: bool = True,      # 🔥 speed 쓸지 말지
):
    """
    use_speed = True  → 모든 피처 사용 (TotalTraffic, GetOn, GetOff, RouteCount, Speed)
    use_speed = False → Speed 채널 제거 (마지막 채널만 날림)
    """

    X = np.load(X_path)  # (S, N, T_in, F=5)
    Y = np.load(Y_path)  # (S, N, T_out)

    if use_speed:
        X_used = X                      # (S, N, T_in, 5)
    else:
        X_used = X[..., :-1]            # (S, N, T_in, 4)  ← Speed 제거

    S, N, T_in, F = X_used.shape
    _, _, T_out = Y.shape

    print(f"[Data] use_speed={use_speed}")
    print(f"[Data] X_used: {X_used.shape}, Y: {Y.shape}")

    dataset = TrafficSamplesDataset(X_used, Y)

    # 아래는 그대로
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

    meta = dict(N=N, T_in=T_in, F=F, T_out=T_out)
    return train_loader, val_loader, test_loader, meta


# =========================
# 2. No-Graph Baseline 모델들
# =========================

class MLPBaseline(nn.Module):
    """
    그래프 정보를 전혀 사용하지 않는 Baseline.
    
    - 입력: x (B, N, T_in, F)
    - 내부: 각 노드 시퀀스 (T_in * F) 를 벡터로 펼쳐서
            공유 MLP 에 넣는다 (노드별 독립 처리, 파라미터 공유).
    - 출력: y_hat (B, N, T_out)
    """
    def __init__(
        self,
        T_in: int,
        F: int,
        T_out: int = 1,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        self.T_in = T_in
        self.F = F
        self.T_out = T_out

        in_dim = T_in * F
        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # 마지막 출력층: T_out (보통 1시간 예측)
        layers.append(nn.Linear(prev_dim, T_out))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F)
        return: (B, N, T_out)
        """
        B, N, T_in, F = x.shape
        assert T_in == self.T_in and F == self.F

        # (B, N, T_in, F) -> (B * N, T_in * F)
        x_flat = x.view(B * N, T_in * F)

        # MLP 통과: (B * N, T_out)
        y_flat = self.mlp(x_flat)

        # (B, N, T_out) 형태로 되돌리기
        y = y_flat.view(B, N, self.T_out)
        return y


class LSTMBaseline(nn.Module):
    """
    그래프 정보를 전혀 쓰지 않는 LSTM Baseline.
    
    - 입력: x (B, N, T_in, F)
    - 내부: (B * N, T_in, F) 시퀀스로 변환 후
            공유 LSTM 적용 → 마지막 타임스텝 hidden 사용
    - 출력: y_hat (B, N, T_out)
    """
    def __init__(
        self,
        F: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        T_out: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        self.F = F
        self.T_out = T_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # 입력 (B*N, T_in, F)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(lstm_out_dim, T_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T_in, F)
        return: (B, N, T_out)
        """
        B, N, T_in, F = x.shape
        assert F == self.F

        # (B, N, T_in, F) -> (B*N, T_in, F)
        x_seq = x.view(B * N, T_in, F)

        out, (h_n, c_n) = self.lstm(x_seq)
        # out: (B*N, T_in, hidden)
        # 여기서는 마지막 타임스텝의 output 사용
        h_last = out[:, -1, :]  # (B*N, hidden*dir)

        y_flat = self.fc(h_last)    # (B*N, T_out)
        y = y_flat.view(B, N, self.T_out)
        return y


# =========================
# 3. 학습 / 평가 루프
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    한 epoch 동안 train_loader 에 대해 학습하고,
    평균 loss / MAE 를 반환.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)  # (B, N, T_in, F)
        yb = yb.to(device)  # (B, N, T_out)

        optimizer.zero_grad()

        y_hat = model(xb)   # (B, N, T_out)
        loss = criterion(y_hat, yb)

        loss.backward()
        optimizer.step()

        # 통계 쌓기
        B = xb.size(0)
        total_loss += loss.item() * B

        # MAE 계산
        mae = (y_hat - yb).abs().mean().item()
        total_mae += mae * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    return avg_loss, avg_mae


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Val/Test 용 평가 함수.
    평균 loss / MAE 반환.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = criterion(y_hat, yb)

        B = xb.size(0)
        total_loss += loss.item() * B

        mae = (y_hat - yb).abs().mean().item()
        total_mae += mae * B
        total_count += B

    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    return avg_loss, avg_mae


# =========================
# 4. 메인: MLP / LSTM 중 하나 골라서 학습
# =========================

def main(model_type: str = "mlp", use_speed: bool = True):
    # 1) DataLoader 준비
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path, Y_path,
        batch_size=2,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        use_speed=use_speed,        # 🔥 여기!
    )
    N, T_in, F, T_out = meta["N"], meta["T_in"], meta["F"], meta["T_out"]
    print("Meta:", meta)

    # 2) 모델 선택 (그대로, F만 meta에서 받아서 사용)
    if model_type == "mlp":
        model = MLPBaseline(
            T_in=T_in,
            F=F,
            T_out=T_out,
            hidden_dims=[64, 64],
            dropout=0.1,
        )
    elif model_type == "lstm":
        model = LSTMBaseline(
            F=F,
            hidden_size=64,
            num_layers=1,
            T_out=T_out,
            dropout=0.0,
            bidirectional=False,
        )
    else:
        raise ValueError("model_type 은 'mlp' 또는 'lstm' 이어야 합니다.")

    model = model.to(device)
    print(model)

    # 3) 학습 (그대로)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 100
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = evaluate(model, val_loader, criterion)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae = evaluate(model, test_loader, criterion)
    print(f"[Test] Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

    # 4) 저장할 때 이름으로 구분
    tag = "withspeed" if use_speed else "nospeed"
    save_path = data_dir / f"baseline_{model_type}_{tag}_best.pth"
    torch.save({
        "model_type": model_type,
        "use_speed": use_speed,
        "state_dict": model.state_dict(),
        "meta": meta,
    }, save_path)
    print("Saved best model to:", save_path)



if __name__ == "__main__":
    # 1) MLP, Speed 없음
    main(model_type="mlp", use_speed=False)

    # 2) MLP, Speed 있음
    # main(model_type="mlp", use_speed=True)

    # 3) LSTM, Speed 없음
    # main(model_type="lstm", use_speed=False)

    # 4) LSTM, Speed 있음
    # main(model_type="lstm", use_speed=True)
