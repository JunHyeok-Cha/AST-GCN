# baseline_optuna.py
#
# Optuna 를 이용해서 No-Graph Baseline (MLP / LSTM)을
# 자동으로 하이퍼파라미터 튜닝하는 스크립트.
#
# - baseline_experiment.py 에 정의된 것들을 import 해서 사용:
#   - get_dataloaders
#   - MLPBaselineMultiTask
#   - LSTMBaselineMultiTask
#   - train_one_epoch
#   - evaluate
#   - device, data_dir, X_path, Y_path
#
# - objective_mlp / objective_lstm 두 개의 objective 정의
# - 각 모델에 대해 study.optimize(...)로 best 하이퍼파라미터 찾기
# - best params 로 다시 학습해서 Test 성능 측정 + 모델 저장

from pathlib import Path
from typing import Dict

import optuna
import torch
from torch import nn

# 👇 같은 폴더에 있는 baseline_experiment.py import
from baseline_experiment import (
    get_dataloaders,
    MLPBaselineMultiTask,
    LSTMBaselineMultiTask,
    train_one_epoch,
    evaluate,
    device,
    data_dir,
    X_path,
    Y_path,
)


# =========================================
# 1. MLP용 Objective
# =========================================

def objective_mlp(trial: optuna.trial.Trial) -> float:
    """
    Optuna가 한 번의 trial마다 호출하는 objective 함수 (MLP 버전).
    - 하이퍼파라미터를 샘플링하고
    - 몇 epoch 학습 + 검증
    - best validation loss 를 반환
    """

    # ----- 1) 하이퍼파라미터 샘플링 -----
    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [32, 64, 128])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [32, 64, 128])
    num_layers  = trial.suggest_int("num_layers", 1, 2)
    dropout     = trial.suggest_float("dropout", 0.0, 0.3)

    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [2, 4])

    # hidden_dims 리스트 구성
    if num_layers == 1:
        hidden_dims = [hidden_dim1]
    else:
        hidden_dims = [hidden_dim1, hidden_dim2]

    # ----- 2) 데이터 로더 준비 -----
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    # ----- 3) 모델 생성 -----
    model = MLPBaselineMultiTask(
        T_in=T_in,
        F=F,
        T_out=T_out,
        num_targets=num_targets,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ----- 4) 학습 루프 (validation 기준) -----
    num_epochs = 40       # 탐색용: 40 epoch 정도
    best_val_loss = float("inf")
    patience = 8
    patience_cnt = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        # 진행 상황 로그 (선택)
        print(
            f"[MLP][Trial {trial.number}][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # optuna pruning 에 보고
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # simple early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    return best_val_loss


def train_best_mlp(best_params: Dict):
    """
    Optuna에서 찾은 best_params 로
    MLPBaselineMultiTask 를 다시 학습하고
    Test 성능을 측정 + 모델 저장.
    """

    hidden_dim1 = best_params["hidden_dim1"]
    hidden_dim2 = best_params.get("hidden_dim2", 64)
    num_layers  = best_params["num_layers"]
    dropout     = best_params["dropout"]

    lr           = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    batch_size   = best_params["batch_size"]

    if num_layers == 1:
        hidden_dims = [hidden_dim1]
    else:
        hidden_dims = [hidden_dim1, hidden_dim2]

    # 데이터 로더 (동일 split 사용)
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    model = MLPBaselineMultiTask(
        T_in=T_in,
        F=F,
        T_out=T_out,
        num_targets=num_targets,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_epochs = 80
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
            f"[MLP-BEST][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
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
        f"[MLP-BEST] Test Loss: {test_loss:.4f}, "
        f"MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, "
        f"MAE(speed): {test_mae_sp:.4f}"
    )

    save_path = data_dir / "baseline_mlp_multitask_optuna_best.pth"
    torch.save(
        {
            "model_type": "mlp_multitask_optuna",
            "state_dict": model.state_dict(),
            "meta": meta,
            "best_params": best_params,
        },
        save_path,
    )
    print("Saved tuned MLP model to:", save_path)


# =========================================
# 2. LSTM용 Objective
# =========================================

def objective_lstm(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective (LSTM 버전).
    """

    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers  = trial.suggest_int("num_layers", 1, 2)
    dropout     = trial.suggest_float("dropout", 0.0, 0.3)
    bidirectional = trial.suggest_categorical("bidirectional", [False, True])

    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [2, 4])

    # ----- 데이터 로더 -----
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    # ----- 모델 생성 -----
    model = LSTMBaselineMultiTask(
        F=F,
        hidden_size=hidden_size,
        num_layers=num_layers,
        T_out=T_out,
        num_targets=num_targets,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_epochs = 40
    best_val_loss = float("inf")
    patience = 8
    patience_cnt = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        print(
            f"[LSTM][Trial {trial.number}][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    return best_val_loss


def train_best_lstm(best_params: Dict):
    """
    Optuna best_params 로 LSTMBaselineMultiTask 를 다시 학습 + Test 평가 + 저장.
    """

    hidden_size  = best_params["hidden_size"]
    num_layers   = best_params["num_layers"]
    dropout      = best_params["dropout"]
    bidirectional = best_params["bidirectional"]

    lr           = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    batch_size   = best_params["batch_size"]

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    model = LSTMBaselineMultiTask(
        F=F,
        hidden_size=hidden_size,
        num_layers=num_layers,
        T_out=T_out,
        num_targets=num_targets,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_epochs = 80
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
            f"[LSTM-BEST][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
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
        f"[LSTM-BEST] Test Loss: {test_loss:.4f}, "
        f"MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, "
        f"MAE(speed): {test_mae_sp:.4f}"
    )

    save_path = data_dir / "baseline_lstm_multitask_optuna_best.pth"
    torch.save(
        {
            "model_type": "lstm_multitask_optuna",
            "state_dict": model.state_dict(),
            "meta": meta,
            "best_params": best_params,
        },
        save_path,
    )
    print("Saved tuned LSTM model to:", save_path)


# =========================================
# 3. 실행 진입점
# =========================================

if __name__ == "__main__":
    # --------- 1) MLP 튜닝 ---------
    print("===== Optuna for MLP Baseline (Multi-task) =====")
    study_mlp = optuna.create_study(
        direction="minimize",
        study_name="baseline_mlp_multitask",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study_mlp.optimize(objective_mlp, n_trials=20)

    print("=== MLP Study Finished ===")
    print("Best val_loss:", study_mlp.best_trial.value)
    print("Best params :", study_mlp.best_trial.params)

    # best params 로 최종 모델 학습 + 평가
    train_best_mlp(study_mlp.best_trial.params)

    # --------- 2) LSTM 튜닝 ---------
    print("\n===== Optuna for LSTM Baseline (Multi-task) =====")
    study_lstm = optuna.create_study(
        direction="minimize",
        study_name="baseline_lstm_multitask",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study_lstm.optimize(objective_lstm, n_trials=20)

    print("=== LSTM Study Finished ===")
    print("Best val_loss:", study_lstm.best_trial.value)
    print("Best params :", study_lstm.best_trial.params)

    train_best_lstm(study_lstm.best_trial.params)
