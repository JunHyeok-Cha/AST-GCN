# baseline_multiseed_eval.py
#
# 목적:
#   - Optuna로 찾은 best 하이퍼파라미터(MLP / LSTM)를 고정해두고
#   - random seed만 여러 개 바꿔가며 다시 학습
#   - 각 seed별 Test Loss / MAE(all) / MAE(traffic) / MAE(speed) 측정
#   - 마지막에 평균(mean)과 표준편차(std) 출력
#
# 전제:
#   - baseline_experiment.py, baseline_optuna.py 를 먼저 실행해서
#       baseline_mlp_multitask_optuna_best.pth
#       baseline_lstm_multitask_optuna_best.pth
#     가 저장되어 있어야 한다.
#
# 사용:
#   $ python baseline_multiseed_eval.py
#
#   필요하면 seeds 리스트만 바꿔서 돌리면 됨.

from pathlib import Path
import numpy as np
import torch
from torch import nn
import sys
from pathlib import Path

# baseline_experiment.py가 있는 폴더
PROJECT_DIR = Path("/mnt/c/새 폴더")
sys.path.append(str(PROJECT_DIR))

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


def set_global_seed(seed: int):
    """
    파이썬 / NumPy / PyTorch 전역 seed 설정.
    - 가중치 초기화
    - DataLoader shuffle
    등에 영향을 줌.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_mlp_multi_seed(
    ckpt_path: Path,
    seeds = (0, 1, 2, 3, 4),
    num_epochs: int = 80,
):
    """
    Optuna로 튜닝된 MLP baseline 에 대해
    여러 random seed로 학습 반복 → Test 성능 평균/표준편차 계산.
    """
    print("\n==================== MLP Multi-seed Eval ====================\n")

    # 1) ckpt에서 best_params 가져오기
    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_params = ckpt["best_params"]
    print("Loaded best_params (MLP):", best_params)

    results = []

    for seed in seeds:
        print(f"\n----- [MLP] seed = {seed} -----")
        set_global_seed(seed)

        # 2) 데이터 로더 (split은 seed=42로 고정 → 모든 run에서 동일한 train/val/test)
        train_loader, val_loader, test_loader, meta = get_dataloaders(
            X_path=X_path,
            Y_path=Y_path,
            batch_size=best_params["batch_size"],
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        T_in = meta["T_in"]
        F = meta["F"]
        T_out = meta["T_out"]
        num_targets = meta["num_targets"]

        # hidden_dims 구성
        hidden_dim1 = best_params["hidden_dim1"]
        num_layers  = best_params["num_layers"]
        hidden_dims = [hidden_dim1]
        if num_layers == 2:
            hidden_dims.append(best_params["hidden_dim2"])

        dropout = best_params["dropout"]

        # 3) 모델 생성 (seed마다 새로 초기화)
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
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

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
                f"[MLP][seed={seed}][Epoch {epoch:03d}] "
                f"Train Loss: {train_loss:.4f}, MAE(all): {train_mae_all:.4f}, "
                f"MAE(tr): {train_mae_tr:.4f}, MAE(sp): {train_mae_sp:.4f} | "
                f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
                f"MAE(tr): {val_mae_tr:.4f}, MAE(sp): {val_mae_sp:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)

        # 4) 이 seed에서의 Test 성능
        test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
            model, test_loader, criterion
        )
        print(
            f"[MLP][seed={seed}] Test Loss: {test_loss:.4f}, "
            f"MAE(all): {test_mae_all:.4f}, "
            f"MAE(traffic): {test_mae_tr:.4f}, MAE(speed): {test_mae_sp:.4f}"
        )

        results.append(
            dict(
                seed=seed,
                loss=test_loss,
                mae_all=test_mae_all,
                mae_tr=test_mae_tr,
                mae_sp=test_mae_sp,
            )
        )

    # 5) seed들에 대한 평균 / 표준편차
    loss_arr    = np.array([r["loss"]    for r in results])
    mae_all_arr = np.array([r["mae_all"] for r in results])
    mae_tr_arr  = np.array([r["mae_tr"]  for r in results])
    mae_sp_arr  = np.array([r["mae_sp"]  for r in results])

    def summarize(name, arr):
        print(
            f"{name}: mean={arr.mean():.4f}, std={arr.std(ddof=1):.4f}, "
            f"values={np.round(arr, 4)}"
        )

    print("\n===== [MLP] Summary over seeds =====")
    summarize("Test Loss    ", loss_arr)
    summarize("MAE(all)     ", mae_all_arr)
    summarize("MAE(traffic) ", mae_tr_arr)
    summarize("MAE(speed)   ", mae_sp_arr)

    return dict(
        seeds=list(seeds),
        loss=loss_arr,
        mae_all=mae_all_arr,
        mae_tr=mae_tr_arr,
        mae_sp=mae_sp_arr,
    )


def run_lstm_multi_seed(
    ckpt_path: Path,
    seeds = (0, 1, 2, 3, 4),
    num_epochs: int = 80,
):
    """
    Optuna로 튜닝된 LSTM baseline 에 대해
    여러 random seed로 학습 반복 → Test 성능 평균/표준편차 계산.
    """
    print("\n==================== LSTM Multi-seed Eval ====================\n")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_params = ckpt["best_params"]
    print("Loaded best_params (LSTM):", best_params)

    results = []

    for seed in seeds:
        print(f"\n----- [LSTM] seed = {seed} -----")
        set_global_seed(seed)

        train_loader, val_loader, test_loader, meta = get_dataloaders(
            X_path=X_path,
            Y_path=Y_path,
            batch_size=best_params["batch_size"],
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        F = meta["F"]
        T_out = meta["T_out"]
        num_targets = meta["num_targets"]

        hidden_size  = best_params["hidden_size"]
        num_layers   = best_params["num_layers"]
        dropout      = best_params["dropout"]
        bidirectional = best_params["bidirectional"]

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
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

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
                f"[LSTM][seed={seed}][Epoch {epoch:03d}] "
                f"Train Loss: {train_loss:.4f}, MAE(all): {train_mae_all:.4f}, "
                f"MAE(tr): {train_mae_tr:.4f}, MAE(sp): {train_mae_sp:.4f} | "
                f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
                f"MAE(tr): {val_mae_tr:.4f}, MAE(sp): {val_mae_sp:.4f}"
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
            f"[LSTM][seed={seed}] Test Loss: {test_loss:.4f}, "
            f"MAE(all): {test_mae_all:.4f}, "
            f"MAE(traffic): {test_mae_tr:.4f}, MAE(speed): {test_mae_sp:.4f}"
        )

        results.append(
            dict(
                seed=seed,
                loss=test_loss,
                mae_all=test_mae_all,
                mae_tr=test_mae_tr,
                mae_sp=test_mae_sp,
            )
        )

    loss_arr    = np.array([r["loss"]    for r in results])
    mae_all_arr = np.array([r["mae_all"] for r in results])
    mae_tr_arr  = np.array([r["mae_tr"]  for r in results])
    mae_sp_arr  = np.array([r["mae_sp"]  for r in results])

    def summarize(name, arr):
        print(
            f"{name}: mean={arr.mean():.4f}, std={arr.std(ddof=1):.4f}, "
            f"values={np.round(arr, 4)}"
        )

    print("\n===== [LSTM] Summary over seeds =====")
    summarize("Test Loss    ", loss_arr)
    summarize("MAE(all)     ", mae_all_arr)
    summarize("MAE(traffic) ", mae_tr_arr)
    summarize("MAE(speed)   ", mae_sp_arr)

    return dict(
        seeds=list(seeds),
        loss=loss_arr,
        mae_all=mae_all_arr,
        mae_tr=mae_tr_arr,
        mae_sp=mae_sp_arr,
    )


if __name__ == "__main__":
    # ✅ Optuna 결과 파일 경로 (baseline_optuna.py에서 저장한 이름과 맞춰줘야 함)
    mlp_ckpt  = data_dir / "baseline_mlp_multitask_optuna_best.pth"
    lstm_ckpt = data_dir / "baseline_lstm_multitask_optuna_best.pth"

    seeds = (0, 1, 2, 3, 4)

    mlp_results  = run_mlp_multi_seed(mlp_ckpt,  seeds=seeds, num_epochs=80)
    lstm_results = run_lstm_multi_seed(lstm_ckpt, seeds=seeds, num_epochs=80)
