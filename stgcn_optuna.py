# stgcn_optuna.py
#
# Optuna 를 이용해서 ST-GCN 멀티태스크 모델의
# 하이퍼파라미터(hidden_channels, num_blocks, dropout, lr, weight_decay, batch_size)를
# 자동 튜닝하는 스크립트.
#
# 전제:
#   stgcn_experiment.py 에 다음 심볼들이 정의되어 있어야 한다.
#     - get_dataloaders
#     - STGCNMultiTask
#     - train_one_epoch
#     - evaluate
#     - device
#     - data_dir
#     - X_path, Y_path, A_path
#
# 출력:
#   - Optuna best trial 정보 (val_loss, params)
#   - best params 로 다시 학습한 최종 Test 성능
#   - res/ 디렉토리에 stgcn_multitask_optuna_best.pth 저장

from typing import Dict

import numpy as np
import torch
from torch import nn
import optuna

from stgcn_experiment import (
    get_dataloaders,
    STGCNMultiTask,
    train_one_epoch,
    evaluate,
    device,
    data_dir,
    X_path,
    Y_path,
    A_path,
)


# =========================================
# 1. ST-GCN용 Optuna objective
# =========================================

def objective_stgcn(trial: optuna.trial.Trial) -> float:
    """
    Optuna가 한 trial 마다 호출하는 objective 함수(ST-GCN 버전).

    - 하이퍼파라미터를 샘플링하고
    - 일정 epoch 동안 train/val 학습
    - best validation loss 를 반환
    """

    # ----- 1) 하이퍼파라미터 샘플링 -----
    hidden_channels = trial.suggest_categorical(
        "hidden_channels", [16, 32, 64, 128]
    )
    num_blocks = trial.suggest_int("num_blocks", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    adj_emb_dim = trial.suggest_categorical("adj_emb_dim", [8, 16, 32])
    gcn_beta    = trial.suggest_float("gcn_beta", 0.0, 0.7)
    warmup_ep   = trial.suggest_categorical("adaptive_warmup_epochs", [0, 5, 10])

    # ----- 2) 데이터 로더 -----
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,   # 고정해서 trial 간 비교 공정하게
    )

    N = meta["N"]
    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    # ----- 3) 인접행렬 로드 -----
    A_norm = np.load(A_path)
    assert A_norm.shape == (N, N), "A_norm shape이 X의 N과 맞지 않습니다."

    # ----- 4) 모델 생성 -----
    model = STGCNMultiTask(
        N_nodes=N,
        T_in=T_in,
        F_in=F,
        T_out=T_out,
        num_targets=num_targets,
        A_norm=A_norm,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        kernel_size=3,
        dropout=dropout,
        use_residual=True,
        use_adaptive_adj=True,
        adj_emb_dim=adj_emb_dim,
        gcn_beta=gcn_beta,
        adaptive_warmup_epochs=warmup_ep,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ----- 5) 학습 루프 (validation 기준) -----
    num_epochs = 40       # 탐색용 epoch 수
    best_val_loss = float("inf")
    patience = 8
    patience_cnt = 0

    for epoch in range(1, num_epochs + 1):
        model.set_epoch(epoch)
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        print(
            f"[STGCN][Trial {trial.number}][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE(all): {val_mae_all:.4f}"
        )

    # ✅ warmup 동안은 pruning/earlystop 금지
        if epoch > warmup_ep:
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
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return best_val_loss


# =========================================
# 2. best params 로 최종 학습 + Test 평가
# =========================================

def train_best_stgcn(best_params: Dict):
    """
    Optuna 가 찾은 best_params 로 STGCNMultiTask 를
    다시 학습하고 Test 성능을 측정한 뒤, .pth 로 저장.
    """

    hidden_channels = best_params["hidden_channels"]
    num_blocks = best_params["num_blocks"]
    dropout = best_params["dropout"]

    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    batch_size = best_params["batch_size"]

    # ✅ 추가: adaptive 관련 튜닝값 반영
    adj_emb_dim = best_params["adj_emb_dim"]
    gcn_beta    = best_params["gcn_beta"]
    warmup_ep   = best_params["adaptive_warmup_epochs"]

    # 데이터 로더 (동일 split 유지)
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    N = meta["N"]
    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    A_norm = np.load(A_path)
    assert A_norm.shape == (N, N), "A_norm shape이 X의 N과 맞지 않습니다."

    model = STGCNMultiTask(
        N_nodes=N,
        T_in=T_in,
        F_in=F,
        T_out=T_out,
        num_targets=num_targets,
        A_norm=A_norm,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        kernel_size=3,
        dropout=dropout,
        use_residual=True,
        use_adaptive_adj=True,
        adj_emb_dim=adj_emb_dim,
        gcn_beta=gcn_beta,
        adaptive_warmup_epochs=warmup_ep,
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
        model.set_epoch(epoch)
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        print(
            f"[STGCN-BEST][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE(all): {val_mae_all:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # 최종 Test 성능
    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion
    )
    print(
        f"[STGCN-BEST] Test Loss: {test_loss:.4f}, "
        f"MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, "
        f"MAE(speed): {test_mae_sp:.4f}"
    )

    # 저장
    save_path = data_dir / "stgcn_multitask_optuna_best.pth"
    torch.save(
        {
            "model_type": "stgcn_multitask_optuna",
            "state_dict": model.state_dict(),
            "meta": meta,
            "best_params": best_params,
        },
        save_path,
    )
    print("Saved tuned ST-GCN model to:", save_path)


# =========================================
# 3. 실행 진입점
# =========================================

if __name__ == "__main__":
    print("===== Optuna for ST-GCN Multi-task =====")
    study = optuna.create_study(
        direction="minimize",
        study_name="stgcn_multitask",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # 모델 복잡도 고려해서 trial 수는 적당히 (예: 20~30)
    study.optimize(objective_stgcn, n_trials=30)

    print("=== ST-GCN Study Finished ===")
    print("Best val_loss:", study.best_trial.value)
    print("Best params :", study.best_trial.params)

    # best params 로 다시 학습 + Test 평가 + 저장
    train_best_stgcn(study.best_trial.params)
