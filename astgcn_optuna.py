# astgcn_optuna.py
#
# Optuna 를 이용해 AST-GCN(full) 하이퍼파라미터 자동 튜닝.
# - 방향: validation loss 최소화
# - 튜닝 대상:
#     hidden_channels, num_blocks, dropout, att_dim,
#     lr, weight_decay, batch_size
#
# 전제:
#   astgcn_experiment.py 에 다음 심볼들이 정의되어 있어야 한다:
#     - get_dataloaders_ast
#     - ASTGCNMultiTask
#     - train_one_epoch
#     - evaluate
#     - A_path
#     - device

from pathlib import Path
import numpy as np
import torch
from torch import nn
import optuna

# 👇 네가 만든 astgcn_experiment.py에서 가져온다고 가정
from astgcn_experiment import (
    get_dataloaders_ast,
    ASTGCNMultiTask,
    train_one_epoch,
    evaluate,
    A_path,
    device,
)

data_dir = Path("/mnt/c/Source/python/AST-GCN/res")


def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna가 한 trial마다 호출하는 objective 함수.
    여기서:
      1) 하이퍼파라미터 샘플링
      2) 모델/옵티마이저 생성
      3) 몇 epoch 학습 + 검증
      4) "최종 val_loss" 반환
    """

    # ---- 1. 하이퍼파라미터 샘플링 ----
    use_periodic = True  # full AST-GCN 버전 기준으로 튜닝

    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    num_blocks      = trial.suggest_int("num_blocks", 1, 2)
    dropout         = trial.suggest_float("dropout", 0.0, 0.3)
    att_dim         = trial.suggest_categorical("att_dim", [16, 32, 64])

    lr             = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay   = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size     = trial.suggest_categorical("batch_size", [2, 4, 8])

    # ---- 2. DataLoader 준비 (batch_size도 튜닝 대상) ----
    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,  # 고정해서 trial 간 비교 가능하게
    )

    # ---- 3. 모델 생성 ----
    A_norm = np.load(A_path)
    assert A_norm.shape == (meta["N"], meta["N"])

    model = ASTGCNMultiTask(
        meta=meta,
        A_norm=A_norm,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        kernel_size=3,
        dropout=dropout,
        att_dim=att_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---- 4. 학습 루프 (Epoch 수는 살짝 줄여서 탐색 속도 확보) ----
    num_epochs = 40          # 튜닝용은 40 정도로 (최종 실험은 이보다 늘려도 됨)
    best_val_loss = float("inf")
    patience = 8             # early stopping
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        print(
            f"[ASTGCN][Trial {trial.number}][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE(all): {val_mae_all:.4f}"
        )
        
        # Optuna에 현재 결과 보고 (pruning 용)
        trial.report(val_loss, step=epoch)

        # 성능이 너무 안 좋으면 early prune
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # best 갱신 + 단순 early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # objective 는 "최종 best validation loss"를 반환
    return best_val_loss


def train_best_model(best_params: dict, use_periodic: bool = True):
    """
    Optuna가 찾아준 best_params 로
    train+val 전체를 다시 학습하고 test 성능 재측정하는 헬퍼.
    (선택적으로 사용)
    """
    # 여기서는 간단하게 train/val/test split 그대로 사용해도 되고,
    # 진지하게 가려면 train+val 재합쳐서 다시 split 해도 됨.
    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=best_params["batch_size"],
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    A_norm = np.load(A_path)
    assert A_norm.shape == (meta["N"], meta["N"])

    model = ASTGCNMultiTask(
        meta=meta,
        A_norm=A_norm,
        hidden_channels=best_params["hidden_channels"],
        num_blocks=best_params["num_blocks"],
        kernel_size=3,
        dropout=best_params["dropout"],
        att_dim=best_params["att_dim"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    # 최종 학습에서는 epoch 수를 좀 늘려도 됨 (예: 80~100)
    num_epochs = 80
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
        model, test_loader, criterion, use_periodic
    )
    print(
        "[BEST PARAMS MODEL] "
        f"Test Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(tr): {test_mae_tr:.4f}, MAE(sp): {test_mae_sp:.4f}"
    )

    # 저장
    tag = "full_optuna"
    save_path = data_dir / f"astgcn_multitask_{tag}_best.pth"
    torch.save(
        {
            "model_type": "astgcn_multitask_optuna",
            "use_periodic": use_periodic,
            "state_dict": model.state_dict(),
            "meta": meta,
            "best_params": best_params,
        },
        save_path,
    )
    print("Saved best tuned model to:", save_path)


if __name__ == "__main__":
    # ---- 1) Optuna study 생성 & 최적화 ----
    study = optuna.create_study(
        direction="minimize",
        study_name="astgcn_full_periodic",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # 실험 횟수는 상황 보고 20~50 정도로 조절
    study.optimize(objective, n_trials=30)

    print("=== Optuna Finished ===")
    print("Best trial number:", study.best_trial.number)
    print("Best val_loss   :", study.best_trial.value)
    print("Best params     :", study.best_trial.params)

    # ---- 2) best params 로 최종 모델 학습 + 테스트 ----
    train_best_model(study.best_trial.params, use_periodic=True)
