# astgcn_optuna.py
#
# Optuna로 AST-GCN(Edge-level SAtt + Sparse Adaptive A) 하이퍼파라미터 튜닝
# - 목적: validation loss 최소화
# - 최신 astgcn_experiment.py(=ASTGCNMultiTaskEdge)에 맞춤
#
# 튜닝 대상(기본):
#   hidden_channels, num_blocks, dropout, att_dim,
#   lr, weight_decay, batch_size
#
# (추가로 adaptive 관련도 튜닝 가능하게 포함)
#   adj_emb_dim, beta_final, adaptive_warmup_epochs, beta_ramp_epochs
#
# 전제:
#   astgcn_experiment.py 에 다음 심볼이 있어야 함:
#     - get_dataloaders_ast
#     - ASTGCNMultiTaskEdge
#     - train_one_epoch
#     - evaluate
#     - A_path
#     - device
#     - data_dir

from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import optuna

from astgcn_experiment import (
    get_dataloaders_ast,
    ASTGCNMultiTaskEdge,
    train_one_epoch,
    evaluate,
    A_path,
    device,
    data_dir,
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================
# 1) Objective (use_periodic 인자로 받음)
# =========================================

def objective(trial: optuna.trial.Trial, use_periodic: bool) -> float:
    """
    1) 하이퍼파라미터 샘플링
    2) 모델/옵티마이저 생성
    3) 학습 + 검증
    4) best val_loss 반환
    """

    # trial간 비교를 공정하게: split + init seed 고정
    base_seed = 42
    set_global_seed(base_seed)

    # ----- (A) 기본 튜닝 파라미터 -----
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    num_blocks      = trial.suggest_int("num_blocks", 1, 2)
    dropout         = trial.suggest_float("dropout", 0.0, 0.3)
    att_dim         = trial.suggest_categorical("att_dim", [16, 32, 64])

    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [2, 4])

    # ----- (B) adaptive 관련 (선택 튜닝) -----
    # 네 ST-GCN에서 adaptive가 효과 있었으니 기본적으로 켠 상태를 전제.
    # 필요하면 아래 범위를 더 좁혀서 탐색 시간 줄여도 됨.
    adj_emb_dim = trial.suggest_categorical("adj_emb_dim", [8, 16, 32])
    beta_final  = trial.suggest_float("beta_final", 0.0, 0.7)
    warmup_ep   = trial.suggest_categorical("adaptive_warmup_epochs", [0, 5, 10])
    beta_ramp   = trial.suggest_categorical("beta_ramp_epochs", [5, 10, 20])

    # ----- 2) DataLoader (split seed 고정) -----
    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=base_seed,  # trial 간 split 동일
    )

    # ----- 3) adjacency 로드 -----
    A_fixed_np = np.load(A_path).astype(np.float32)
    assert A_fixed_np.shape == (meta["N"], meta["N"])

    # ----- 4) 모델 생성 (최신 ASTGCNMultiTaskEdge) -----
    model = ASTGCNMultiTaskEdge(
        meta=meta,
        A_fixed_np=A_fixed_np,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        kernel_size=3,
        dropout=dropout,
        att_dim=att_dim,
        use_adaptive_adj=True,
        adj_emb_dim=adj_emb_dim,
        adaptive_warmup_epochs=warmup_ep,
        beta_final=beta_final,
        beta_ramp_epochs=beta_ramp,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ----- 5) 학습 루프 -----
    num_epochs = 40
    best_val_loss = float("inf")
    patience = 8
    patience_cnt = 0

    mode_str = "FULL" if use_periodic else "RECENT"

    for epoch in range(1, num_epochs + 1):
        model.set_epoch(epoch)

        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        print(
            f"[ASTGCN-{mode_str}][Trial {trial.number}][Epoch {epoch:03d}] "
            f"beta={getattr(model, 'cur_beta', 0.0):.3f} | "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE(all): {val_mae_all:.4f}"
        )

        # ✅ warmup 동안은 pruning/earlystop을 너무 강하게 걸면 오판함
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
            # warmup 구간에서는 best만 갱신
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return best_val_loss


# =========================================
# 2) Best params로 재학습 + Test 평가 + 저장
# =========================================

def train_best_model(best_params: dict, use_periodic: bool = True):
    base_seed = 42
    set_global_seed(base_seed)

    train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
        use_periodic=use_periodic,
        batch_size=best_params["batch_size"],
        val_ratio=0.15,
        test_ratio=0.15,
        seed=base_seed,
    )

    A_fixed_np = np.load(A_path).astype(np.float32)
    assert A_fixed_np.shape == (meta["N"], meta["N"])

    model = ASTGCNMultiTaskEdge(
        meta=meta,
        A_fixed_np=A_fixed_np,
        hidden_channels=best_params["hidden_channels"],
        num_blocks=best_params["num_blocks"],
        kernel_size=3,
        dropout=best_params["dropout"],
        att_dim=best_params["att_dim"],
        use_adaptive_adj=True,
        adj_emb_dim=best_params["adj_emb_dim"],
        adaptive_warmup_epochs=best_params["adaptive_warmup_epochs"],
        beta_final=best_params["beta_final"],
        beta_ramp_epochs=best_params["beta_ramp_epochs"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    num_epochs = 80
    best_val_loss = float("inf")
    best_state = None

    mode_str = "FULL" if use_periodic else "RECENT"

    for epoch in range(1, num_epochs + 1):
        model.set_epoch(epoch)

        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, use_periodic
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion, use_periodic
        )

        print(
            f"[ASTGCN-{mode_str}-BEST][Epoch {epoch:03d}] "
            f"beta={getattr(model, 'cur_beta', 0.0):.3f} | "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE(all): {val_mae_all:.4f}"
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
        f"[ASTGCN-{mode_str}-BEST] "
        f"Test Loss: {test_loss:.4f}, MAE(all): {test_mae_all:.4f}, "
        f"MAE(tr): {test_mae_tr:.4f}, MAE(sp): {test_mae_sp:.4f}"
    )

    tag = "full_optuna_edgeatt_adaptive" if use_periodic else "recent_optuna_edgeatt_adaptive"
    save_path = data_dir / f"astgcn_multitask_{tag}_best.pth"

    torch.save(
        {
            "model_type": "astgcn_multitask_optuna_edgeatt_adaptive",
            "use_periodic": use_periodic,
            "state_dict": model.state_dict(),
            "meta": meta,
            "best_params": best_params,
            "A_path_used": str(A_path),
            "test_metrics": {
                "loss": float(test_loss),
                "mae_all": float(test_mae_all),
                "mae_traffic": float(test_mae_tr),
                "mae_speed": float(test_mae_sp),
            },
        },
        save_path,
    )
    print(f"Saved best tuned model ({mode_str}) to:", save_path)


# =========================================
# 3) 실행 진입점
# =========================================

if __name__ == "__main__":
    # -------- FULL periodic (use_periodic=True) --------
    print("\n===== Optuna for AST-GCN (FULL PERIODIC, edge-att + sparse adaptive) =====")

    study_full = optuna.create_study(
        direction="minimize",
        study_name="astgcn_full_periodic_edgeatt_adaptive",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study_full.optimize(
        lambda trial: objective(trial, use_periodic=True),
        n_trials=30,
    )

    print("=== Optuna Finished (FULL) ===")
    print("Best trial number:", study_full.best_trial.number)
    print("Best val_loss   :", study_full.best_trial.value)
    print("Best params     :", study_full.best_trial.params)

    train_best_model(study_full.best_trial.params, use_periodic=True)

    # -------- 필요하면 RECENT도 켜서 돌리면 됨 --------
    # print("\n===== Optuna for AST-GCN (RECENT ONLY) =====")
    # study_recent = optuna.create_study(
    #     direction="minimize",
    #     study_name="astgcn_recent_only_edgeatt_adaptive",
    #     pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    # )
    # study_recent.optimize(
    #     lambda trial: objective(trial, use_periodic=False),
    #     n_trials=30,
    # )
    # print("=== Optuna Finished (RECENT) ===")
    # print("Best trial number:", study_recent.best_trial.number)
    # print("Best val_loss   :", study_recent.best_trial.value)
    # print("Best params     :", study_recent.best_trial.params)
    # train_best_model(study_recent.best_trial.params, use_periodic=False)
