# stgcn_multiseed_eval.py
#
# 목적:
#   - Optuna best params로 모델/학습 설정 고정
#   - seed만 바꿔가며 (split + init) 반복학습
#   - Test 성능 mean ± std 출력
#
# 전제:
#   - data_dir / "stgcn_multitask_optuna_best.pth" 존재
#   - stgcn_experiment.py 에:
#       get_dataloaders, STGCNMultiTask, train_one_epoch, evaluate,
#       device, data_dir, X_path, Y_path, A_path
#
from typing import Dict, List
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

# stgcn_experiment.py가 있는 폴더
PROJECT_DIR = Path("/mnt/c/새 폴더")
sys.path.append(str(PROJECT_DIR))

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


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # (선택) 재현성 더 올리기(속도 조금 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_seed(
    seed: int,
    best_params: Dict,
    max_epochs: int = 80,
    patience: int = 10,
    min_delta: float = 1e-4,
):
    print(f"\n========== [ST-GCN] Seed = {seed} ==========")
    set_global_seed(seed)

    # ----- optuna best params (없으면 기본값 fallback) -----
    hidden_channels = best_params["hidden_channels"]
    num_blocks      = best_params["num_blocks"]
    dropout         = best_params["dropout"]
    lr              = best_params["lr"]
    weight_decay    = best_params["weight_decay"]
    batch_size      = best_params["batch_size"]

    # adaptive 관련 (너 optuna에 넣은 값들)
    adj_emb_dim = best_params.get("adj_emb_dim", 16)
    gcn_beta    = best_params.get("gcn_beta", 0.3)
    warmup_ep   = best_params.get("adaptive_warmup_epochs", 10)

    # ----- 데이터 split도 seed별로 변경 -----
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        X_path=X_path,
        Y_path=Y_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    N = meta["N"]
    T_in = meta["T_in"]
    F = meta["F"]
    T_out = meta["T_out"]
    num_targets = meta["num_targets"]

    # ----- adjacency 로드 -----
    A_norm = np.load(A_path)
    assert A_norm.shape == (N, N), "A_norm shape mismatch"

    # ----- 모델 생성 (adaptiveA 포함) -----
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ----- 학습 + early stopping -----
    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(1, max_epochs + 1):
        # ✅ warmup + beta ramp 적용 필수
        model.set_epoch(epoch)

        train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
            model, val_loader, criterion
        )

        print(
            f"[Seed {seed}][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, MAE(all): {train_mae_all:.4f}, "
            f"MAE(tr): {train_mae_tr:.4f}, MAE(sp): {train_mae_sp:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE(all): {val_mae_all:.4f}, "
            f"MAE(tr): {val_mae_tr:.4f}, MAE(sp): {val_mae_sp:.4f}"
        )

        # ✅ warmup 동안엔 early stop 너무 빨리 걸리면 안 좋음 (optuna랑 동일한 철학)
        if epoch <= warmup_ep:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        # early stopping
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[Seed {seed}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(model, test_loader, criterion)
    print(
        f"[Seed {seed}] TEST -- "
        f"Loss: {test_loss:.4f}, "
        f"MAE(all): {test_mae_all:.4f}, "
        f"MAE(traffic): {test_mae_tr:.4f}, "
        f"MAE(speed): {test_mae_sp:.4f}"
    )

    # seed 끝날 때 GPU 메모리 정리(선택)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": float(test_loss),
        "mae_all": float(test_mae_all),
        "mae_traffic": float(test_mae_tr),
        "mae_speed": float(test_mae_sp),
    }


def summarize_results(seeds: List[int], results: List[Dict[str, float]]):
    losses  = np.array([r["loss"] for r in results], dtype=np.float32)
    mae_all = np.array([r["mae_all"] for r in results], dtype=np.float32)
    mae_tr  = np.array([r["mae_traffic"] for r in results], dtype=np.float32)
    mae_sp  = np.array([r["mae_speed"] for r in results], dtype=np.float32)

    def m_s(x):
        return float(x.mean()), float(x.std(ddof=0))

    loss_m, loss_s = m_s(losses)
    all_m,  all_s  = m_s(mae_all)
    tr_m,   tr_s   = m_s(mae_tr)
    sp_m,   sp_s   = m_s(mae_sp)

    print("\n================= ST-GCN Multi-seed Summary =================")
    print("Seeds:", seeds)
    print(f"Loss        : mean={loss_m:.4f}, std={loss_s:.4f}")
    print(f"MAE(all)    : mean={all_m:.4f}, std={all_s:.4f}")
    print(f"MAE(traffic): mean={tr_m:.4f}, std={tr_s:.4f}")
    print(f"MAE(speed)  : mean={sp_m:.4f}, std={sp_s:.4f}")
    print("=============================================================\n")


if __name__ == "__main__":
    ckpt_path = data_dir / "stgcn_multitask_optuna_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    best_params = ckpt.get("best_params", None)
    if best_params is None:
        raise RuntimeError("Checkpoint에 best_params가 없습니다. stgcn_optuna.py 저장 형식 확인 필요")

    print("Loaded best_params from:", ckpt_path)
    print("best_params:", best_params)

    seeds = [0, 1, 2, 3, 4]

    results = []
    for s in seeds:
        res = run_single_seed(seed=s, best_params=best_params, max_epochs=80, patience=10)
        results.append(res)

    summarize_results(seeds, results)
