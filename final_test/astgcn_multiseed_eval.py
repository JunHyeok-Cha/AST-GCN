# astgcn_multiseed_eval.py
import numpy as np
import torch
from torch import nn
import sys
from pathlib import Path
from copy import deepcopy

# astgcn_experiment.py가 있는 폴더
PROJECT_DIR = Path("/mnt/c/새 폴더")
sys.path.append(str(PROJECT_DIR))

from astgcn_experiment import (
    get_dataloaders_ast,
    ASTGCNMultiTaskEdge,   # ✅ 최신: Edge-att + sparse adaptive 모델
    train_one_epoch,
    evaluate,
    A_path,
    device,
)

data_dir = Path("/mnt/c/새 폴더/res")


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_astgcn_multi_seed(
    ckpt_path: Path,
    use_periodic: bool,
    seeds=(0, 1, 2, 3, 4),
    num_epochs: int = 80,
    patience: int = 10,
    split_seed_mode: str = "per_seed",   # "fixed" or "per_seed"
    fixed_split_seed: int = 42,
):
    """
    split_seed_mode:
      - "fixed"    : 데이터 split은 fixed_split_seed로 고정, 모델 초기화만 seed별로 바뀜
      - "per_seed" : 데이터 split도 seed별로 바뀜(더 빡센 multi-seed)
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_params = ckpt.get("best_params", None)
    if best_params is None:
        raise KeyError(f"Checkpoint에 best_params가 없음: {ckpt_path}")

    print("Loaded best_params:", best_params)

    # adjacency는 한 번만 로드 (매 seed마다 다시 안 읽게)
    A_fixed_np = np.load(A_path).astype(np.float32)

    def pick(name, default):
        if name in best_params:
            return best_params[name]
        # 혹시 ckpt["config"]에 들어있는 경우도 대비
        cfg = ckpt.get("config", {})
        if isinstance(cfg, dict) and (name in cfg):
            return cfg[name]
        return default

    # ✅ optuna 튜닝값(없으면 기본값 fallback)
    hidden_channels = pick("hidden_channels", 32)
    num_blocks      = pick("num_blocks", 1)
    dropout         = pick("dropout", 0.1)
    att_dim         = pick("att_dim", 32)

    lr              = pick("lr", 5e-4)
    weight_decay    = pick("weight_decay", 1e-5)
    batch_size      = pick("batch_size", 2)

    # ✅ adaptive 관련(없으면 기본값)
    adj_emb_dim             = pick("adj_emb_dim", 16)
    adaptive_warmup_epochs  = pick("adaptive_warmup_epochs", 10)
    beta_final              = pick("beta_final", 0.3)
    beta_ramp_epochs        = pick("beta_ramp_epochs", 10)

    results = []

    for seed in seeds:
        print(f"\n===== AST-GCN EDGE (use_periodic={use_periodic}) | seed={seed} =====")
        set_global_seed(seed)

        # split seed 결정
        if split_seed_mode == "fixed":
            split_seed = fixed_split_seed
        elif split_seed_mode == "per_seed":
            split_seed = seed
        else:
            raise ValueError("split_seed_mode must be 'fixed' or 'per_seed'")

        # 데이터 로더
        train_loader, val_loader, test_loader, meta = get_dataloaders_ast(
            use_periodic=use_periodic,
            batch_size=batch_size,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=split_seed,
        )

        assert A_fixed_np.shape == (meta["N"], meta["N"])

        # ✅ 최신 모델 생성
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
            adaptive_warmup_epochs=adaptive_warmup_epochs,
            beta_final=beta_final,
            beta_ramp_epochs=beta_ramp_epochs,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        bad = 0

        for epoch in range(1, num_epochs + 1):
            # ✅ adaptive beta 스케줄 반영
            model.set_epoch(epoch)

            train_loss, train_mae_all, train_mae_tr, train_mae_sp = train_one_epoch(
                model, train_loader, optimizer, criterion, use_periodic
            )
            val_loss, val_mae_all, val_mae_tr, val_mae_sp = evaluate(
                model, val_loader, criterion, use_periodic
            )

            print(
                f"[seed={seed}][Epoch {epoch:03d}] beta={getattr(model, 'cur_beta', 0.0):.3f} | "
                f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
                f"ValMAE(all) {val_mae_all:.4f} (tr {val_mae_tr:.4f}, sp {val_mae_sp:.4f})"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                bad = 0
            # else:
            #     bad += 1
            #     if bad >= patience:
            #         print(f"[seed={seed}] Early stopping at epoch {epoch}")
            #         break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_mae_all, test_mae_tr, test_mae_sp = evaluate(
            model, test_loader, criterion, use_periodic
        )
        print(
            f"[seed={seed}] TEST | Loss {test_loss:.4f} | "
            f"MAE(all) {test_mae_all:.4f} | MAE(tr) {test_mae_tr:.4f} | MAE(sp) {test_mae_sp:.4f}"
        )

        results.append(
            dict(seed=seed, loss=test_loss, mae_all=test_mae_all, mae_tr=test_mae_tr, mae_sp=test_mae_sp)
        )

    # ===== summary =====
    loss_arr    = np.array([r["loss"] for r in results], dtype=np.float64)
    mae_all_arr = np.array([r["mae_all"] for r in results], dtype=np.float64)
    mae_tr_arr  = np.array([r["mae_tr"] for r in results], dtype=np.float64)
    mae_sp_arr  = np.array([r["mae_sp"] for r in results], dtype=np.float64)

    def summarize(name, arr):
        print(f"{name}: mean={arr.mean():.4f}, std={arr.std(ddof=1):.4f}, values={np.round(arr, 4)}")

    print("\n===== Summary over seeds =====")
    print("seeds:", list(seeds), "| split_seed_mode:", split_seed_mode, "| fixed_split_seed:", fixed_split_seed)
    summarize("Test Loss    ", loss_arr)
    summarize("MAE(all)     ", mae_all_arr)
    summarize("MAE(traffic) ", mae_tr_arr)
    summarize("MAE(speed)   ", mae_sp_arr)

    return dict(
        seeds=list(seeds),
        results=results,
        loss=loss_arr,
        mae_all=mae_all_arr,
        mae_tr=mae_tr_arr,
        mae_sp=mae_sp_arr,
    )


if __name__ == "__main__":
    full_ckpt = data_dir / "astgcn_multitask_full_optuna_edgeatt_adaptive_best.pth"

    run_astgcn_multi_seed(
        ckpt_path=full_ckpt,
        use_periodic=True,
        seeds=(0, 1, 2, 3, 4),
        num_epochs=80,
        patience=10,
        split_seed_mode="per_seed",  # split도 같이 흔들려면 per_seed
        fixed_split_seed=42,
    )
