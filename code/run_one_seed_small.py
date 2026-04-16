"""
run_one_seed_small.py — Run ONE SEED with smaller d_model to reduce memory usage
Usage: python run_one_seed_small.py <dropout> <seed_index>
"""
import os, sys, json, warnings
import numpy as np, torch, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
warnings.filterwarnings("ignore")

from layer4_mamba_moe import (
    OilMoE, set_seed, oil_price_data_preprocess_v6,
    train_moe_model, evaluate_moe_model,
)

FILE_PATH = r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv"
FEAT_COLS = ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
             "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
             "Excavator","Excavator_YoY","M2_M1_Spread"]
TARGET_COL = "Daqing"
SEQ_LEN = 52
SEEDS = [42, 123, 2024]
N_EXPERTS = 3
N_REGIMES = 3
D_MODEL = 16  # REDUCED from 32 -> 16 to save memory

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_one_seed_small.py <dropout> <seed_index 0-2>")
        print(f"  SEEDS = {SEEDS}")
        sys.exit(1)
    
    dropout = float(sys.argv[1])
    seed_idx = int(sys.argv[2])
    seed = SEEDS[seed_idx]
    
    print("="*70)
    print(f"  Run ONE SEED (SMALL): dropout={dropout}, seed={seed} (idx={seed_idx}), d_model={D_MODEL}")
    print("="*70)
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data preprocessing
    (train_loader, val_loader, test_loader,
     scaler_feat, scaler_target, scaler_gate,
     all_feat_cols, gate_dim,
     regime_detector, regime_proba_aligned, regime_labels_aligned,
     base_p, true_p, dates) = oil_price_data_preprocess_v6(
        file_path=FILE_PATH, feat_cols=FEAT_COLS,
        target_col=TARGET_COL, seq_len=SEQ_LEN,
        n_regimes=N_REGIMES, gate_window=4, vol_window=12, hmm_mode="multi")
    
    n_features = len(all_feat_cols)
    
    # Build model with smaller d_model
    model = OilMoE(
        n_features=n_features, gate_input_dim=gate_dim,
        n_experts=N_EXPERTS, d_model=D_MODEL, pred_len=1,
        gate_hidden_dim=16,
        expert_configs=[{
            "input_dim": n_features, "d_model": D_MODEL,
            "d_state": 16, "n_layers": 1, "dropout": dropout
        } for _ in range(N_EXPERTS)])
    
    model = model.to(device)
    print(f"  [Model] Params: {sum(p.numel() for p in model.parameters())}")
    
    model, history = train_moe_model(model, train_loader, val_loader,
        epochs=100, lr=1e-4, balance_weight=0.01, patience=15,
        device=device, loss_alpha=1.0, loss_beta=0.01)
    
    _, _, _, _, _, metrics = evaluate_moe_model(
        model, test_loader, scaler_target, base_p, true_p, device=device)
    
    res = {
        "dropout": dropout,
        "seed": seed,
        "dir_acc": metrics["dir_acc"],
        "mae": metrics["mae_model"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape_model"],
        "return_corr": metrics["r_corr"],
        "vs_rw_pct": metrics["pct_vs_rw"],
    }
    
    print(f"\n  RESULT: dropout={dropout}, seed={seed}")
    print(f"    DA={res['dir_acc']:.1f}%  MAE={res['mae']:.4f}")
    
    # Append to CSV
    out_file = os.path.join(SCRIPT_DIR, "..", "output", f"dropout_{dropout}.csv")
    df_new = pd.DataFrame([res])
    
    if os.path.exists(out_file):
        df_existing = pd.read_csv(out_file)
        # Check if this (dropout, seed) already exists
        existing = df_existing[(df_existing["dropout"] == dropout) & 
                               (df_existing["seed"] == seed)]
        if len(existing) > 0:
            # Overwrite
            df_existing = df_existing[~((df_existing["dropout"] == dropout) & 
                                       (df_existing["seed"] == seed))]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(out_file, index=False)
    print(f"  [Saved] to {out_file}, total rows: {len(df_combined)}")
    
    # Cleanup
    del model, train_loader, val_loader, test_loader
    del regime_detector, regime_proba_aligned, regime_labels_aligned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("\n  Done. Exiting.")

if __name__ == "__main__":
    main()
