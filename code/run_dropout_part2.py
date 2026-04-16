"""
run_dropout_part2.py — continue from checkpoint, run dropout=0.4 only
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
D_MODEL = 32

def run_one_dropout(dropout_rate):
    """Run all seeds for one dropout value, clear GPU after each seed."""
    results = []
    
    for seed in SEEDS:
        print(f"\n  >>> Dropout={dropout_rate}, Seed={seed}")
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
        
        # Build model
        model = OilMoE(
            n_features=n_features, gate_input_dim=gate_dim,
            n_experts=N_EXPERTS, d_model=D_MODEL, pred_len=1,
            gate_hidden_dim=16,
            expert_configs=[{
                "input_dim": n_features, "d_model": D_MODEL,
                "d_state": 16, "n_layers": 1, "dropout": dropout_rate
            } for _ in range(N_EXPERTS)])
        
        model, history = train_moe_model(model, train_loader, val_loader,
            epochs=100, lr=1e-4, balance_weight=0.01, patience=15,
            device=device, loss_alpha=1.0, loss_beta=0.01)
        
        _, _, _, _, _, metrics = evaluate_moe_model(
            model, test_loader, scaler_target, base_p, true_p, device=device)
        
        # Evaluate already printed results during evaluation
        res = {
            "dropout": dropout_rate,
            "seed": seed,
            "dir_acc": metrics["dir_acc"],
            "mae": metrics["mae_model"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape_model"],
            "return_corr": metrics["r_corr"],
            "vs_rw_pct": metrics["pct_vs_rw"],
        }
        results.append(res)
        
        print(f"  => DA={res['dir_acc']:.1f}%  MAE={res['mae']:.4f}")
        
        # Clean up aggressively
        del model, train_loader, val_loader, test_loader
        del regime_detector, regime_proba_aligned, regime_labels_aligned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"  [GPU] cleared")
        
        # Save intermediate results
        out_file = os.path.join(SCRIPT_DIR, "..", "output", f"dropout_{dropout_rate}.csv")
        pd.DataFrame(results).to_csv(out_file, index=False)
    
    return results

if __name__ == "__main__":
    print("="*70)
    print("  Run Dropout=0.4 (Part 2)")
    print("="*70)
    
    results_04 = run_one_dropout(0.4)
    
    # Aggregate with existing
    print("\n\n>>> Aggregating all results...")
    all_dfs = []
    for dp in [0.3, 0.4]:
        dp_file = os.path.join(SCRIPT_DIR, "..", "output", f"dropout_{dp}.csv")
        if os.path.exists(dp_file):
            all_dfs.append(pd.read_csv(dp_file))
    combined = pd.concat(all_dfs, ignore_index=True)
    final_file = os.path.join(SCRIPT_DIR, "..", "output", "dropout_sensitivity_partial.csv")
    combined.to_csv(final_file, index=False)
    print(f"  Saved partial to: {final_file}")
    print(f"  Total rows: {len(combined)}")
