"""
dropout_sensitivity.py — HMM-MoE dropout rate sensitivity ablation
Test dropout values: {0.3, 0.4, 0.55, 0.7}
Each value trained with SEED=42, 123, 2024 → report mean ± std
"""
import os, sys, json, warnings
import numpy as np, torch, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
warnings.filterwarnings("ignore")

_orig = json.JSONEncoder.default
def _jp(self, obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return _orig(self, obj)
json.JSONEncoder.default = _jp

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
DROPOUTS = [0.3, 0.4, 0.55, 0.7]
N_EXPERTS = 3
N_REGIMES = 3
D_MODEL = 32

def run_dropout_experiment(dropout_rate, seed):
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
    
    # Build model with given dropout
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
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "dropout": dropout_rate,
        "seed": seed,
        "dir_acc": metrics["dir_acc"],
        "mae": metrics["mae_model"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape_model"],
        "return_corr": metrics["r_corr"],
        "vs_rw_pct": metrics["pct_vs_rw"],
    }

if __name__ == "__main__":
    print("="*70)
    print("  Dropout Rate Sensitivity Ablation")
    print(f"  Dropouts: {DROPOUTS}")
    print(f"  Seeds: {SEEDS}")
    print("="*70)
    
    all_results = []
    
    for dropout in DROPOUTS:
        print(f"\n\n>>> Dropout = {dropout}")
        for seed in SEEDS:
            print(f"  Seed = {seed}")
            res = run_dropout_experiment(dropout, seed)
            all_results.append(res)
            print(f"    DA={res['dir_acc']:.1f}%  MAE={res['mae']:.4f}")
    
    # Aggregate results
    df = pd.DataFrame(all_results)
    summary = df.groupby("dropout").agg({
        "dir_acc": ["mean", "std"],
        "mae": ["mean", "std"],
    }).round(3)
    
    print("\n\n" + "="*70)
    print("  AGGREGATED RESULTS")
    print("="*70)
    print(summary)
    
    # Save
    output_file = os.path.join(SCRIPT_DIR, "..", "output", "dropout_sensitivity.csv")
    df.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    
    # Print markdown table
    print("\n  Markdown Table:")
    print("\n| Dropout | DA (mean) | DA (std) | MAE (mean) | MAE (std) |")
    print("|---------|-----------|----------|------------|-----------|")
    for dropout in DROPOUTS:
        subset = df[df["dropout"] == dropout]
        da_mean = (subset["dir_acc"].mean() * 10).round(1) / 10
        da_std = (subset["dir_acc"].std() * 10).round(2) / 10
        mae_mean = subset["mae"].mean().round(3)
        mae_std = subset["mae"].std().round(4)
        print(f"| {dropout:>5} | {da_mean:>9.1f}% | {da_std:>8.2f}pp | {mae_mean:>10.3f} | {mae_std:>9.4f} |")
