"""hyperparam_sensitivity.py — 超参敏感性分析"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
warnings.filterwarnings("ignore")

# Patch JSON for numpy types
_orig = json.JSONEncoder.default
def _jp(self, obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return _orig(self, obj)
json.JSONEncoder.default = _jp

from layer4_mamba_moe import (
    OilMoE, OilRegimeDetector, set_seed,
    build_log_returns, build_gate_features,
    train_moe_model, evaluate_moe_model,
    oil_price_data_preprocess_v6,
)

FILE_PATH = r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv"
FEAT_COLS = ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
             "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
             "Excavator","Excavator_YoY","M2_M1_Spread"]
TARGET_COL = "Daqing"
BASE_CFG = {
    "N_REGIMES": 3, "HMM_MODE": "multi", "N_EXPERTS": 3,
    "SEQ_LEN": 52, "D_MODEL": 32, "D_STATE": 16, "N_LAYERS": 1,
    "DROPOUT": 0.55, "GATE_HIDDEN_DIM": 16, "GATE_WINDOW": 4, "VOL_WINDOW": 12,
    "LR": 1e-4, "EPOCHS": 100, "PATIENCE": 15,
    "BALANCE_WEIGHT": 0.01, "LOSS_ALPHA": 1.0, "LOSS_BETA": 0.01,
    "SEED": 42,
    "BATCH_SIZE": 16,  # reduced from 32 for OOM
}

def to_t(a): return torch.tensor(a, dtype=torch.float32)

def run_single(cfg_overrides, label):
    """Run one experiment with config overrides."""
    cfg = BASE_CFG.copy()
    cfg.update(cfg_overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["SEED"])

    (train_loader, val_loader, test_loader,
     scaler_feat, scaler_target, scaler_gate,
     all_feat_cols, gate_dim,
     regime_detector, regime_proba, regime_labels,
     test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
        file_path=FILE_PATH, feat_cols=FEAT_COLS,
        target_col=TARGET_COL, seq_len=cfg["SEQ_LEN"],
        n_regimes=cfg["N_REGIMES"], gate_window=cfg["GATE_WINDOW"],
        vol_window=cfg["VOL_WINDOW"], hmm_mode=cfg["HMM_MODE"])

    bx, bg, _ = next(iter(train_loader))
    n_features = bx.shape[2]

    model = OilMoE(
        n_features=n_features, gate_input_dim=gate_dim,
        n_experts=cfg["N_EXPERTS"], d_model=cfg["D_MODEL"], pred_len=1,
        gate_hidden_dim=cfg["GATE_HIDDEN_DIM"],
        expert_configs=[{
            "input_dim": n_features, "d_model": cfg["D_MODEL"],
            "d_state": cfg["D_STATE"], "n_layers": cfg["N_LAYERS"],
            "dropout": cfg["DROPOUT"],
        } for _ in range(cfg["N_EXPERTS"])])

    model, _ = train_moe_model(
        model, train_loader, val_loader,
        epochs=cfg["EPOCHS"], lr=cfg["LR"],
        balance_weight=cfg["BALANCE_WEIGHT"], patience=cfg["PATIENCE"],
        device=device, loss_alpha=cfg["LOSS_ALPHA"], loss_beta=cfg["LOSS_BETA"])

    _, _, _, _, _, metrics = evaluate_moe_model(
        model, test_loader, scaler_target,
        test_base_prices, test_true_prices, device=device)

    return {'label': label, 'da': metrics['dir_acc'], 'mae': metrics['mae_model'], 'mape': metrics['mape_model']}

def main():
    results = []

    # --- Lambda sensitivity ---
    print("\n" + "="*60)
    print("  LAMBDA SENSITIVITY (direction loss weight)")
    print("="*60)
    for lam in [0, 0.5, 1.0, 2.0, 5.0]:
        label = f"lambda={lam}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        r = run_single({"LOSS_ALPHA": lam}, label)
        r['param'] = 'lambda'
        r['value'] = lam
        r['runtime'] = time.time() - t0
        results.append(r)
        print(f"  DA={r['da']:.1f}%  MAE={r['mae']:.4f}  MAPE={r['mape']:.2f}%  ({r['runtime']:.0f}s)")

    # --- Expert count sensitivity ---
    print("\n" + "="*60)
    print("  EXPERT COUNT SENSITIVITY")
    print("="*60)
    for n_exp in [2, 3, 4, 5]:
        label = f"experts={n_exp}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        r = run_single({"N_EXPERTS": n_exp}, label)
        r['param'] = 'n_experts'
        r['value'] = n_exp
        r['runtime'] = time.time() - t0
        results.append(r)
        print(f"  DA={r['da']:.1f}%  MAE={r['mae']:.4f}  MAPE={r['mape']:.2f}%  ({r['runtime']:.0f}s)")

    # --- Lookback window sensitivity ---
    print("\n" + "="*60)
    print("  LOOKBACK WINDOW SENSITIVITY")
    print("="*60)
    for seq_len in [26, 39, 52, 78]:
        label = f"lookback={seq_len}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        cfg = BASE_CFG.copy()
        cfg["SEQ_LEN"] = seq_len
        device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(cfg["SEED"])

        (train_loader, val_loader, test_loader,
         scaler_feat, scaler_target, scaler_gate,
         all_feat_cols, gate_dim,
         regime_detector, regime_proba, regime_labels,
         test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
            file_path=cfg["FILE_PATH"], feat_cols=cfg["FEAT_COLS"],
            target_col=cfg["TARGET_COL"], seq_len=cfg["SEQ_LEN"],
            n_regimes=cfg["N_REGIMES"], gate_window=cfg["GATE_WINDOW"],
            vol_window=cfg["VOL_WINDOW"], hmm_mode=cfg["HMM_MODE"])

        bx, bg, _ = next(iter(train_loader))
        n_features = bx.shape[2]

        model = OilMoE(
            n_features=n_features, gate_input_dim=gate_dim,
            n_experts=cfg["N_EXPERTS"], d_model=cfg["D_MODEL"], pred_len=1,
            gate_hidden_dim=cfg["GATE_HIDDEN_DIM"],
            expert_configs=[{
                "input_dim": n_features, "d_model": cfg["D_MODEL"],
                "d_state": cfg["D_STATE"], "n_layers": cfg["N_LAYERS"],
                "dropout": cfg["DROPOUT"],
            } for _ in range(cfg["N_EXPERTS"])])

        model, _ = train_moe_model(
            model, train_loader, val_loader,
            epochs=cfg["EPOCHS"], lr=cfg["LR"],
            balance_weight=cfg["BALANCE_WEIGHT"], patience=cfg["PATIENCE"],
            device=device, loss_alpha=cfg["LOSS_ALPHA"], loss_beta=cfg["LOSS_BETA"])

        _, _, _, _, _, metrics = evaluate_moe_model(
            model, test_loader, scaler_target,
            test_base_prices, test_true_prices, device=device)

        r = {
            'label': label, 'da': metrics['dir_acc'], 'mae': metrics['mae_model'], 'mape': metrics['mape'],
            'param': 'lookback', 'value': seq_len, 'runtime': time.time() - t0
        }
        results.append(r)
        print(f"  DA={r['da']:.1f}%  MAE={r['mae']:.4f}  MAPE={r['mape']:.2f}%  ({r['runtime']:.0f}s)")

    # --- Summary ---
    df = pd.DataFrame(results)
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "hyperparam_sensitivity.csv"), index=False)

    print(f"\n{'='*60}")
    print("  HYPERPARAMETER SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    for param in ['lambda', 'n_experts', 'lookback']:
        sub = df[df['param'] == param]
        print(f"\n  {param}:")
        for _, row in sub.iterrows():
            print(f"    {row['value']:>5} → DA={row['da']:.1f}%  MAE={row['mae']:.4f}")

    print(f"\nSaved to {output_dir}/hyperparam_sensitivity.csv")

if __name__ == "__main__":
    main()
