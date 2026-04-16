"""multi_seed.py — 多随机种子稳健性实验"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd, torch
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

SEEDS = [42, 123, 2024]
EPOCHS = 100
PATIENCE = 15

CONFIG = {
    "FILE_PATH": r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv",
    "FEAT_COLS": ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
                  "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
                  "Excavator","Excavator_YoY","M2_M1_Spread"],
    "TARGET_COL": "Daqing",
    "N_REGIMES": 3, "HMM_MODE": "multi", "N_EXPERTS": 3,
    "SEQ_LEN": 52, "D_MODEL": 32, "D_STATE": 16, "N_LAYERS": 1,
    "DROPOUT": 0.55, "GATE_HIDDEN_DIM": 16, "GATE_WINDOW": 4, "VOL_WINDOW": 12,
    "LR": 1e-4, "EPOCHS": EPOCHS, "PATIENCE": PATIENCE,
    "BALANCE_WEIGHT": 0.01, "LOSS_ALPHA": 1.0, "LOSS_BETA": 0.01,
    "OUTPUT_DIR": "output",
}

def run_one(seed):
    cfg = CONFIG.copy()
    cfg["SEED"] = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(SCRIPT_DIR, cfg["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed)

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
        expert_configs=[{"input_dim": n_features, "d_model": cfg["D_MODEL"],
                         "d_state": cfg["D_STATE"], "n_layers": cfg["N_LAYERS"],
                         "dropout": cfg["DROPOUT"]} for _ in range(cfg["N_EXPERTS"])])

    model, _ = train_moe_model(
        model, train_loader, val_loader,
        epochs=cfg["EPOCHS"], lr=cfg["LR"],
        balance_weight=cfg["BALANCE_WEIGHT"], patience=cfg["PATIENCE"],
        device=device, loss_alpha=cfg["LOSS_ALPHA"], loss_beta=cfg["LOSS_BETA"])

    _, _, _, _, _, metrics = evaluate_moe_model(
        model, test_loader, scaler_target,
        test_base_prices, test_true_prices, device=device)

    return {
        'seed': seed,
        'da': metrics['dir_acc'],
        'mae': metrics['mae_model'],
        'mae_rw': metrics['mae_rw'],
        'mape': metrics['mape_model'],
        'r_corr': metrics['r_corr'],
    }

def main():
    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\n  SEED = {seed}\n{'='*60}")
        t0 = time.time()
        r = run_one(seed)
        r['runtime_s'] = time.time() - t0
        results.append(r)
        print(f"  DA={r['da']:.1f}%  MAE={r['mae']:.4f}  MAPE={r['mape']:.2f}%  ({r['runtime_s']:.0f}s)")

    df = pd.DataFrame(results)
    print(f"\n{'='*60}\n  MULTI-SEED SUMMARY\n{'='*60}")
    print(f"  DA  : {df['da'].mean():.1f}% ± {df['da'].std():.1f}%  [{df['da'].min():.1f}, {df['da'].max():.1f}]")
    print(f"  MAE : {df['mae'].mean():.4f} ± {df['mae'].std():.4f}  [{df['mae'].min():.4f}, {df['mae'].max():.4f}]")
    print(f"  MAPE: {df['mape'].mean():.2f}% ± {df['mape'].std():.2f}%")

    output_dir = os.path.join(SCRIPT_DIR, "output")
    df.to_csv(os.path.join(output_dir, "multi_seed_results.csv"), index=False)
    print(f"\nSaved to {output_dir}/multi_seed_results.csv")

if __name__ == "__main__":
    main()
