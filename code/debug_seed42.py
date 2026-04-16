"""debug_seed42.py — 排查HMM-MoE seed=42的DA差异"""
import os, sys, json, numpy as np, torch
os.chdir(r'D:\阶梯计划\论文\P1_MambaMoE_OilPrice\code')
sys.path.insert(0, 'src')

_orig = json.JSONEncoder.default
def _jp(self, obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return _orig(self, obj)
json.JSONEncoder.default = _jp

from layer4_mamba_moe import OilMoE, set_seed, oil_price_data_preprocess_v6, train_moe_model, evaluate_moe_model

FILE_PATH = r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv"
FEAT_COLS = ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
             "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
             "Excavator","Excavator_YoY","M2_M1_Spread"]

for seed in [42, 123, 2024]:
    print(f"\n{'='*50}")
    print(f"  SEED = {seed}")
    print(f"{'='*50}")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (train_loader, val_loader, test_loader,
     scaler_feat, scaler_target, scaler_gate,
     all_feat_cols, gate_dim,
     regime_detector, regime_proba, regime_labels,
     test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
        file_path=FILE_PATH, feat_cols=FEAT_COLS,
        target_col="Daqing", seq_len=52, n_regimes=3,
        gate_window=4, vol_window=12, hmm_mode="multi")

    bx, bg, _ = next(iter(train_loader))
    n_features = bx.shape[2]

    model = OilMoE(
        n_features=n_features, gate_input_dim=gate_dim,
        n_experts=3, d_model=32, pred_len=1, gate_hidden_dim=16,
        expert_configs=[{"input_dim": n_features, "d_model": 32,
                         "d_state": 16, "n_layers": 1, "dropout": 0.55} for _ in range(3)])

    model, _ = train_moe_model(model, train_loader, val_loader,
        epochs=100, lr=1e-4, balance_weight=0.01, patience=15,
        device=device, loss_alpha=1.0, loss_beta=0.01)

    _, _, _, _, _, metrics = evaluate_moe_model(
        model, test_loader, scaler_target,
        test_base_prices, test_true_prices, device=device)

    da = metrics["dir_acc"]
    mae = metrics["mae_model"]
    print(f"  RESULT: DA={da:.1f}%  MAE={mae:.4f}")

print("\nDone.")
