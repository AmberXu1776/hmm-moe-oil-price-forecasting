"""
hmm_leakage_test.py — HMM训练范围对照实验
full-sample HMM vs train-only HMM，验证数据泄漏影响
"""
import os, sys, json, warnings
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
    train_moe_model, evaluate_moe_model, OilRegimeDetector,
    build_log_returns, build_gate_features,
)

FILE_PATH = r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv"
FEAT_COLS = ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
             "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
             "Excavator","Excavator_YoY","M2_M1_Spread"]
TARGET_COL = "Daqing"
SEQ_LEN = 52
SEED = 42

def run_with_train_only_hmm(seed=42):
    """Manually replicate the pipeline with train-only HMM fitting."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: Load data
    df = pd.read_csv(FILE_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    all_cols = FEAT_COLS + [TARGET_COL]
    df[all_cols] = df[all_cols].interpolate(method="linear")
    for col in all_cols:
        m, s = df[col].mean(), df[col].std()
        df.loc[(df[col] < m-3*s) | (df[col] > m+3*s), col] = df[col].median()
    
    # Step 2: Build HMM features
    regime_detector = OilRegimeDetector(n_regimes=3, covariance_type="diag",
                                         n_init=15, min_covar=1e-6)
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, FEAT_COLS, TARGET_COL, vol_window=12, hmm_mode="multi")
    
    # Step 3: Build log returns
    lr_df, all_feat_cols, scaler_feat, scaler_target, original_prices = \
        build_log_returns(df.copy(), FEAT_COLS, TARGET_COL)
    
    # Step 4: Figure out alignment and split points
    T_hmm = len(hmm_features)
    T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)
    T_samples = T_aligned - SEQ_LEN
    n_train = int(T_samples * 0.8)
    n_val = int(T_samples * 0.1)
    
    # Step 5: Fit HMM ONLY on training portion
    # Training samples use hmm_features indices 0 to (SEQ_LEN + n_train)
    # But actually hmm_features has T_hmm entries, and T_aligned = min(T_hmm, T_lr)
    # The sliding window starts at index SEQ_LEN of the aligned data
    # Sample i (0-indexed) uses hmm_features_aligned[i+SEQ_LEN-1]
    # So training samples (0..n_train-1) use hmm_features_aligned[SEQ_LEN-1..SEQ_LEN+n_train-1]
    # In the original (non-aligned) hmm_features, we need to figure the offset
    offset = T_hmm - T_aligned
    train_hmm_end = offset + SEQ_LEN + n_train
    train_hmm_data = hmm_features[:train_hmm_end]
    
    print(f"\n  [Train-Only HMM] Fitting on {len(train_hmm_data)}/{len(hmm_features)} samples "
          f"(first {train_hmm_end} of {T_hmm})")
    
    regime_detector.fit(train_hmm_data)
    
    # Step 6: Predict on ALL data (forward algorithm handles unseen data)
    regime_proba_raw, regime_labels_raw = regime_detector.predict_proba(hmm_features)
    regime_detector.describe_regimes(hmm_features, regime_labels_raw)
    
    # Step 7: Align and build sliding windows (same as original)
    regime_proba_aligned = regime_proba_raw[-T_aligned:]
    regime_labels_aligned = regime_labels_raw[-T_aligned:]
    lr_df_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    original_prices = original_prices[-T_aligned - 1:]
    n_raw_feats = len(FEAT_COLS)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    X_seq_list, X_gate_list, Y_list = [], [], []
    for i in range(SEQ_LEN, T_aligned):
        X_seq_list.append(lr_df_aligned[all_feat_cols].iloc[i-SEQ_LEN:i].values)
        X_gate_list.append(build_gate_features(
            regime_proba_aligned, lr_df_aligned, all_feat_cols,
            TARGET_COL, n_raw_feats, i, 4, 12))
        Y_list.append(lr_df_aligned["target_scaled"].iloc[i])
    
    X_seq = np.array(X_seq_list)
    X_gate = np.array(X_gate_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    print(f"  Samples: {len(Y)}, X_seq: {X_seq.shape}, X_gate: {X_gate.shape}")
    gate_dim = X_gate.shape[1]
    
    from sklearn.preprocessing import StandardScaler
    scaler_gate = StandardScaler()
    X_gate = scaler_gate.fit_transform(X_gate).astype(np.float32)
    
    N = len(Y)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    
    def make_loader(s, e, shuffle=False):
        return DataLoader(
            TensorDataset(to_t(X_seq[s:e]), to_t(X_gate[s:e]),
                          to_t(Y[s:e]).unsqueeze(1)),
            batch_size=32, shuffle=shuffle)
    
    train_loader = make_loader(0, n_train, shuffle=True)
    val_loader = make_loader(n_train, n_train+n_val)
    test_loader = make_loader(n_train+n_val, N)
    test_ds = TensorDataset(to_t(X_seq[n_train+n_val:]), to_t(X_gate[n_train+n_val:]),
                            to_t(Y[n_train+n_val:]).unsqueeze(1))
    
    bx, bg, _ = next(iter(train_loader))
    n_features = bx.shape[2]
    
    # Get test prices
    test_offset = SEQ_LEN + n_train + n_val
    bp = original_prices[test_offset:test_offset+len(test_ds)]
    tp = original_prices[test_offset+1:test_offset+1+len(test_ds)]
    
    # Train HMM-MoE
    model = OilMoE(
        n_features=n_features, gate_input_dim=gate_dim,
        n_experts=3, d_model=32, pred_len=1, gate_hidden_dim=16,
        expert_configs=[{"input_dim": n_features, "d_model": 32,
                         "d_state": 16, "n_layers": 1, "dropout": 0.55} for _ in range(3)])
    
    model, _ = train_moe_model(model, train_loader, val_loader,
        epochs=100, lr=1e-4, balance_weight=0.01, patience=15,
        device=device, loss_alpha=1.0, loss_beta=0.01)
    
    _, _, _, _, _, metrics = evaluate_moe_model(
        model, test_loader, scaler_target, bp, tp, device=device)
    
    return metrics

if __name__ == "__main__":
    print("="*60)
    print("  HMM Leakage Test: Full-Sample vs Train-Only")
    print("="*60)
    
    # Test 1: Full-sample (use standard pipeline)
    print("\n\n>>> TEST 1: Full-Sample HMM (standard pipeline) <<<")
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    (train_loader, val_loader, test_loader,
     _, scaler_target, _, _, gate_dim,
     _, _, _, base_p, true_p, _) = oil_price_data_preprocess_v6(
        file_path=FILE_PATH, feat_cols=FEAT_COLS,
        target_col=TARGET_COL, seq_len=SEQ_LEN,
        n_regimes=3, gate_window=4, vol_window=12, hmm_mode="multi")
    
    bx, bg, _ = next(iter(train_loader))
    nf = bx.shape[2]
    
    set_seed(SEED)
    model1 = OilMoE(
        n_features=nf, gate_input_dim=gate_dim,
        n_experts=3, d_model=32, pred_len=1, gate_hidden_dim=16,
        expert_configs=[{"input_dim": nf, "d_model": 32,
                         "d_state": 16, "n_layers": 1, "dropout": 0.55} for _ in range(3)])
    
    model1, _ = train_moe_model(model1, train_loader, val_loader,
        epochs=100, lr=1e-4, balance_weight=0.01, patience=15,
        device=device, loss_alpha=1.0, loss_beta=0.01)
    
    _, _, _, _, _, m1 = evaluate_moe_model(
        model1, test_loader, scaler_target, base_p, true_p, device=device)
    print(f"\n  Full-Sample: DA={m1['dir_acc']:.1f}%  MAE={m1['mae_model']:.4f}")
    
    del model1
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Test 2: Train-only HMM
    print("\n\n>>> TEST 2: Train-Only HMM <<<")
    m2 = run_with_train_only_hmm(seed=SEED)
    print(f"\n  Train-Only: DA={m2['dir_acc']:.1f}%  MAE={m2['mae_model']:.4f}")
    
    # Compare
    print(f"\n\n{'='*60}")
    print(f"  COMPARISON (seed={SEED})")
    print(f"{'='*60}")
    print(f"  Full-Sample HMM:  DA={m1['dir_acc']:.1f}%  MAE={m1['mae_model']:.4f}")
    print(f"  Train-Only HMM:   DA={m2['dir_acc']:.1f}%  MAE={m2['mae_model']:.4f}")
    da_diff = abs(m1['dir_acc'] - m2['dir_acc'])
    mae_diff = abs(m1['mae_model'] - m2['mae_model'])
    print(f"  DA difference:    {da_diff:.1f}pp")
    print(f"  MAE difference:   {mae_diff:.4f}")
    
    if da_diff < 3.0:
        print(f"\n  ✅ DA difference < 3pp — HMM training scope has minimal impact")
    else:
        print(f"\n  ⚠️ DA difference >= 3pp — HMM training scope may matter")
