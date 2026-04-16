"""
Rolling window v2: 5-7 windows + improved DM test with Newey-West lag selection.
Modifies CONFIG to reduce RW_TRAIN and RW_STEP, then runs run_rolling_window with
an improved DM test that uses sqrt(n) Newey-West lag.
"""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import numpy as np
import pandas as pd
from scipy import stats

# Import main and monkey-patch CONFIG
import main as m
m.CONFIG["RUN_SINGLE"] = False
m.CONFIG["RUN_ROLLING_WINDOW"] = False  # we call it ourselves

# Adjust for more windows: reduce train and step
m.CONFIG["RW_TRAIN"] = 600
m.CONFIG["RW_STEP"] = 40
# Keep RW_VAL=100, RW_TEST=100 → window_size=800

print("="*60)
print("ROLLING WINDOW V2 - Extended windows + Improved DM test")
print(f"RW_TRAIN=600, RW_VAL=100, RW_TEST=100, RW_STEP=40")
print("="*60)

# Run rolling window
m.run_rolling_window(m.CONFIG)

# Now load results and do improved DM test
output_dir = os.path.join(SCRIPT_DIR, m.CONFIG["OUTPUT_DIR"])
res_path = os.path.join(output_dir, "rolling_window_results.csv")
if not os.path.exists(res_path):
    print("ERROR: rolling_window_results.csv not found!")
    sys.exit(1)

df_res = pd.read_csv(res_path)
print(f"\n\n{'='*60}")
print(f"IMPROVED DM TEST (Newey-West, lag=sqrt(n))")
print(f"{'='*60}")

# We need raw abs errors for DM test. Re-run is expensive, so let's check if
# the results file has them. It doesn't (they're dropped). Let's re-derive from
# saved data or just note that we need to modify the save logic.
# Actually, the original code stores abs_err arrays per row. Let me check if 
# rolling_dm_test.csv was saved with the original DM. We'll just report improved DM.

orig_dm_path = os.path.join(output_dir, "rolling_dm_test.csv")
if os.path.exists(orig_dm_path):
    df_orig_dm = pd.read_csv(orig_dm_path)
    print("Original DM results:")
    print(df_orig_dm.to_string(index=False))

# The issue is we don't have the raw error arrays after saving to CSV.
# We need to modify run_rolling_window to also save raw errors, or compute DM inline.
# Better approach: patch the function to save v2 files.

# Actually, let me just re-run with modified save logic by wrapping.
# The simplest approach: modify the DM test section in the saved results by
# recalculating from the raw data. But raw errors aren't saved.

# Let me instead directly modify main.py's run_rolling_window to use improved DM
# and save v2 files. But that's invasive.

# Best approach: re-run is too expensive. Instead, let's apply Newey-West correction
# to the existing DM statistics if we have them, or note that the improved test
# needs to be integrated into the code.

# For now, let's modify the code and save properly. Let's patch the function.
print("\nNote: Improved DM test requires raw error arrays.")
print("Modifying code to save v2 results with Newey-West DM test...")

# Actually the rolling window already ran above and saved results. Let me check
# if raw errors are in the df. They're not - they're dropped before saving.
# Let me modify the code to keep them for v2.

# Let me take a different approach: modify main.py to add improved DM and v2 saving,
# then re-run. But that's expensive. 

# PRACTICAL APPROACH: Modify the code, save v2 files inline.
# Let me just edit the end of run_rolling_window to also compute improved DM.

print("\nPatching and re-running with improved DM...")

# Modify the function to save v2
import types

orig_rw = m.run_rolling_window

def patched_rw(cfg):
    """Run rolling window with improved DM test and v2 output."""
    import torch.nn as nn
    from layer4_mamba_moe import (
        OilMoE, OilRegimeDetector, set_seed,
        build_log_returns, build_gate_features,
        train_moe_model, evaluate_moe_model,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from torch.utils.data import DataLoader, TensorDataset

    output_dir = os.path.join(SCRIPT_DIR, cfg["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(cfg["FILE_PATH"]).sort_values("date").reset_index(drop=True)
    all_cols = cfg["FEAT_COLS"] + [cfg["TARGET_COL"]]
    df[all_cols] = df[all_cols].interpolate(method="linear")
    for col in all_cols:
        mv, sv = df[col].mean(), df[col].std()
        outliers = (df[col] < mv - 3*sv) | (df[col] > mv + 3*sv)
        df.loc[outliers, col] = df[col].median()

    lr_df, all_feat_cols, _, _, original_prices = build_log_returns(
        df.copy(), cfg["FEAT_COLS"], cfg["TARGET_COL"])

    regime_detector = OilRegimeDetector(
        n_regimes=cfg["N_REGIMES"], covariance_type="diag",
        n_init=15, min_covar=1e-6)
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, cfg["FEAT_COLS"], cfg["TARGET_COL"],
        vol_window=cfg["VOL_WINDOW"], hmm_mode=cfg["HMM_MODE"])
    regime_detector.fit(hmm_features)
    regime_proba_raw, _ = regime_detector.predict_proba(hmm_features)

    T_hmm = len(regime_proba_raw)
    T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)
    regime_proba = regime_proba_raw[-T_aligned:]
    lr_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    orig_p = original_prices[-T_aligned - 1:]
    n_regimes = cfg["N_REGIMES"]
    n_raw = len(cfg["FEAT_COLS"])
    seq_len = cfg["SEQ_LEN"]
    gate_window = cfg["GATE_WINDOW"]
    vol_window = cfg["VOL_WINDOW"]

    X_seq_all, X_gate_all, Y_all = [], [], []
    for i in range(seq_len, T_aligned):
        X_seq_all.append(lr_aligned[all_feat_cols].iloc[i - seq_len:i].values)
        X_gate_all.append(build_gate_features(
            regime_proba, lr_aligned, all_feat_cols,
            cfg["TARGET_COL"], n_raw, i, gate_window, vol_window))
        Y_all.append(lr_aligned["target_scaled"].iloc[i])
    X_seq_all = np.array(X_seq_all, dtype=np.float32)
    X_gate_all = np.array(X_gate_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    scaler_gate = StandardScaler()
    X_gate_all = scaler_gate.fit_transform(X_gate_all).astype(np.float32)

    N_total = len(Y_all)
    n_features = X_seq_all.shape[2]
    gate_dim = X_gate_all.shape[1]

    rw_train = cfg["RW_TRAIN"]
    rw_val = cfg["RW_VAL"]
    rw_test = cfg["RW_TEST"]
    rw_step = cfg["RW_STEP"]
    window_size = rw_train + rw_val + rw_test

    print(f"\n{'='*60}")
    print(f"  ROLLING WINDOW V2 VALIDATION")
    print(f"  Total samples: {N_total}, Window: {window_size}, Step: {rw_step}")
    n_windows = (N_total - window_size) // rw_step + 1
    print(f"  Number of windows: ~{n_windows}")
    print(f"{'='*60}")

    results = []
    start = 0
    win_id = 0

    while start + window_size <= N_total:
        win_id += 1
        s_test = start + rw_train + rw_val
        e_test = s_test + rw_test
        print(f"\n--- Window {win_id}: [{start}:{s_test}] train+val, [{s_test}:{e_test}] test ---")

        X_tr = X_seq_all[start:start + rw_train]
        Xg_tr = X_gate_all[start:start + rw_train]
        Y_tr = Y_all[start:start + rw_train]
        X_va = X_seq_all[start + rw_train:s_test]
        Xg_va = X_gate_all[start + rw_train:s_test]
        Y_va = Y_all[start + rw_train:s_test]
        X_te = X_seq_all[s_test:e_test]
        Xg_te = X_gate_all[s_test:e_test]
        Y_te = Y_all[s_test:e_test]

        sc_feat = StandardScaler()
        flat_tr = X_tr.reshape(-1, n_features)
        sc_feat.fit(flat_tr)
        X_tr_s = sc_feat.transform(flat_tr).reshape(X_tr.shape).astype(np.float32)
        X_va_s = sc_feat.transform(X_va.reshape(-1, n_features)).reshape(X_va.shape).astype(np.float32)
        X_te_s = sc_feat.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape).astype(np.float32)

        sc_y = StandardScaler()
        Y_tr_s = sc_y.fit_transform(Y_tr.reshape(-1, 1)).flatten().astype(np.float32)
        Y_va_s = sc_y.transform(Y_va.reshape(-1, 1)).flatten().astype(np.float32)
        Y_te_s = sc_y.transform(Y_te.reshape(-1, 1)).flatten().astype(np.float32)

        sc_g = StandardScaler()
        Xg_tr_s = sc_g.fit_transform(Xg_tr).astype(np.float32)
        Xg_va_s = sc_g.transform(Xg_va).astype(np.float32)
        Xg_te_s = sc_g.transform(Xg_te).astype(np.float32)

        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        def mk_loader(xs, xg, y, shuffle=False):
            return DataLoader(TensorDataset(to_t(xs), to_t(xg), to_t(y).unsqueeze(1)),
                              batch_size=32, shuffle=shuffle)

        tr_ld = mk_loader(X_tr_s, Xg_tr_s, Y_tr_s, shuffle=True)
        va_ld = mk_loader(X_va_s, Xg_va_s, Y_va_s)
        te_ld = mk_loader(X_te_s, Xg_te_s, Y_te_s)

        # HMM-MoE
        set_seed(cfg["SEED"])
        model = OilMoE(
            n_features=n_features, gate_input_dim=gate_dim,
            n_experts=cfg["N_EXPERTS"],
            d_model=cfg["D_MODEL"], pred_len=1,
            gate_hidden_dim=cfg["GATE_HIDDEN_DIM"],
            expert_configs=[{
                "input_dim": n_features, "d_model": cfg["D_MODEL"],
                "d_state": cfg["D_STATE"], "n_layers": cfg["N_LAYERS"],
                "dropout": cfg["DROPOUT"],
            } for _ in range(cfg["N_EXPERTS"])],
        )
        model, _ = train_moe_model(
            model, tr_ld, va_ld,
            epochs=cfg["RW_EPOCHS"], lr=cfg["LR"],
            balance_weight=cfg["BALANCE_WEIGHT"],
            patience=cfg["RW_PATIENCE"], device=device,
            loss_alpha=cfg["LOSS_ALPHA"], loss_beta=cfg["LOSS_BETA"],
        )

        model.eval()
        r_pred_list, r_true_list = [], []
        with torch.no_grad():
            for bx, bg, by in te_ld:
                pred, _ = model(bx.to(device), bg.to(device))
                r_pred_list.append(pred.cpu().numpy().flatten())
                r_true_list.append(by.numpy().flatten())
        r_pred_moe = sc_y.inverse_transform(np.concatenate(r_pred_list).reshape(-1,1)).flatten()
        r_true = sc_y.inverse_transform(np.concatenate(r_true_list).reshape(-1,1)).flatten()

        price_offset = seq_len + s_test
        base_p = orig_p[price_offset:price_offset + rw_test]
        true_p = orig_p[price_offset + 1:price_offset + 1 + rw_test]
        n_t = min(len(r_pred_moe), len(base_p), len(true_p))
        r_pred_moe = r_pred_moe[:n_t]
        r_true = r_true[:n_t]
        base_p = base_p[:n_t]
        true_p = true_p[:n_t]

        pred_p_moe = base_p * np.exp(r_pred_moe)
        mae_moe = mean_absolute_error(true_p, pred_p_moe)
        da_moe = np.mean((r_pred_moe * r_true) > 0) * 100
        abs_err_moe = np.abs(true_p - pred_p_moe)

        # XGBoost
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=cfg["SEED"])
        xgb_tr = X_tr_s[:, -1, :]
        xgb_te = X_te_s[:, -1, :]
        xgb.fit(xgb_tr, Y_tr_s)
        r_pred_xgb_sc = xgb.predict(xgb_te)
        r_pred_xgb = sc_y.inverse_transform(r_pred_xgb_sc.reshape(-1,1)).flatten()[:n_t]
        pred_p_xgb = base_p * np.exp(r_pred_xgb)
        mae_xgb = mean_absolute_error(true_p, pred_p_xgb)
        da_xgb = np.mean((r_pred_xgb * r_true) > 0) * 100
        abs_err_xgb = np.abs(true_p - pred_p_xgb)

        # LSTM
        class SimpleLSTM(nn.Module):
            def __init__(self, inp, hid=64, layers=2, drop=0.2):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
                self.fc = nn.Linear(hid, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        set_seed(cfg["SEED"])
        lstm = SimpleLSTM(n_features).to(device)
        lstm_tr = DataLoader(TensorDataset(to_t(X_tr_s), to_t(Y_tr_s).unsqueeze(1)),
                             batch_size=32, shuffle=True)
        lstm_va = DataLoader(TensorDataset(to_t(X_va_s), to_t(Y_va_s).unsqueeze(1)),
                             batch_size=32)
        opt_l = torch.optim.Adam(lstm.parameters(), lr=1e-3)
        best_val, patience_cnt = float('inf'), 0
        best_state = None
        for ep in range(cfg["RW_EPOCHS"]):
            lstm.train()
            for bx, by in lstm_tr:
                pred = lstm(bx.to(device))
                loss = nn.functional.mse_loss(pred, by.to(device))
                opt_l.zero_grad(); loss.backward(); opt_l.step()
            lstm.eval()
            vl = sum(nn.functional.mse_loss(lstm(bx.to(device)), by.to(device)).item()
                     for bx, by in lstm_va)
            if vl < best_val:
                best_val = vl; patience_cnt = 0
                best_state = {k: v.clone() for k, v in lstm.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= cfg["RW_PATIENCE"]:
                    break
        if best_state:
            lstm.load_state_dict(best_state)
        lstm.eval()
        with torch.no_grad():
            r_pred_lstm_sc = lstm(to_t(X_te_s).to(device)).cpu().numpy().flatten()
        r_pred_lstm = sc_y.inverse_transform(r_pred_lstm_sc.reshape(-1,1)).flatten()[:n_t]
        pred_p_lstm = base_p * np.exp(r_pred_lstm)
        mae_lstm = mean_absolute_error(true_p, pred_p_lstm)
        da_lstm = np.mean((r_pred_lstm * r_true) > 0) * 100
        abs_err_lstm = np.abs(true_p - pred_p_lstm)

        print(f"  MoE     DA={da_moe:.1f}%  MAE={mae_moe:.4f}")
        print(f"  XGBoost DA={da_xgb:.1f}%  MAE={mae_xgb:.4f}")
        print(f"  LSTM    DA={da_lstm:.1f}%  MAE={mae_lstm:.4f}")

        results.append({
            'window': win_id, 'start': start, 'end': e_test,
            'mae_moe': mae_moe, 'da_moe': da_moe,
            'mae_xgb': mae_xgb, 'da_xgb': da_xgb,
            'mae_lstm': mae_lstm, 'da_lstm': da_lstm,
            'abs_err_moe': abs_err_moe, 'abs_err_xgb': abs_err_xgb, 'abs_err_lstm': abs_err_lstm,
        })

        start += rw_step

    if not results:
        print("No complete windows found!")
        return

    df_res = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  ROLLING WINDOW V2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Windows: {len(results)}")
    for name in ['moe', 'xgb', 'lstm']:
        das = df_res[f'da_{name}']
        maes = df_res[f'mae_{name}']
        print(f"  {name.upper():6s}  DA: {das.mean():.1f}% ± {das.std():.1f}%  "
              f"(best={das.max():.1f}%, worst={das.min():.1f}%)  MAE: {maes.mean():.4f}")

    # ── Improved DM Test with Newey-West ──
    print(f"\n{'='*60}")
    print(f"  IMPROVED DIEBOLD-MARIANO TEST (Newey-West, lag=√n)")
    print(f"{'='*60}")

    def dm_test_nw(e1, e2, lag=None):
        """
        Diebold-Mariano test with Newey-West HAC variance estimator.
        lag: if None, use sqrt(n) as per Newey-West optimal lag selection.
        """
        d = e1 - e2
        n = len(d)
        if n < 3:
            return np.nan, np.nan, lag
        if lag is None:
            lag = int(np.sqrt(n))
        d_mean = np.mean(d)
        # Newey-West variance estimator
        gamma0 = np.sum((d - d_mean)**2) / n
        gamma_sum = gamma0
        for j in range(1, lag + 1):
            w = 1 - j / (lag + 1)  # Bartlett kernel
            gamma_j = np.sum((d[j:] - d_mean) * (d[:-j] - d_mean)) / n
            gamma_sum += 2 * w * gamma_j
        var_d = gamma_sum / n
        dm_stat = d_mean / np.sqrt(var_d) if var_d > 0 else 0.0
        p_val = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
        return dm_stat, p_val, lag

    dm_results = []
    pairs = [
        ('moe', 'xgb', 'MoE vs XGBoost'),
        ('moe', 'lstm', 'MoE vs LSTM'),
        ('xgb', 'lstm', 'XGBoost vs LSTM'),
    ]

    for name1, name2, label in pairs:
        print(f"\n  {label}:")
        for _, row in df_res.iterrows():
            e1 = np.array(row[f'abs_err_{name1}'])
            e2 = np.array(row[f'abs_err_{name2}'])
            dm_stat, p_val, lag_used = dm_test_nw(e1, e2)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    Window {int(row['window'])}: DM={dm_stat:.4f}, p={p_val:.4f} {sig}  (lag={lag_used})")
            dm_results.append({
                'window': int(row['window']),
                'comparison': label,
                'model_1': name1, 'model_2': name2,
                'dm_stat': dm_stat, 'p_value': p_val,
                'lag': lag_used, 'significant': sig,
            })

    # Save v2 results
    save_df = df_res.drop(columns=['abs_err_moe', 'abs_err_xgb', 'abs_err_lstm'])
    save_df.to_csv(os.path.join(output_dir, "rolling_window_results_v2.csv"), index=False)
    print(f"\nV2 results saved to {output_dir}/rolling_window_results_v2.csv")

    df_dm = pd.DataFrame(dm_results)
    df_dm.to_csv(os.path.join(output_dir, "rolling_dm_test_v2.csv"), index=False)
    print(f"V2 DM test saved to {output_dir}/rolling_dm_test_v2.csv")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows: {len(results)}")
    print(f"  Config: RW_TRAIN={rw_train}, RW_VAL={rw_val}, RW_TEST={rw_test}, RW_STEP={rw_step}")
    for name in ['moe', 'xgb', 'lstm']:
        das = df_res[f'da_{name}']
        maes = df_res[f'mae_{name}']
        print(f"  {name.upper():6s}  Avg DA: {das.mean():.1f}%  Avg MAE: {maes.mean():.4f}")

    # DM summary per pair
    for name1, name2, label in pairs:
        pair_dm = [r for r in dm_results if r['model_1'] == name1 and r['model_2'] == name2]
        if pair_dm:
            mean_dm = np.mean([r['dm_stat'] for r in pair_dm])
            mean_p = np.mean([r['p_value'] for r in pair_dm])
            n_sig = sum(1 for r in pair_dm if r['p_value'] < 0.05)
            print(f"  DM {label}: mean_stat={mean_dm:.4f}, mean_p={mean_p:.4f}, significant_at_0.05={n_sig}/{len(pair_dm)}")


if __name__ == "__main__":
    patched_rw(m.CONFIG)
