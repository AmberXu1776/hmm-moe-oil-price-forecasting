"""
hmm_lstm_baseline.py — HMM+LSTM baseline for reviewer response
================================================================
Uses HMM regime probabilities as extra features fed into a single LSTM.
Follows the same rolling-window framework as main.py.
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─── Same CONFIG as main.py ─────────────────────────────────────
CONFIG = {
    "FILE_PATH": r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv",
    "FEAT_COLS": [
        "OPEC", "Brent", "WTI",
        "USDCNY", "Dollar_index", "US2Y",
        "PMI_China", "PMI_US",
        "DJIA", "SP500", "VIX",
        "GPR",
        "Shengli",
        "Excavator", "Excavator_YoY",
        "M2_M1_Spread",
    ],
    "TARGET_COL": "Daqing",
    "N_REGIMES": 3,
    "HMM_MODE": "multi",
    "N_EXPERTS": 3,
    "SEQ_LEN": 52,
    "D_MODEL": 32,
    "D_STATE": 16,
    "N_LAYERS": 1,
    "DROPOUT": 0.55,
    "GATE_HIDDEN_DIM": 16,
    "GATE_WINDOW": 4,
    "VOL_WINDOW": 12,
    "LR": 1e-4,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "BALANCE_WEIGHT": 0.01,
    "LOSS_ALPHA": 1.0,
    "LOSS_BETA": 0.01,
    "SEED": 42,
    "OUTPUT_DIR": "output",
    "RW_TRAIN": 700,
    "RW_VAL": 100,
    "RW_TEST": 100,
    "RW_STEP": 50,
    "RW_EPOCHS": 50,
    "RW_PATIENCE": 10,
}


class SimpleLSTM(nn.Module):
    def __init__(self, inp, hid=64, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_simple_lstm(model, tr_ld, va_ld, epochs, patience, device, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, patience_cnt = float('inf'), 0
    best_state = None
    for ep in range(epochs):
        model.train()
        for bx, by in tr_ld:
            pred = model(bx.to(device))
            loss = nn.functional.mse_loss(pred, by.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = sum(nn.functional.mse_loss(model(bx.to(device)), by.to(device)).item()
                 for bx, by in va_ld)
        if vl < best_val:
            best_val = vl; patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_hmm_lstm_baseline(cfg):
    from layer4_mamba_moe import (
        OilRegimeDetector, set_seed,
        build_log_returns, build_gate_features,
    )

    output_dir = os.path.join(SCRIPT_DIR, cfg["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HMM+LSTM Baseline] Device: {device}")

    # ── Load & clean data ──
    df = pd.read_csv(cfg["FILE_PATH"]).sort_values("date").reset_index(drop=True)
    all_cols = cfg["FEAT_COLS"] + [cfg["TARGET_COL"]]
    df[all_cols] = df[all_cols].interpolate(method="linear")
    for col in all_cols:
        mv, sv = df[col].mean(), df[col].std()
        outliers = (df[col] < mv - 3*sv) | (df[col] > mv + 3*sv)
        df.loc[outliers, col] = df[col].median()

    # ── Build log-returns ──
    lr_df, all_feat_cols, _, _, original_prices = build_log_returns(
        df.copy(), cfg["FEAT_COLS"], cfg["TARGET_COL"])

    # ── Full HMM ──
    regime_detector = OilRegimeDetector(
        n_regimes=cfg["N_REGIMES"], covariance_type="diag",
        n_init=15, min_covar=1e-6)
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, cfg["FEAT_COLS"], cfg["TARGET_COL"],
        vol_window=cfg["VOL_WINDOW"], hmm_mode=cfg["HMM_MODE"])
    regime_detector.fit(hmm_features)
    regime_proba_raw, _ = regime_detector.predict_proba(hmm_features)

    # Align
    T_hmm = len(regime_proba_raw)
    T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)
    regime_proba = regime_proba_raw[-T_aligned:]
    lr_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    orig_p = original_prices[-T_aligned - 1:]
    n_regimes = cfg["N_REGIMES"]
    n_raw = len(cfg["FEAT_COLS"])
    seq_len = cfg["SEQ_LEN"]

    # ── Build sliding windows with regime features ──
    # Key difference: X_seq_augmented = [original_features | regime_probabilities]
    X_seq_all, Y_all = [], []
    for i in range(seq_len, T_aligned):
        raw_feats = lr_aligned[all_feat_cols].iloc[i - seq_len:i].values  # (seq_len, F)
        # Regime probs for this window: (seq_len, n_regimes)
        rp = regime_proba[i - seq_len:i]
        # Concatenate: (seq_len, F + n_regimes)
        augmented = np.concatenate([raw_feats, rp], axis=1)
        X_seq_all.append(augmented)
        Y_all.append(lr_aligned["target_scaled"].iloc[i])
    X_seq_all = np.array(X_seq_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    N_total = len(Y_all)
    n_features_raw = X_seq_all.shape[2]  # F + n_regimes

    # ── Rolling window ──
    rw_train = cfg["RW_TRAIN"]
    rw_val = cfg["RW_VAL"]
    rw_test = cfg["RW_TEST"]
    rw_step = cfg["RW_STEP"]
    window_size = rw_train + rw_val + rw_test

    print(f"  Total samples: {N_total}, Augmented features: {n_features_raw} "
          f"(raw {len(all_feat_cols)} + regime {n_regimes})")
    print(f"  Windows: ~{(N_total - window_size) // rw_step + 1}")

    # Also load existing results to compare
    existing_csv = os.path.join(output_dir, "rolling_window_results.csv")
    if os.path.exists(existing_csv):
        df_existing = pd.read_csv(existing_csv)
        print(f"  Found existing results: {len(df_existing)} windows")
    else:
        df_existing = None

    results = []
    start = 0
    win_id = 0

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)

    while start + window_size <= N_total:
        win_id += 1
        s_test = start + rw_train + rw_val
        e_test = s_test + rw_test
        print(f"\n--- Window {win_id}: [{start}:{s_test}] train+val, [{s_test}:{e_test}] test ---")

        X_tr = X_seq_all[start:start + rw_train]
        Y_tr = Y_all[start:start + rw_train]
        X_va = X_seq_all[start + rw_train:s_test]
        Y_va = Y_all[start + rw_train:s_test]
        X_te = X_seq_all[s_test:e_test]
        Y_te = Y_all[s_test:e_test]

        # Per-window scaling
        sc_feat = StandardScaler()
        flat_tr = X_tr.reshape(-1, n_features_raw)
        sc_feat.fit(flat_tr)
        X_tr_s = sc_feat.transform(flat_tr).reshape(X_tr.shape).astype(np.float32)
        X_va_s = sc_feat.transform(X_va.reshape(-1, n_features_raw)).reshape(X_va.shape).astype(np.float32)
        X_te_s = sc_feat.transform(X_te.reshape(-1, n_features_raw)).reshape(X_te.shape).astype(np.float32)

        sc_y = StandardScaler()
        Y_tr_s = sc_y.fit_transform(Y_tr.reshape(-1, 1)).flatten().astype(np.float32)
        Y_va_s = sc_y.transform(Y_va.reshape(-1, 1)).flatten().astype(np.float32)
        Y_te_s = sc_y.transform(Y_te.reshape(-1, 1)).flatten().astype(np.float32)

        tr_ld = DataLoader(TensorDataset(to_t(X_tr_s), to_t(Y_tr_s).unsqueeze(1)),
                           batch_size=32, shuffle=True)
        va_ld = DataLoader(TensorDataset(to_t(X_va_s), to_t(Y_va_s).unsqueeze(1)),
                           batch_size=32)

        # ── Train HMM+LSTM ──
        set_seed(cfg["SEED"])
        lstm = SimpleLSTM(n_features_raw, hid=64, layers=2, drop=0.2).to(device)
        lstm = train_simple_lstm(lstm, tr_ld, va_ld,
                                 epochs=cfg["RW_EPOCHS"],
                                 patience=cfg["RW_PATIENCE"],
                                 device=device, lr=1e-3)

        # ── Evaluate ──
        lstm.eval()
        with torch.no_grad():
            r_pred_lstm = lstm(to_t(X_te_s).to(device)).cpu().numpy().flatten()
        r_pred_lstm = sc_y.inverse_transform(r_pred_lstm.reshape(-1, 1)).flatten()
        r_true = sc_y.inverse_transform(Y_te_s.reshape(-1, 1)).flatten()

        # Prices
        price_offset = seq_len + s_test
        base_p = orig_p[price_offset:price_offset + rw_test]
        true_p = orig_p[price_offset + 1:price_offset + 1 + rw_test]
        n_t = min(len(r_pred_lstm), len(base_p), len(true_p))
        r_pred_lstm = r_pred_lstm[:n_t]
        r_true = r_true[:n_t]
        base_p = base_p[:n_t]
        true_p = true_p[:n_t]

        pred_p = base_p * np.exp(r_pred_lstm)
        mae = mean_absolute_error(true_p, pred_p)
        da = np.mean((r_pred_lstm * r_true) > 0) * 100

        print(f"  HMM+LSTM  DA={da:.1f}%  MAE={mae:.4f}")

        results.append({
            'window': win_id, 'start': start, 'end': e_test,
            'mae_hmm_lstm': mae, 'da_hmm_lstm': da,
        })

        start += rw_step

    if not results:
        print("No complete windows!")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, "hmm_lstm_baseline.csv"), index=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  HMM+LSTM BASELINE SUMMARY")
    print(f"{'='*60}")
    das = df_res['da_hmm_lstm']
    maes = df_res['mae_hmm_lstm']
    print(f"  HMM+LSTM  DA: {das.mean():.1f}% ± {das.std():.1f}%  MAE: {maes.mean():.4f}")

    # ── Compare with existing results ──
    if df_existing is not None:
        print(f"\n--- Comparison with existing models ---")
        # Merge by window
        merged = df_res.merge(df_existing, on='window', how='inner')
        if len(merged) > 0:
            print(f"  {'Model':12s}  {'DA (mean)':>10s}  {'MAE (mean)':>10s}")
            print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
            for col_name, label in [('da_moe', 'HMM-MoE'), ('da_xgb', 'XGBoost'),
                                     ('da_lstm', 'LSTM'), ('da_hmm_lstm', 'HMM+LSTM')]:
                if col_name in merged.columns:
                    mae_col = col_name.replace('da_', 'mae_')
                    da_mean = merged[col_name].mean()
                    mae_mean = merged[mae_col].mean()
                    print(f"  {label:12s}  {da_mean:9.1f}%  {mae_mean:10.4f}")

            # DM test: HMM-MoE vs HMM+LSTM
            print(f"\n--- DM Test: HMM-MoE vs HMM+LSTM ---")
            from scipy import stats
            for _, row in merged.iterrows():
                # Use MAE difference as proxy (no raw errors, use per-window MAE diff)
                pass
            # Per-window comparison
            mae_diff = merged['mae_moe'].values - merged['mae_hmm_lstm'].values
            n = len(mae_diff)
            if n >= 3:
                dm_stat = np.mean(mae_diff) / (np.std(mae_diff, ddof=1) / np.sqrt(n))
                p_val = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
                print(f"  DM stat={dm_stat:.4f}, p={p_val:.4f} {sig}")
                if dm_stat < 0:
                    print(f"  → HMM-MoE has lower MAE (better)")
                else:
                    print(f"  → HMM+LSTM has lower MAE (better)")

    print(f"\nResults saved to {output_dir}/hmm_lstm_baseline.csv")


if __name__ == "__main__":
    run_hmm_lstm_baseline(CONFIG)
