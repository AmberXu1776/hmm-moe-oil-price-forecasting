"""rolling_window_v2.py — 扩展rolling window + Newey-West DM test"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
warnings.filterwarnings("ignore")

# Patch JSON
_orig = json.JSONEncoder.default
def _jp(self, obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return _orig(self, obj)
json.JSONEncoder.default = _jp

from layer4_mamba_moe import (
    OilMoE, OilRegimeDetector, set_seed,
    build_log_returns, build_gate_features,
    train_moe_model, evaluate_moe_model,
)

FILE_PATH = r"D:\阶梯计划\论文\P1_MambaMoE_OilPrice\data\大杂烩_扩展版.csv"
FEAT_COLS = ["OPEC","Brent","WTI","USDCNY","Dollar_index","US2Y",
             "PMI_China","PMI_US","DJIA","SP500","VIX","GPR","Shengli",
             "Excavator","Excavator_YoY","M2_M1_Spread"]
TARGET_COL = "Daqing"
N_REGIMES = 3; SEQ_LEN = 52; N_EXPERTS = 3; D_MODEL = 32; D_STATE = 16
N_LAYERS = 1; DROPOUT = 0.55; GATE_HIDDEN = 16; GATE_WIN = 4; VOL_WIN = 12
LR = 1e-4; BALANCE = 0.01; ALPHA = 1.0; BETA = 0.01
SEED = 42

# New params: more windows
RW_TRAIN = 600; RW_VAL = 80; RW_TEST = 80; RW_STEP = 40
RW_EPOCHS = 40; RW_PATIENCE = 8

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(FILE_PATH).sort_values("date").reset_index(drop=True)
    all_cols = FEAT_COLS + [TARGET_COL]
    df[all_cols] = df[all_cols].interpolate(method="linear")
    for col in all_cols:
        mv, sv = df[col].mean(), df[col].std()
        out = (df[col] < mv - 3*sv) | (df[col] > mv + 3*sv)
        df.loc[out, col] = df[col].median()

    lr_df, all_feat_cols, _, _, original_prices = build_log_returns(
        df.copy(), FEAT_COLS, TARGET_COL)

    # HMM
    regime_detector = OilRegimeDetector(n_regimes=N_REGIMES, covariance_type="diag",
                                         n_init=15, min_covar=1e-6)
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, FEAT_COLS, TARGET_COL, vol_window=VOL_WIN, hmm_mode="multi")
    regime_detector.fit(hmm_features)
    regime_proba_raw, _ = regime_detector.predict_proba(hmm_features)

    T_hmm = len(regime_proba_raw); T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)
    regime_proba = regime_proba_raw[-T_aligned:]
    lr_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    orig_p = original_prices[-T_aligned - 1:]
    n_raw = len(FEAT_COLS); gate_window = GATE_WIN; vol_window = VOL_WIN

    # Build windows
    X_seq_all, X_gate_all, Y_all = [], [], []
    for i in range(SEQ_LEN, T_aligned):
        X_seq_all.append(lr_aligned[all_feat_cols].iloc[i - SEQ_LEN:i].values)
        X_gate_all.append(build_gate_features(
            regime_proba, lr_aligned, all_feat_cols,
            TARGET_COL, n_raw, i, gate_window, vol_window))
        Y_all.append(lr_aligned["target_scaled"].iloc[i])
    X_seq_all = np.array(X_seq_all, dtype=np.float32)
    X_gate_all = np.array(X_gate_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    scaler_gate = StandardScaler()
    X_gate_all = scaler_gate.fit_transform(X_gate_all).astype(np.float32)

    N_total = len(Y_all)
    n_features = X_seq_all.shape[2]
    gate_dim = X_gate_all.shape[1]
    window_size = RW_TRAIN + RW_VAL + RW_TEST

    n_windows = (N_total - window_size) // RW_STEP + 1
    print(f"Total samples: {N_total}, Window: {window_size}, Step: {RW_STEP}")
    print(f"Number of windows: {n_windows}")

    results = []
    start = 0; win_id = 0

    while start + window_size <= N_total:
        win_id += 1
        s_test = start + RW_TRAIN + RW_VAL
        e_test = s_test + RW_TEST
        print(f"\n--- Window {win_id} ---")

        X_tr = X_seq_all[start:start + RW_TRAIN]
        Xg_tr = X_gate_all[start:start + RW_TRAIN]
        Y_tr = Y_all[start:start + RW_TRAIN]
        X_va = X_seq_all[start + RW_TRAIN:s_test]
        Xg_va = X_gate_all[start + RW_TRAIN:s_test]
        Y_va = Y_all[start + RW_TRAIN:s_test]
        X_te = X_seq_all[s_test:e_test]
        Xg_te = X_gate_all[s_test:e_test]
        Y_te = Y_all[s_test:e_test]

        # Per-window scaling
        sc_feat = StandardScaler()
        X_tr_s = sc_feat.fit_transform(X_tr.reshape(-1, n_features)).reshape(X_tr.shape).astype(np.float32)
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
        def mk_loader_seq(xs, y, shuffle=False):
            return DataLoader(TensorDataset(to_t(xs), to_t(y).unsqueeze(1)),
                              batch_size=32, shuffle=shuffle)

        tr_ld = mk_loader(X_tr_s, Xg_tr_s, Y_tr_s, shuffle=True)
        va_ld = mk_loader(X_va_s, Xg_va_s, Y_va_s)
        te_ld = mk_loader(X_te_s, Xg_te_s, Y_te_s)

        # Price alignment
        price_offset = SEQ_LEN + s_test
        base_p = orig_p[price_offset:price_offset + RW_TEST]
        true_p = orig_p[price_offset + 1:price_offset + 1 + RW_TEST]

        # --- HMM-MoE ---
        set_seed(SEED)
        model = OilMoE(
            n_features=n_features, gate_input_dim=gate_dim, n_experts=N_EXPERTS,
            d_model=D_MODEL, pred_len=1, gate_hidden_dim=GATE_HIDDEN,
            expert_configs=[{"input_dim": n_features, "d_model": D_MODEL,
                             "d_state": D_STATE, "n_layers": N_LAYERS, "dropout": DROPOUT}
                            for _ in range(N_EXPERTS)])
        model, _ = train_moe_model(model, tr_ld, va_ld, epochs=RW_EPOCHS, lr=LR,
                                    balance_weight=BALANCE, patience=RW_PATIENCE,
                                    device=device, loss_alpha=ALPHA, loss_beta=BETA)
        model.eval()
        r_pred_list, r_true_list = [], []
        with torch.no_grad():
            for bx, bg, by in te_ld:
                pred, _ = model(bx.to(device), bg.to(device))
                r_pred_list.append(pred.cpu().numpy().flatten())
                r_true_list.append(by.numpy().flatten())
        r_pred_moe = sc_y.inverse_transform(np.concatenate(r_pred_list).reshape(-1,1)).flatten()
        r_true = sc_y.inverse_transform(np.concatenate(r_true_list).reshape(-1,1)).flatten()

        n_t = min(len(r_pred_moe), len(base_p), len(true_p))
        r_pred_moe = r_pred_moe[:n_t]; r_true = r_true[:n_t]
        base_p = base_p[:n_t]; true_p = true_p[:n_t]
        pred_p_moe = base_p * np.exp(r_pred_moe)
        mae_moe = mean_absolute_error(true_p, pred_p_moe)
        da_moe = np.mean((r_pred_moe * r_true) > 0) * 100
        abs_err_moe = np.abs(true_p - pred_p_moe)

        # --- XGBoost ---
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED)
        xgb.fit(X_tr_s[:, -1, :], Y_tr_s)
        r_pred_xgb = sc_y.inverse_transform(xgb.predict(X_te_s[:, -1, :]).reshape(-1,1)).flatten()[:n_t]
        pred_p_xgb = base_p * np.exp(r_pred_xgb)
        mae_xgb = mean_absolute_error(true_p, pred_p_xgb)
        da_xgb = np.mean((r_pred_xgb * r_true) > 0) * 100
        abs_err_xgb = np.abs(true_p - pred_p_xgb)

        # --- LSTM ---
        class SimpleLSTM(nn.Module):
            def __init__(self, inp, hid=64, layers=2, drop=0.2):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
                self.fc = nn.Linear(hid, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        set_seed(SEED)
        lstm = SimpleLSTM(n_features).to(device)
        lstm_tr = mk_loader_seq(X_tr_s, Y_tr_s, shuffle=True)
        lstm_va = mk_loader_seq(X_va_s, Y_va_s)
        opt_l = torch.optim.Adam(lstm.parameters(), lr=1e-3)
        best_val, pc = float('inf'), 0
        for ep in range(RW_EPOCHS):
            lstm.train()
            for bx, by in lstm_tr:
                pred = lstm(bx.to(device))
                loss = nn.functional.mse_loss(pred, by.to(device))
                opt_l.zero_grad(); loss.backward(); opt_l.step()
            lstm.eval()
            vl = sum(nn.functional.mse_loss(lstm(bx.to(device)), by.to(device)).item()
                     for bx, by in lstm_va)
            if vl < best_val:
                best_val = vl; pc = 0
                best_state = {k: v.clone() for k, v in lstm.state_dict().items()}
            else:
                pc += 1
                if pc >= RW_PATIENCE: break
        lstm.load_state_dict(best_state)
        lstm.eval()
        with torch.no_grad():
            r_pred_lstm = sc_y.inverse_transform(
                lstm(to_t(X_te_s).to(device)).cpu().numpy()).flatten()[:n_t]
        pred_p_lstm = base_p * np.exp(r_pred_lstm)
        mae_lstm = mean_absolute_error(true_p, pred_p_lstm)
        da_lstm = np.mean((r_pred_lstm * r_true) > 0) * 100
        abs_err_lstm = np.abs(true_p - pred_p_lstm)

        print(f"  MoE     DA={da_moe:.1f}%  MAE={mae_moe:.2f}")
        print(f"  XGBoost DA={da_xgb:.1f}%  MAE={mae_xgb:.2f}")
        print(f"  LSTM    DA={da_lstm:.1f}%  MAE={mae_lstm:.2f}")

        results.append({
            'window': win_id, 'start': start, 'end': e_test,
            'mae_moe': mae_moe, 'da_moe': da_moe,
            'mae_xgb': mae_xgb, 'da_xgb': da_xgb,
            'mae_lstm': mae_lstm, 'da_lstm': da_lstm,
            'abs_err_moe': abs_err_moe.tolist(),
            'abs_err_xgb': abs_err_xgb.tolist(),
            'abs_err_lstm': abs_err_lstm.tolist(),
        })
        start += RW_STEP

    if not results:
        print("No windows!"); return

    df_res = pd.DataFrame(results)
    print(f"\n{'='*60}\n  SUMMARY ({len(results)} windows)\n{'='*60}")
    for name in ['moe', 'xgb', 'lstm']:
        das = df_res[f'da_{name}']; maes = df_res[f'mae_{name}']
        print(f"  {name.upper():6s} DA: {das.mean():.1f}% ± {das.std():.1f}% "
              f"(best={das.max():.1f} worst={das.min():.1f})  MAE: {maes.mean():.2f}")

    # DM test with Newey-West
    print(f"\n--- Diebold-Mariano (Newey-West, l=sqrt(n)) ---")
    dm_results = []
    for _, row in df_res.iterrows():
        d = np.array(row['abs_err_moe']) - np.array(row['abs_err_xgb'])
        n = len(d)
        if n < 3: continue
        # Newey-West variance estimator
        l_nw = int(np.sqrt(n))  # lag order
        gamma_0 = np.var(d, ddof=0)
        gamma_sum = 0
        for j in range(1, l_nw + 1):
            w = 1 - j / (l_nw + 1)  # Bartlett kernel
            gamma_j = np.mean(d[j:] * d[:-j])
            gamma_sum += 2 * w * gamma_j
        nw_var = gamma_0 + gamma_sum
        if nw_var <= 0: nw_var = np.var(d, ddof=1)
        dm_stat = np.mean(d) / np.sqrt(nw_var / n)
        p_val = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))  # asymptotic normal
        dm_results.append({'window': row['window'], 'dm_stat': dm_stat,
                           'p_value': p_val, 'nw_lag': l_nw})
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  Window {int(row['window'])}: DM={dm_stat:.4f}, p={p_val:.4f} (NW lag={l_nw}) {sig}")

    if dm_results:
        df_dm = pd.DataFrame(dm_results)
        print(f"\n  Mean DM={df_dm['dm_stat'].mean():.4f}, Mean p={df_dm['p_value'].mean():.4f}")

    save_df = df_res.drop(columns=['abs_err_moe', 'abs_err_xgb', 'abs_err_lstm'])
    save_df.to_csv(os.path.join(output_dir, "rolling_window_results_v2.csv"), index=False)
    if dm_results:
        pd.DataFrame(dm_results).to_csv(os.path.join(output_dir, "rolling_dm_test_v2.csv"), index=False)
    print(f"\nSaved to {output_dir}/")

if __name__ == "__main__":
    main()
