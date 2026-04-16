"""
main.py — Mamba-MoE 油价预测（HMM + MoE 管道）
=================================================
基于花旗杯 layer4_mamba_moe.py 核心代码，只跑 Layer 3+4。
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import time
import warnings
import json
import numpy as np
import pandas as pd
import torch

# Patch JSON to handle numpy types (layer4_mamba_moe.py saves metrics internally)
_original_default = json.JSONEncoder.default
def _json_default(self, obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _original_default(self, obj)
json.JSONEncoder.default = _json_default

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────
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
    "RUN_SINGLE": False,
    "RUN_ROLLING_WINDOW": True,
    "RW_TRAIN": 700,
    "RW_VAL": 100,
    "RW_TEST": 100,
    "RW_STEP": 50,
    "RW_EPOCHS": 50,
    "RW_PATIENCE": 10,
}


def main():
    cfg = CONFIG
    output_dir = os.path.join(SCRIPT_DIR, cfg["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ── imports from layer4_mamba_moe ──
    from layer4_mamba_moe import (
        OilMoE, set_seed,
        oil_price_data_preprocess_v6,
        train_moe_model, evaluate_moe_model,
        plot_regime_timeline, plot_gate_weights,
        plot_prediction_vs_actual, plot_training_history,
    )

    set_seed(cfg["SEED"])

    # ── 1. 数据预处理 ──
    t0 = time.time()
    (train_loader, val_loader, test_loader,
     scaler_feat, scaler_target, scaler_gate,
     all_feat_cols, gate_dim,
     regime_detector, regime_proba, regime_labels,
     test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
        file_path=cfg["FILE_PATH"],
        feat_cols=cfg["FEAT_COLS"],
        target_col=cfg["TARGET_COL"],
        seq_len=cfg["SEQ_LEN"],
        n_regimes=cfg["N_REGIMES"],
        gate_window=cfg["GATE_WINDOW"],
        vol_window=cfg["VOL_WINDOW"],
        hmm_mode=cfg["HMM_MODE"],
    )
    print(f"Preprocess done in {time.time()-t0:.1f}s")

    # Regime timeline plot
    try:
        df_raw = pd.read_csv(cfg["FILE_PATH"]).sort_values("date").reset_index(drop=True)
        df_raw[cfg["FEAT_COLS"] + [cfg["TARGET_COL"]]] = \
            df_raw[cfg["FEAT_COLS"] + [cfg["TARGET_COL"]]].interpolate()
        oil_full = df_raw[cfg["TARGET_COL"]].values
        plot_len = min(len(regime_labels), len(oil_full))
        plot_regime_timeline(
            dates=np.arange(plot_len),
            oil_prices=oil_full[-plot_len:],
            regime_labels=regime_labels[-plot_len:],
            n_regimes=cfg["N_REGIMES"],
            save_path=os.path.join(output_dir, "L3_regime_timeline.png"),
        )
    except Exception as e:
        print(f"Regime plot failed: {e}")

    # ── 2. 构建 MoE ──
    bx, bg, _ = next(iter(train_loader))
    n_features = bx.shape[2]

    model = OilMoE(
        n_features=n_features,
        gate_input_dim=gate_dim,
        n_experts=cfg["N_EXPERTS"],
        d_model=cfg["D_MODEL"],
        pred_len=1,
        gate_hidden_dim=cfg["GATE_HIDDEN_DIM"],
        expert_configs=[
            {
                "input_dim": n_features,
                "d_model": cfg["D_MODEL"],
                "d_state": cfg["D_STATE"],
                "n_layers": cfg["N_LAYERS"],
                "dropout": cfg["DROPOUT"],
            }
            for _ in range(cfg["N_EXPERTS"])
        ],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MoE params: {total_params:,d}")

    # ── 3. 训练 ──
    model, history = train_moe_model(
        model, train_loader, val_loader,
        epochs=cfg["EPOCHS"], lr=cfg["LR"],
        balance_weight=cfg["BALANCE_WEIGHT"],
        patience=cfg["PATIENCE"],
        device=device,
        loss_alpha=cfg["LOSS_ALPHA"],
        loss_beta=cfg["LOSS_BETA"],
    )

    # ── 4. 评估 ──
    pred_p, true_p, r_pred, r_true, gate_weights, metrics = evaluate_moe_model(
        model, test_loader, scaler_target,
        test_base_prices, test_true_prices, device=device,
    )

    # ── 5. 可视化 + 保存 ──
    plot_gate_weights(dates_test, gate_weights, cfg["N_EXPERTS"],
                      save_path=os.path.join(output_dir, "L4_gate_weights.png"))
    plot_prediction_vs_actual(dates_test, pred_p, true_p,
                              save_path=os.path.join(output_dir, "L4_prediction.png"))
    plot_training_history(history,
                          save_path=os.path.join(output_dir, "L4_training.png"))

    # Use short path for torch.save (Chinese path can cause issues)
    import shutil
    tmp_model = os.path.join(os.environ.get('TEMP', '.'), 'mamba_moe.pth')
    torch.save(model.state_dict(), tmp_model)
    shutil.copy2(tmp_model, os.path.join(output_dir, "mamba_moe.pth"))

    pd.Series(metrics).to_csv(os.path.join(output_dir, "metrics.csv"))

    # ── 6. 汇总报告 ──
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Direction Accuracy : {metrics['dir_acc']:.1f}%")
    print(f"  MAE (MoE)          : {metrics['mae_model']:.4f}")
    print(f"  MAE (Random Walk)  : {metrics['mae_rw']:.4f}")
    print(f"  vs RW              : {'BEAT' if metrics['beat_rw'] else 'NOT BEAT'} ({metrics['pct_vs_rw']:+.2f}%)")
    print(f"  MAPE               : {metrics['mape_model']:.2f}%")
    print(f"  Return Corr        : {metrics['r_corr']:.4f}")
    print(f"  Test samples       : {metrics['n_test']}")
    print(f"  Output             : {output_dir}/")
    print("=" * 60)

    return metrics


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ROLLING WINDOW VALIDATION                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_rolling_window(cfg):
    """Rolling window validation with HMM-MoE, XGBoost, LSTM baselines."""
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

    # ── Load & clean data (same as oil_price_data_preprocess_v6) ──
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

    # ── Full HMM on all data ──
    regime_detector = OilRegimeDetector(
        n_regimes=cfg["N_REGIMES"], covariance_type="diag",
        n_init=15, min_covar=1e-6)
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, cfg["FEAT_COLS"], cfg["TARGET_COL"],
        vol_window=cfg["VOL_WINDOW"], hmm_mode=cfg["HMM_MODE"])
    regime_detector.fit(hmm_features)
    regime_proba_raw, _ = regime_detector.predict_proba(hmm_features)

    # Align regime_proba with lr_df
    T_hmm = len(regime_proba_raw)
    T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)
    regime_proba = regime_proba_raw[-T_aligned:]
    lr_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    orig_p = original_prices[-T_aligned - 1:]  # +1 for base->true shift
    n_regimes = cfg["N_REGIMES"]
    n_raw = len(cfg["FEAT_COLS"])
    seq_len = cfg["SEQ_LEN"]
    gate_window = cfg["GATE_WINDOW"]
    vol_window = cfg["VOL_WINDOW"]

    # ── Build sliding windows (same as preprocess_v6) ──
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

    # Scale gate features
    scaler_gate = StandardScaler()
    X_gate_all = scaler_gate.fit_transform(X_gate_all).astype(np.float32)

    N_total = len(Y_all)
    n_features = X_seq_all.shape[2]
    gate_dim = X_gate_all.shape[1]

    # ── Rolling window parameters ──
    rw_train = cfg["RW_TRAIN"]
    rw_val = cfg["RW_VAL"]
    rw_test = cfg["RW_TEST"]
    rw_step = cfg["RW_STEP"]
    window_size = rw_train + rw_val + rw_test

    print(f"\n{'='*60}")
    print(f"  ROLLING WINDOW VALIDATION")
    print(f"  Total samples: {N_total}, Window: {window_size}, Step: {rw_step}")
    print(f"  Number of windows: ~{(N_total - window_size) // rw_step + 1}")
    print(f"{'='*60}")

    results = []
    start = 0
    win_id = 0

    while start + window_size <= N_total:
        win_id += 1
        s_test = start + rw_train + rw_val
        e_test = s_test + rw_test
        print(f"\n--- Window {win_id}: [{start}:{s_test}] train+val, [{s_test}:{e_test}] test ---")

        # ── Slice data ──
        X_tr = X_seq_all[start:start + rw_train]
        Xg_tr = X_gate_all[start:start + rw_train]
        Y_tr = Y_all[start:start + rw_train]
        X_va = X_seq_all[start + rw_train:s_test]
        Xg_va = X_gate_all[start + rw_train:s_test]
        Y_va = Y_all[start + rw_train:s_test]
        X_te = X_seq_all[s_test:e_test]
        Xg_te = X_gate_all[s_test:e_test]
        Y_te = Y_all[s_test:e_test]

        # ── Per-window scaling ──
        sc_feat = StandardScaler()
        flat_tr = X_tr.reshape(-1, n_features)
        sc_feat.fit(flat_tr)
        X_tr_s = sc_feat.transform(flat_tr).reshape(X_tr.shape).astype(np.float32)
        X_va_s = sc_feat.transform(X_va.reshape(-1, n_features)).reshape(X_va.shape).astype(np.float32)
        X_te_s = sc_feat.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape).astype(np.float32)

        # Target scaling
        sc_y = StandardScaler()
        Y_tr_s = sc_y.fit_transform(Y_tr.reshape(-1, 1)).flatten().astype(np.float32)
        Y_va_s = sc_y.transform(Y_va.reshape(-1, 1)).flatten().astype(np.float32)
        Y_te_s = sc_y.transform(Y_te.reshape(-1, 1)).flatten().astype(np.float32)

        # Gate scaling
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

        # ── HMM-MoE ──
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
            patience=cfg["RW_PATIENCE"],
            device=device,
            loss_alpha=cfg["LOSS_ALPHA"], loss_beta=cfg["LOSS_BETA"],
        )

        # Evaluate MoE
        model.eval()
        r_pred_list, r_true_list = [], []
        with torch.no_grad():
            for bx, bg, by in te_ld:
                pred, _ = model(bx.to(device), bg.to(device))
                r_pred_list.append(pred.cpu().numpy().flatten())
                r_true_list.append(by.numpy().flatten())
        r_pred_moe = sc_y.inverse_transform(np.concatenate(r_pred_list).reshape(-1,1)).flatten()
        r_true = sc_y.inverse_transform(np.concatenate(r_true_list).reshape(-1,1)).flatten()

        # Get base/true prices for this window
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

        # ── XGBoost baseline ──
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=cfg["SEED"])
        # Flatten seq dim: use last timestep features
        xgb_tr = X_tr_s[:, -1, :]  # (N, F)
        xgb_te = X_te_s[:, -1, :]
        # XGBoost predicts scaled target
        xgb.fit(xgb_tr, Y_tr_s)
        r_pred_xgb_sc = xgb.predict(xgb_te)
        r_pred_xgb = sc_y.inverse_transform(r_pred_xgb_sc.reshape(-1,1)).flatten()[:n_t]
        pred_p_xgb = base_p * np.exp(r_pred_xgb)
        mae_xgb = mean_absolute_error(true_p, pred_p_xgb)
        da_xgb = np.mean((r_pred_xgb * r_true) > 0) * 100
        abs_err_xgb = np.abs(true_p - pred_p_xgb)

        # ── LSTM baseline ──
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
        # LSTM uses (x_seq, y) without gate
        lstm_tr = DataLoader(TensorDataset(to_t(X_tr_s), to_t(Y_tr_s).unsqueeze(1)),
                             batch_size=32, shuffle=True)
        lstm_va = DataLoader(TensorDataset(to_t(X_va_s), to_t(Y_va_s).unsqueeze(1)),
                             batch_size=32)
        opt_l = torch.optim.Adam(lstm.parameters(), lr=1e-3)
        best_val, patience_cnt = float('inf'), 0
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
        lstm.load_state_dict(best_state)
        lstm.eval()
        with torch.no_grad():
            r_pred_lstm_sc = torch.cat([lstm(bx.to(device)) for bx, in [
                (to_t(X_te_s),)]]).cpu().numpy().flatten()
        # Actually just do it properly
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

    # ── Summary ──
    df_res = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  ROLLING WINDOW SUMMARY")
    print(f"{'='*60}")
    print(f"  Windows: {len(results)}")
    for name in ['moe', 'xgb', 'lstm']:
        das = df_res[f'da_{name}']
        maes = df_res[f'mae_{name}']
        print(f"  {name.upper():6s}  DA: {das.mean():.1f}% ± {das.std():.1f}%  "
              f"(best={das.max():.1f}%, worst={das.min():.1f}%)  MAE: {maes.mean():.4f}")

    # ── Diebold-Mariano Test ──
    print(f"\n--- Diebold-Mariano Tests ---")
    dm_results = []
    for _, row in df_res.iterrows():
        d = np.array(row['abs_err_moe']) - np.array(row['abs_err_xgb'])
        n = len(d)
        if n < 3:
            continue
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))
        from scipy import stats
        p_val = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))
        dm_results.append({'window': row['window'], 'dm_stat': dm_stat, 'p_value': p_val})

    if dm_results:
        df_dm = pd.DataFrame(dm_results)
        print(f"  MoE vs XGBoost: mean DM={df_dm['dm_stat'].mean():.4f}, "
              f"mean p={df_dm['p_value'].mean():.4f}")
        for _, r in df_dm.iterrows():
            sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
            print(f"    Window {int(r['window'])}: DM={r['dm_stat']:.4f}, p={r['p_value']:.4f} {sig}")
        df_dm.to_csv(os.path.join(output_dir, "rolling_dm_test.csv"), index=False)

    # Save main results (without raw error arrays)
    save_df = df_res.drop(columns=['abs_err_moe', 'abs_err_xgb', 'abs_err_lstm'])
    save_df.to_csv(os.path.join(output_dir, "rolling_window_results.csv"), index=False)
    print(f"\nResults saved to {output_dir}/rolling_window_results.csv")


if __name__ == "__main__":
    if CONFIG.get("RUN_SINGLE", True):
        main()
    if CONFIG.get("RUN_ROLLING_WINDOW"):
        run_rolling_window(CONFIG)
