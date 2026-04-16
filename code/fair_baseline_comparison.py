"""
fair_baseline_comparison.py — 公平baseline对比实验
所有模型统一使用direction-aware loss + 3-seed + 相同数据pipeline
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
SEEDS = [42, 123, 2024]
LOSS_ALPHA = 1.0  # direction loss weight (same as HMM-MoE)
SEQ_LEN = 52
EPOCHS = 100
PATIENCE = 15

def to_t(a): return torch.tensor(a, dtype=torch.float32)

class DirectionAwareLoss(nn.Module):
    """Direction-aware loss: MSE + λ * direction penalty"""
    def __init__(self, alpha=1.0, tau=0.1):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
    
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        # Direction loss: penalize wrong direction
        pd_ = pred[1:] - pred[:-1]
        td_ = target[1:] - target[:-1]
        dir_loss = torch.mean(F.relu(-pd_ * td_))
        return mse_loss + self.alpha * dir_loss

# ============================================================
# Neural Network Baselines (with direction-aware loss)
# ============================================================

class LSTMBaseline(nn.Module):
    def __init__(self, inp, hid=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUBaseline(nn.Module):
    def __init__(self, inp, hid=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(inp, hid, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class BiGRUBaseline(nn.Module):
    def __init__(self, inp, hid=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(inp, hid, layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid*2, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class MLPBaseline(nn.Module):
    def __init__(self, inp, hid=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid//2, 1)
        )
    def forward(self, x):
        return self.net(x[:, -1, :])  # use last timestep

class CNNBaseline(nn.Module):
    def __init__(self, inp, hid=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hid, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        # x: (B, L, F) -> conv expects (B, F, L)
        out = self.conv(x.transpose(1, 2))
        return self.fc(out.squeeze(-1))

class TransformerBaseline(nn.Module):
    def __init__(self, inp, d_model=64, nhead=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(inp, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

class LSTMAttnBaseline(nn.Module):
    def __init__(self, inp, hid=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        w = F.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(1)
        return self.fc(ctx)

class GRUAttnBaseline(nn.Module):
    def __init__(self, inp, hid=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(inp, hid, layers, batch_first=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        w = F.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(1)
        return self.fc(ctx)

# ============================================================
# Training helper
# ============================================================

def train_neural(model, train_ld, val_ld, epochs, patience, lr, device, loss_fn):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait = float('inf'), 0
    best_state = None
    for ep in range(epochs):
        model.train()
        for bx, by in train_ld:
            pred = model(bx.to(device))
            loss = loss_fn(pred, by.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for bx, by in val_ld:
                vl += F.mse_loss(model(bx.to(device)), by.to(device)).item() * bx.size(0)
        vl /= len(val_ld.dataset) if hasattr(val_ld, 'dataset') else 1
        if vl < best_val:
            best_val = vl; wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model

def eval_neural(model, test_ld, sc_y, base_p, true_p, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, in [(to_t(test_ld.dataset.tensors[0].numpy()),)]:
            pass
    # Simpler: just predict all at once
    X_test = test_ld.dataset.tensors[0]
    with torch.no_grad():
        pred_sc = model(X_test.to(device)).cpu().numpy().flatten()
    pred_r = sc_y.inverse_transform(pred_sc.reshape(-1,1)).flatten()
    n = min(len(pred_r), len(base_p), len(true_p))
    pred_r = pred_r[:n]; base_p = base_p[:n]; true_p = true_p[:n]
    pred_p = base_p * np.exp(pred_r)
    r_true_raw = np.log(true_p / base_p)
    mae = mean_absolute_error(true_p, pred_p)
    da = np.mean((pred_r * r_true_raw) > 0) * 100
    mape = np.mean(np.abs((true_p - pred_p) / true_p)) * 100
    return {'da': da, 'mae': mae, 'mape': mape}

# ============================================================
# Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    # Neural baselines to test
    NEURAL_MODELS = {
        'LSTM': lambda nf: LSTMBaseline(nf),
        'GRU': lambda nf: GRUBaseline(nf),
        'BiGRU': lambda nf: BiGRUBaseline(nf),
        'MLP': lambda nf: MLPBaseline(nf),
        'CNN-1D': lambda nf: CNNBaseline(nf),
        'Transformer': lambda nf: TransformerBaseline(nf),
        'LSTM+Attention': lambda nf: LSTMAttnBaseline(nf),
        'GRU+Attention': lambda nf: GRUAttnBaseline(nf),
    }
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")
        set_seed(seed)
        
        # Data preprocessing (same as HMM-MoE)
        (train_loader, val_loader, test_loader,
         scaler_feat, scaler_target, scaler_gate,
         all_feat_cols, gate_dim,
         regime_detector, regime_proba, regime_labels,
         test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
            file_path=FILE_PATH, feat_cols=FEAT_COLS,
            target_col=TARGET_COL, seq_len=SEQ_LEN,
            n_regimes=3, gate_window=4, vol_window=12, hmm_mode="multi")
        
        bx, bg, by = next(iter(train_loader))
        n_features = bx.shape[2]
        
        # Get raw arrays for sklearn baselines
        # Extract from loaders
        X_train = train_loader.dataset.tensors[0].numpy()
        Y_train = train_loader.dataset.tensors[2].numpy()
        X_val = val_loader.dataset.tensors[0].numpy()
        Y_val = val_loader.dataset.tensors[2].numpy()
        X_test = test_loader.dataset.tensors[0].numpy()
        Y_test = test_loader.dataset.tensors[2].numpy()
        
        # Flatten for sklearn: use last timestep
        X_tr_flat = X_train[:, -1, :]
        X_va_flat = X_val[:, -1, :]
        X_te_flat = X_test[:, -1, :]
        sc_y = scaler_target
        
        # Build neural loaders (without gate)
        loss_fn = DirectionAwareLoss(alpha=LOSS_ALPHA)
        train_ld = DataLoader(TensorDataset(to_t(X_train), to_t(Y_train).unsqueeze(1)),
                              batch_size=32, shuffle=True)
        val_ld = DataLoader(TensorDataset(to_t(X_val), to_t(Y_val).unsqueeze(1)),
                            batch_size=32)
        test_ld = DataLoader(TensorDataset(to_t(X_test), to_t(Y_test).unsqueeze(1)),
                             batch_size=32)
        
        base_p = test_base_prices
        true_p = test_true_prices
        
        # --- Neural models with direction-aware loss ---
        for name, builder in NEURAL_MODELS.items():
            print(f"\n  [{name}] seed={seed}...", end=" ", flush=True)
            t0 = time.time()
            set_seed(seed)
            model = builder(n_features)
            model = train_neural(model, train_ld, val_ld, EPOCHS, PATIENCE, 1e-3, device, loss_fn)
            metrics = eval_neural(model, test_ld, sc_y, base_p, true_p, device)
            r = {'model': name, 'seed': seed, 'loss': 'dir_aware',
                 'da': metrics['da'], 'mae': metrics['mae'], 'mape': metrics['mape'],
                 'runtime': time.time()-t0}
            results.append(r)
            print(f"DA={metrics['da']:.1f}%  MAE={metrics['mae']:.4f}  ({r['runtime']:.0f}s)")
        
        # --- Sklearn baselines (XGBoost, LightGBM, etc.) ---
        # For tree-based models, we can't directly use direction-aware loss,
        # but we can use a proxy: train on residuals of direction-weighted targets
        # OR just report them as-is with a note that tree methods optimize for MSE natively
        
        from xgboost import XGBRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge, Lasso, LinearRegression
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        SKLEARN_MODELS = {
            'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=seed),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=seed),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed),
            'SVR': SVR(C=1.0, epsilon=0.1),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'LinearRegression': LinearRegression(),
            'KNN': KNeighborsRegressor(n_neighbors=5),
        }
        
        for name, mdl in SKLEARN_MODELS.items():
            print(f"  [{name}] seed={seed}...", end=" ", flush=True)
            t0 = time.time()
            mdl.fit(X_tr_flat, Y_train)
            pred_sc = mdl.predict(X_te_flat)
            pred_r = sc_y.inverse_transform(pred_sc.reshape(-1,1)).flatten()
            n = min(len(pred_r), len(base_p), len(true_p))
            pred_r = pred_r[:n]; bp = base_p[:n]; tp = true_p[:n]
            pred_p = bp * np.exp(pred_r)
            r_true = np.log(tp / bp)
            mae = mean_absolute_error(tp, pred_p)
            da = np.mean((pred_r * r_true) > 0) * 100
            mape = np.mean(np.abs((tp - pred_p) / tp)) * 100
            r = {'model': name, 'seed': seed, 'loss': 'MSE_native',
                 'da': da, 'mae': mae, 'mape': mape, 'runtime': time.time()-t0}
            results.append(r)
            print(f"DA={da:.1f}%  MAE={mae:.4f}")
        
        # --- HMM-MoE (reference) ---
        print(f"  [HMM-MoE] seed={seed}...", end=" ", flush=True)
        t0 = time.time()
        set_seed(seed)
        
        hmm_model = OilMoE(
            n_features=n_features, gate_input_dim=gate_dim,
            n_experts=3, d_model=32, pred_len=1,
            gate_hidden_dim=16,
            expert_configs=[{
                "input_dim": n_features, "d_model": 32,
                "d_state": 16, "n_layers": 1, "dropout": 0.55,
            } for _ in range(3)])
        
        hmm_model, _ = train_moe_model(
            hmm_model, train_loader, val_loader,
            epochs=EPOCHS, lr=1e-4, balance_weight=0.01, patience=PATIENCE,
            device=device, loss_alpha=LOSS_ALPHA, loss_beta=0.01)
        
        _, _, _, _, _, metrics = evaluate_moe_model(
            hmm_model, test_loader, sc_y,
            base_p, true_p, device=device)
        
        r = {'model': 'HMM-MoE', 'seed': seed, 'loss': 'dir_aware',
             'da': metrics['dir_acc'], 'mae': metrics['mae_model'], 
             'mape': metrics['mape_model'], 'runtime': time.time()-t0}
        results.append(r)
        print(f"DA={metrics['dir_acc']:.1f}%  MAE={metrics['mae_model']:.4f}")
    
    # --- Summary ---
    df = pd.DataFrame(results)
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "fair_baseline_comparison.csv"), index=False)
    
    # Compute 3-seed means
    print(f"\n{'='*70}")
    print(f"  FAIR BASELINE COMPARISON — 3-SEED MEAN ± STD")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Loss':<12} {'DA (%)':<18} {'MAE':<15}")
    print(f"  {'-'*70}")
    
    for model_name in df['model'].unique():
        sub = df[df['model'] == model_name]
        loss_type = sub['loss'].iloc[0]
        da_mean, da_std = sub['da'].mean(), sub['da'].std()
        mae_mean, mae_std = sub['mae'].mean(), sub['mae'].std()
        print(f"  {model_name:<25} {loss_type:<12} {da_mean:.1f} ± {da_std:.1f}      {mae_mean:.2f} ± {mae_std:.2f}")
    
    print(f"\nSaved to {output_dir}/fair_baseline_comparison.csv")

if __name__ == "__main__":
    main()
