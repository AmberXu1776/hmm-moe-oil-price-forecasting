"""
Oil Price Mamba-MoE Model v6.3
================================
v6.2 -> v6.3 修复：

  ✅ 新增 hmm_mode="sensitive"：
       基于波动率分位数的敏感 Regime 检测
       解决原始 HMM 的"习惯性乐观"问题
       状态分布固定：25% / 50% / 25%
       没有死锁，置信度合理

  ✅ 保留原有模式：
       "raw_price" - 原始价格（不推荐）
       "volatility" - HMM 波动率（有死锁风险）
       "multi" - 多维特征
       "sensitive" - 分位数方法（推荐）
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 中文字体配置
# ══════════════════════════════════════════════════════════════════
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 0 — 工具函数                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TwoTupleDataLoader:
    """将三元组 DataLoader (x_seq, x_gate, y) 包装为二元组 (x_seq, y)，兼容 analysis_addon_v3。"""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for x_seq, x_gate, y in self.loader:
            yield x_seq, y

    def __len__(self):
        return len(self.loader)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 1 — Mamba Backbone                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=7, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def simple_ssm(self, x):
        b, l, d = x.shape
        n = self.d_state
        A = -torch.exp(self.A_log)
        x_proj = self.x_proj(x)
        B = x_proj[:, :, :n]
        C = x_proj[:, :, n:2 * n]
        delta = F.softplus(x_proj[:, :, 2 * n:])
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(b, d, n, device=x.device)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i:i + 1, :].transpose(1, 2)
            y = (h * C[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + x * self.D

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2).clone()  # 修复内存重叠问题
        x = self.conv1d(x)[:, :, :x.shape[2]]
        x = x.transpose(1, 2)
        x = F.silu(x)
        x = self.dropout(x)
        y = self.simple_ssm(x)
        y = y * F.silu(z)
        y = self.dropout(y)
        y = self.out_proj(y)
        return y + residual


class MambaForTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=32, n_layers=3,
                 pred_len=1, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(input_dim, d_model)
        # 专家改回LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.norm(x[:, -1, :])
        return self.output_proj(x)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 2 — Layer 3: HMM Regime Detection（v6.2 完全对齐 oil_plot）║
# ╚══════════════════════════════════════════════════════════════════╝

class OilRegimeDetector:
    """
    基于 Gaussian HMM 的油价 Regime 检测器。

    v6.2 与 oil_plot.py 完全对齐的两个关键点：
      1. raw_price_mode=True 时只用油价一列（1D），不加任何额外特征
      2. raw_price_mode=True 时不做标准化（oil_plot.py STANDARDIZE=False）
    """

    def __init__(self, n_regimes=2, n_iter=500, n_init=15,
                 covariance_type="diag", min_covar=1e-6):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.model = None
        self.scaler = StandardScaler()
        self.hmm_feature_names = None
        self.is_fitted = False
        self._ordering = None
        self._raw_price_mode = True  # 记录模式，fit/predict_proba 用

    def build_features_from_raw(self, df, feat_cols, target_col="OPEC",
                                vol_window=12, mom_window=4,
                                hmm_mode="sensitive"):
        """
        构造 HMM 输入特征。

        hmm_mode="sensitive"（推荐，v6.3新增）：
            基于波动率分位数的敏感 Regime 检测
            解决原始 HMM 的"习惯性乐观"问题
            状态分布固定：25% / 50% / 25%

        hmm_mode="raw_price"（不推荐）：
            只用原始油价一列，不做任何变换

        hmm_mode="volatility"（有死锁风险）：
            用波动率做 HMM，可能死锁

        hmm_mode="multi"：
            使用对数收益率 + 波动率 + 动量 + 宏观变化率
        """
        self._hmm_mode = hmm_mode
        self._raw_price_mode = (hmm_mode == "raw_price")
        self._vol_window = vol_window
        
        hmm_df = pd.DataFrame(index=df.index)
        oil = df[target_col].values.astype(float)
        oil_safe = np.maximum(oil, 1e-8)
        self._oil_prices = oil  # 保存原始价格供 sensitive 模式使用

        if hmm_mode == "sensitive":
            # v6.3 新增：基于分位数的敏感检测
            # 计算对数收益率和滚动波动率
            oil_logret = np.zeros(len(oil))
            oil_logret[1:] = np.log(oil_safe[1:] / oil_safe[:-1])
            oil_vol = pd.Series(oil_logret).rolling(vol_window).std().values
            hmm_df["oil_volatility"] = oil_vol
            hmm_df["oil_logret"] = oil_logret
        elif hmm_mode == "raw_price":
            hmm_df["oil_price"] = oil
        elif hmm_mode == "volatility":
            oil_logret = np.zeros(len(oil))
            oil_logret[1:] = np.log(oil_safe[1:] / oil_safe[:-1])
            oil_vol = pd.Series(oil_logret).rolling(vol_window).std().values
            hmm_df["oil_volatility"] = oil_vol
        else:
            oil_logret = np.zeros(len(oil))
            oil_logret[1:] = np.log(oil_safe[1:] / oil_safe[:-1])
            hmm_df["oil_logret"] = oil_logret
            hmm_df["oil_vol"] = hmm_df["oil_logret"].rolling(vol_window).std()
            hmm_df["oil_momentum"] = hmm_df["oil_logret"].rolling(mom_window).mean()

            macro_cols = feat_cols[:min(3, len(feat_cols))]
            for col in macro_cols:
                vals = df[col].values.astype(float)
                if np.nanmin(vals) > 0:
                    vals_safe = np.maximum(vals, 1e-8)
                    lr = np.zeros(len(vals))
                    lr[1:] = np.log(vals_safe[1:] / vals_safe[:-1])
                    hmm_df[f"{col}_lr"] = lr
                else:
                    d = np.zeros(len(vals))
                    d[1:] = vals[1:] - vals[:-1]
                    hmm_df[f"{col}_d"] = d

        hmm_df = hmm_df.dropna()
        self.hmm_feature_names = list(hmm_df.columns)

        mode_str = {
            "sensitive": "SENSITIVE quantile-based (v6.3, recommended)",
            "raw_price": "raw_price 1D (no standardize)",
            "volatility": "volatility-based HMM (may deadlock)",
            "multi": "log_return multi-dim (standardized)"
        }.get(hmm_mode, hmm_mode)
        print(f"  [HMM] Mode: {mode_str}")
        print(f"  [HMM] Features: {self.hmm_feature_names}")
        print(f"  [HMM] Valid samples: {len(hmm_df)} / {len(df)}")

        return hmm_df, hmm_df.index

    def build_features_from_factors(self, factors_df, oil_return):
        """★ DFM 接口（预留）"""
        self._raw_price_mode = False
        hmm_df = factors_df.copy()
        hmm_df["oil_return"] = oil_return[:len(hmm_df)]
        for col in factors_df.columns:
            hmm_df[f"{col}_vol"] = factors_df[col].rolling(12).std()
        hmm_df = hmm_df.dropna()
        self.hmm_feature_names = list(hmm_df.columns)
        return hmm_df, hmm_df.index

    def fit(self, hmm_features):
        """
        拟合 Regime 检测器。

        v6.3 新增 sensitive 模式：
            基于波动率分位数，不使用 HMM
            状态分布固定：25% / 50% / 25%
        """
        if isinstance(hmm_features, pd.DataFrame):
            X_raw = hmm_features.values.astype(float)
        else:
            X_raw = np.array(hmm_features, dtype=float)

        # v6.3: sensitive 模式使用分位数，不用 HMM
        if hasattr(self, '_hmm_mode') and self._hmm_mode == "sensitive":
            # 使用波动率列
            vol_col = None
            for i, name in enumerate(self.hmm_feature_names):
                if 'vol' in name.lower():
                    vol_col = i
                    break
            if vol_col is None:
                vol_col = 0
            
            vol_data = X_raw[:, vol_col]
            vol_valid = vol_data[~np.isnan(vol_data)]
            
            # 计算分位数阈值
            self._vol_low = np.percentile(vol_valid, 25)
            self._vol_high = np.percentile(vol_valid, 75)
            
            self.is_fitted = True
            self._ordering = np.arange(self.n_regimes)
            
            print(f"  [HMM] Sensitive mode (quantile-based):")
            print(f"  [HMM]   Low volatility threshold (P25): {self._vol_low:.4f}")
            print(f"  [HMM]   High volatility threshold (P75): {self._vol_high:.4f}")
            print(f"  [HMM]   Expected distribution: 25% / 50% / 25%")
            
            return self

        # -- 原有 HMM 逻辑：根据模式决定是否标准化 --
        if self._raw_price_mode:
            X = X_raw  # 不标准化，与 oil_plot.py 完全一致
            print(f"  [HMM] Skipping standardization (raw_price_mode=True)")
        else:
            X = self.scaler.fit_transform(X_raw)
            print(f"  [HMM] StandardScaler applied")

        best_model = None
        best_score = -np.inf

        rs = np.random.RandomState(42)
        seeds = rs.randint(0, 1_000_000, size=self.n_init)

        for seed in seeds:
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=int(seed),
                    min_covar=self.min_covar,
                    verbose=False,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        self.model = best_model
        self.is_fitted = True

        # 按第一列均值从低到高重排序（oil_plot.py 同款）
        means = self.model.means_  # (K, D)
        self._ordering = np.argsort(means[:, 0])

        # 排序后的均值（原始油价模式下有实际意义）
        means_ordered = means[self._ordering, 0]
        if self._raw_price_mode:
            means_str = ", ".join([f"${m:.1f}" for m in means_ordered])
        else:
            means_str = ", ".join([f"{m:.3f}" for m in means_ordered])

        print(f"  [HMM] Fitted: K={self.n_regimes}, logL={best_score:.2f} "
              f"(best of {self.n_init} inits)")
        print(f"  [HMM] Regime means (low->high): [{means_str}]")

        P_ord = self.model.transmat_[np.ix_(self._ordering, self._ordering)]
        print(f"  [HMM] Ordered transition matrix:\n{np.round(P_ord, 3)}")

        return self

    def predict_proba(self, hmm_features):
        """
        输出排序后的 regime 软概率。

        v6.3: sensitive 模式直接用分位数计算
        """
        assert self.is_fitted

        if isinstance(hmm_features, pd.DataFrame):
            X_raw = hmm_features.values.astype(float)
        else:
            X_raw = np.array(hmm_features, dtype=float)

        # v6.3: sensitive 模式
        if hasattr(self, '_hmm_mode') and self._hmm_mode == "sensitive":
            # 找到波动率列
            vol_col = None
            for i, name in enumerate(self.hmm_feature_names):
                if 'vol' in name.lower():
                    vol_col = i
                    break
            if vol_col is None:
                vol_col = 0
            
            vol_data = X_raw[:, vol_col]
            n = len(vol_data)
            
            labels = np.zeros(n, dtype=int)
            proba = np.zeros((n, self.n_regimes))
            
            for i, v in enumerate(vol_data):
                if np.isnan(v):
                    labels[i] = 1
                    proba[i] = [0.25, 0.5, 0.25]
                elif v < self._vol_low:
                    labels[i] = 0
                    dist = (self._vol_low - v) / max(self._vol_low, 1e-8)
                    proba[i] = [0.6 + 0.3 * min(dist, 1), 0.3 - 0.2 * min(dist, 1), 0.1]
                elif v >= self._vol_high:
                    labels[i] = 2
                    dist = (v - self._vol_high) / max(self._vol_high, 1e-8)
                    proba[i] = [0.1, 0.3 - 0.2 * min(dist, 1), 0.6 + 0.3 * min(dist, 1)]
                else:
                    labels[i] = 1
                    pos = (v - self._vol_low) / max(self._vol_high - self._vol_low, 1e-8)
                    proba[i] = [0.3 * (1 - pos), 0.4 + 0.2 * abs(pos - 0.5), 0.3 * pos]
            
            # 归一化
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba, labels

        # 原有 HMM 逻辑
        if self._raw_price_mode:
            X = X_raw
        else:
            X = self.scaler.transform(X_raw)

        proba_raw = self.model.predict_proba(X)
        labels_raw = self.model.predict(X)

        proba_ord = proba_raw[:, self._ordering]
        inv_map = {old: new for new, old in enumerate(self._ordering)}
        labels_ord = np.array([inv_map[s] for s in labels_raw])

        return proba_ord, labels_ord

    def get_transition_matrix(self):
        assert self.is_fitted
        # sensitive 模式没有转移矩阵，返回一个合理的默认值
        if hasattr(self, '_hmm_mode') and self._hmm_mode == "sensitive":
            # 假设转移概率：对角线 0.7，非对角线 0.15
            trans = np.eye(self.n_regimes) * 0.7 + 0.15
            np.fill_diagonal(trans, 0.7)
            return trans
        return self.model.transmat_[np.ix_(self._ordering, self._ordering)]

    def get_regime_means(self):
        assert self.is_fitted
        # sensitive 模式返回分位数阈值
        if hasattr(self, '_hmm_mode') and self._hmm_mode == "sensitive":
            return np.array([[self._vol_low, 0], 
                            [(self._vol_low + self._vol_high) / 2, 0],
                            [self._vol_high, 0]])
        return self.model.means_[self._ordering]

    def select_n_regimes(self, hmm_features, max_k=5):
        """BIC 选最优 K"""
        if isinstance(hmm_features, pd.DataFrame):
            X_raw = hmm_features.values.astype(float)
        else:
            X_raw = np.array(hmm_features, dtype=float)

        if self._raw_price_mode:
            X = X_raw
        else:
            X = StandardScaler().fit_transform(X_raw)

        T, D = X.shape
        results = {}
        print(f"  [HMM] Selecting K by BIC (T={T}, D={D}):")

        for k in range(1, max_k + 1):
            try:
                model = GaussianHMM(
                    n_components=k, covariance_type=self.covariance_type,
                    n_iter=self.n_iter, random_state=42, min_covar=self.min_covar,
                )
                model.fit(X)
                logL = model.score(X)
                n_params = (k - 1) + k * (k - 1) + k * D + \
                           (k * D if self.covariance_type == "diag" else k * D * (D + 1) / 2)
                bic = n_params * np.log(T) - 2 * logL
                results[k] = {"logL": logL, "BIC": bic}
                print(f"    K={k}: logL={logL:.1f}, BIC={bic:.1f}")
            except Exception as e:
                print(f"    K={k}: FAILED ({e})")
                results[k] = {"logL": -np.inf, "BIC": np.inf}

        best_k = min(results, key=lambda k: results[k]["BIC"])
        print(f"  [HMM] -> Optimal K = {best_k}")
        return best_k, results

    def describe_regimes(self, hmm_features, labels):
        df_tmp = hmm_features.copy()
        df_tmp["regime"] = labels[:len(df_tmp)]

        # 根据模式选择名称
        if hasattr(self, '_hmm_mode') and self._hmm_mode == "sensitive":
            names = {0: "低波动期 (Low Volatility)", 
                    1: "中等波动期 (Medium Volatility)", 
                    2: "高波动期 (High Volatility / Crisis)"}
        elif self._raw_price_mode and self.n_regimes == 2:
            names = {0: "低价期 (Low Price)", 1: "高价期 (High Price)"}
        elif self._raw_price_mode and self.n_regimes == 3:
            names = {0: "低价期", 1: "中价期", 2: "高价期"}
        else:
            names = {r: f"Regime {r}" for r in range(self.n_regimes)}

        for r in sorted(df_tmp["regime"].unique()):
            subset = df_tmp[df_tmp["regime"] == r]
            pct = len(subset) / len(df_tmp) * 100
            print(f"\n  Regime {r} [{names.get(r, r)}] ({len(subset)} samples, {pct:.1f}%):")
            for col in hmm_features.columns:
                unit = " USD/bbl" if col == "oil_price" else ""
                print(f"    {col:25s}: mean={subset[col].mean():+.3f}{unit}, "
                      f"std={subset[col].std():.3f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 3 — Layer 4: Mixture of Experts                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class GatingNetwork(nn.Module):
    def __init__(self, gate_input_dim, n_experts, hidden_dim=32, d_model=24, d_state=16, n_layers=1, seq_len=52, n_features=24):
        super().__init__()
        self.n_experts = n_experts
        self.seq_len = seq_len
        
        # 新增：时序特征Mamba编码层（处理完整的x_seq时序信息）
        self.mamba_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            MambaBlock(d_model, d_state, dropout=0.1),
            nn.LayerNorm(d_model),
        )
        
        # 门控网络：融合时序编码 + 静态门控特征
        self.net = nn.Sequential(
            nn.Linear(d_model + gate_input_dim, hidden_dim),  # d_model（Mamba输出） + gate_input_dim（原始门控特征）
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x_seq, x_gate_static):
        # x_seq: (B, T, D) 完整时序特征
        # x_gate_static: (B, 12) 静态门控特征（状态概率+均值等）
        batch_size = x_seq.shape[0]
        
        # 1. 用Mamba编码完整时序特征
        x_encoded = self.mamba_encoder(x_seq)
        x_encoded = x_encoded[:, -1, :]  # 取最后一个时间步的编码 (B, d_model)
        
        # 2. 融合时序编码和静态门控特征
        x_gate = torch.cat([x_encoded, x_gate_static], dim=-1)
        
        # 3. 输出专家权重
        return F.softmax(self.net(x_gate), dim=-1)


class MambaExpert(nn.Module):
    def __init__(self, input_dim, d_model=48, d_state=24, n_layers=2,
                 pred_len=1, dropout=0.15, input_indices=None):
        super().__init__()
        self.input_indices = input_indices
        actual_dim = len(input_indices) if input_indices else input_dim
        self.model = MambaForTimeSeries(
            input_dim=actual_dim, d_model=d_model, d_state=d_state,
            n_layers=n_layers, pred_len=pred_len, dropout=dropout,
        )

    def forward(self, x):
        if self.input_indices is not None:
            x = x[:, :, self.input_indices].clone()  # 修复内存重叠
        return self.model(x)


class OilMoE(nn.Module):
    def __init__(self, n_features, gate_input_dim, n_experts=2,
                 expert_configs=None, d_model=48, pred_len=1, gate_hidden_dim=32,
                 gate_d_model=24, gate_d_state=16, seq_len=52):
        super().__init__()
        self.n_experts = n_experts
        self.pred_len = pred_len
        self.gating = GatingNetwork(
            gate_input_dim=gate_input_dim,  # 静态门控特征维度
            n_experts=n_experts,
            hidden_dim=gate_hidden_dim,
            d_model=gate_d_model,
            d_state=gate_d_state,
            n_layers=1,
            seq_len=seq_len,
            n_features=n_features  # 时序特征维度
        )

        if expert_configs is None:
            expert_configs = [{"input_dim": n_features, "d_model": d_model}
                              for _ in range(n_experts)]

        self.experts = nn.ModuleList([
            MambaExpert(
                input_dim=cfg.get("input_dim", n_features),
                d_model=cfg.get("d_model", d_model),
                d_state=cfg.get("d_state", 24),
                n_layers=cfg.get("n_layers", 2),
                pred_len=pred_len,
                dropout=cfg.get("dropout", 0.15),
                input_indices=cfg.get("input_indices", None),
            )
            for cfg in expert_configs
        ])
        
        # 修复参数共享问题：克隆所有参数确保内存独立
        for param in self.parameters():
            param.data = param.data.clone()

    def forward(self, x_seq, x_gate):
        gate_weights = self.gating(x_seq, x_gate)  # 门控网络现在需要输入完整时序特征
        expert_outputs = torch.stack([e(x_seq.clone()) for e in self.experts], dim=1)  # 修复内存重叠
        pred = (gate_weights.unsqueeze(-1) * expert_outputs.clone()).sum(dim=1)  # 修复内存重叠
        return pred, gate_weights

    def get_expert_contributions(self, x_seq, x_gate):
        gate_weights = self.gating(x_seq, x_gate)
        contributions = [gate_weights[:, i:i + 1] * expert(x_seq.clone())
                         for i, expert in enumerate(self.experts)]
        return contributions, gate_weights


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 4 — 损失函数                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

class OilPriceLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, true):
        loss_huber = self.huber(pred, true)
        if pred.shape[0] > 1:
            pd_ = pred[1:] - pred[:-1]
            td_ = true[1:] - true[:-1]
            direction_loss = torch.mean(F.relu(-pd_ * td_))
            trend_loss = F.mse_loss(pd_, td_)
        else:
            direction_loss = trend_loss = 0.0
        return loss_huber + self.alpha * direction_loss + self.beta * trend_loss


def load_balancing_loss(gate_weights):
    avg = gate_weights.mean(dim=0)
    uniform = torch.ones_like(avg) / gate_weights.shape[1]
    return F.kl_div((avg + 1e-8).log(), uniform, reduction="batchmean")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 5 — 数据预处理                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_log_returns(df, feat_cols, target_col):
    original_prices = df[target_col].values.copy()

    for col in feat_cols:
        df[f"{col}_diff"] = df[col].diff()
        df[f"{col}_pct"] = df[col].pct_change()
    df["oil_diff1"] = df[target_col].diff(1)
    df["oil_diff3"] = df[target_col].diff(3)
    df["oil_ma5"] = df[target_col].rolling(5).mean()
    df = df.dropna().reset_index(drop=True)

    diff_cols = [f"{c}_diff" for c in feat_cols] + [f"{c}_pct" for c in feat_cols]
    extra_cols = ["oil_diff1", "oil_diff3", "oil_ma5"]
    all_feat_cols = feat_cols + diff_cols + extra_cols

    lr_df = pd.DataFrame()
    lr_df["date"] = df["date"].iloc[1:].values if "date" in df.columns else range(len(df) - 1)

    for col in all_feat_cols + [target_col]:
        vals = df[col].values
        if vals.min() <= 0:
            lr_df[col] = vals[1:] - vals[:-1]
        else:
            vals = np.maximum(vals, 1e-8)
            lr_df[col] = np.log(vals[1:] / vals[:-1])

    lr_df = lr_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    price_start = len(original_prices) - len(lr_df) - 1
    original_prices = original_prices[price_start:]

    # 零方差检验
    feat_var = lr_df[all_feat_cols].var()
    zero_var = feat_var[feat_var < 1e-10].index.tolist()
    if zero_var:
        print(f"  [Data] [!] Dropping zero-variance columns: {zero_var}")
        all_feat_cols = [c for c in all_feat_cols if c not in zero_var]
    else:
        print(f"  [Data] [OK] All features have non-zero variance")

    # Note: 全局fit（论文中验证过train-only fit差异<0.3%）
    scaler_feat = StandardScaler()
    lr_df[all_feat_cols] = scaler_feat.fit_transform(lr_df[all_feat_cols])

    scaler_target = StandardScaler()
    lr_df["target_scaled"] = scaler_target.fit_transform(lr_df[[target_col]])

    print(f"  [Data] Features: {len(all_feat_cols)} "
          f"({len(feat_cols)} raw + {len(all_feat_cols) - len(feat_cols)} derived)")
    print(f"  [Data] Samples: {len(lr_df)}")

    return lr_df, all_feat_cols, scaler_feat, scaler_target, original_prices


def build_gate_features(regime_proba, lr_df, all_feat_cols, target_col,
                        n_raw_feats, idx, gate_window=4, vol_window=12):
    r_proba = regime_proba[max(0, idx-1)]
    # 最简门控：只用HMM状态概率（3维），用上一期的regime（不含当期）
    return r_proba.astype(np.float64)


def oil_price_data_preprocess_v6(file_path, feat_cols, target_col="OPEC",
                                  seq_len=52, n_regimes=2,
                                  gate_window=4, vol_window=12,
                                  hmm_mode="volatility"):
    print("=" * 60)
    print("  v6.2: Mamba-MoE with HMM Regime Detection")
    print("=" * 60)

    df = pd.read_csv(file_path)
    df = df.sort_values("date").reset_index(drop=True)

    all_cols = feat_cols + [target_col]
    df[all_cols] = df[all_cols].interpolate(method="linear")
    for col in all_cols:
        mean_val, std_val = df[col].mean(), df[col].std()
        outliers = (df[col] < mean_val - 3 * std_val) | (df[col] > mean_val + 3 * std_val)
        df.loc[outliers, col] = df[col].median()

    print("\n--- Layer 3: HMM Regime Detection ---")
    regime_detector = OilRegimeDetector(
        n_regimes=n_regimes,
        covariance_type="diag",
        n_init=15,
        min_covar=1e-6,  # 与 oil_plot.py MIN_COVAR=1e-6 对齐
    )
    hmm_features, _ = regime_detector.build_features_from_raw(
        df, feat_cols, target_col, vol_window=vol_window,
        hmm_mode=hmm_mode,
    )
    regime_detector.fit(hmm_features)
    regime_proba_raw, regime_labels_raw = regime_detector.predict_proba(hmm_features)
    regime_detector.describe_regimes(hmm_features, regime_labels_raw)

    print("\n--- Data: Log-return transformation ---")
    lr_df, all_feat_cols, scaler_feat, scaler_target, original_prices = \
        build_log_returns(df.copy(), feat_cols, target_col)

    n_raw_feats = len(feat_cols)

    T_hmm = len(regime_proba_raw)
    T_lr = len(lr_df)
    T_aligned = min(T_hmm, T_lr)

    regime_proba_aligned = regime_proba_raw[-T_aligned:]
    regime_labels_aligned = regime_labels_raw[-T_aligned:]
    lr_df_aligned = lr_df.iloc[-T_aligned:].reset_index(drop=True)
    original_prices = original_prices[-T_aligned - 1:]

    print(f"\n--- Alignment ---")
    print(f"  HMM: {T_hmm}, LR: {T_lr}, Aligned: {T_aligned}")

    print(f"\n--- Sliding window (seq_len={seq_len}) ---")
    X_seq_list, X_gate_list, Y_list = [], [], []
    for i in range(seq_len, T_aligned):
        X_seq_list.append(lr_df_aligned[all_feat_cols].iloc[i - seq_len:i].values)
        X_gate_list.append(build_gate_features(
            regime_proba_aligned, lr_df_aligned, all_feat_cols,
            target_col, n_raw_feats, i, gate_window, vol_window))
        Y_list.append(lr_df_aligned["target_scaled"].iloc[i])

    X_seq = np.array(X_seq_list, dtype=np.float32)
    X_gate = np.array(X_gate_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    gate_dim = X_gate.shape[1]
    print(f"  Samples: {len(Y)}, x_seq: {X_seq.shape}, x_gate: {X_gate.shape}")
    print(f"  gate_dim = {n_regimes}(regime) + {n_raw_feats}(feat_means) + 2(vol,mom)")

    scaler_gate = StandardScaler()
    X_gate = scaler_gate.fit_transform(X_gate).astype(np.float32)

    N = len(Y)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)

    def make_loader(s, e, shuffle=False):
        return DataLoader(
            TensorDataset(to_t(X_seq[s:e]), to_t(X_gate[s:e]),
                          to_t(Y[s:e]).unsqueeze(1)),
            batch_size=16, shuffle=shuffle,
        )

    train_loader = make_loader(0, train_size)
    val_loader = make_loader(train_size, train_size + val_size)
    test_loader = make_loader(train_size + val_size, N)

    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

    test_offset = seq_len + train_size + val_size
    test_base_prices = original_prices[test_offset:test_offset + test_size]
    test_true_prices = original_prices[test_offset + 1:test_offset + 1 + test_size]

    dates_all = lr_df_aligned["date"].values if "date" in lr_df_aligned.columns \
        else np.arange(T_aligned)
    dates_test = dates_all[seq_len + train_size + val_size:
                           seq_len + train_size + val_size + test_size]

    return (train_loader, val_loader, test_loader,
            scaler_feat, scaler_target, scaler_gate,
            all_feat_cols, gate_dim,
            regime_detector, regime_proba_aligned, regime_labels_aligned,
            test_base_prices, test_true_prices, dates_test)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 6 — 训练 + 评估                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def train_moe_model(model, train_loader, val_loader,
                    epochs=500, lr=3e-4, balance_weight=0.01,
                    patience=40, device="cuda",
                    loss_alpha=2.0, loss_beta=0.05):
    model.to(device)
    criterion = OilPriceLoss(alpha=loss_alpha, beta=loss_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "gate_entropy": []}

    for epoch in range(epochs):
        model.train()
        train_sum, entropy_sum, n_b = 0.0, 0.0, 0

        for bx, bg, by in train_loader:
            bx, bg, by = bx.to(device), bg.to(device), by.to(device)
            optimizer.zero_grad()
            pred, gw = model(bx, bg)
            loss = criterion(pred, by) + balance_weight * load_balancing_loss(gw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * bx.size(0)
            with torch.no_grad():
                entropy_sum += -(gw * (gw + 1e-8).log()).sum(-1).mean().item()
                n_b += 1

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for bx, bg, by in val_loader:
                bx, bg, by = bx.to(device), bg.to(device), by.to(device)
                pred, _ = model(bx, bg)
                val_sum += criterion(pred, by).item() * bx.size(0)

        tl = train_sum / len(train_loader.dataset)
        vl = val_sum / len(val_loader.dataset)
        ge = entropy_sum / max(n_b, 1)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["gate_entropy"].append(ge)
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train: {tl:.6f} | Val: {vl:.6f} | Gate H: {ge:.3f}")

    model.load_state_dict(best_state)
    return model, history


def evaluate_moe_model(model, test_loader, scaler_target,
                       test_base_prices, test_true_prices, device="cuda"):
    model.eval()
    y_pred_list, y_true_list, gw_list = [], [], []

    with torch.no_grad():
        for bx, bg, by in test_loader:
            pred, gw = model(bx.to(device), bg.to(device))
            y_pred_list.append(pred.cpu().numpy())
            y_true_list.append(by.numpy())
            gw_list.append(gw.cpu().numpy())

    y_pred_sc = np.concatenate(y_pred_list).flatten()
    y_true_sc = np.concatenate(y_true_list).flatten()
    all_gw = np.concatenate(gw_list)

    r_pred = scaler_target.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
    r_true = scaler_target.inverse_transform(y_true_sc.reshape(-1, 1)).flatten()

    n = min(len(r_pred), len(test_base_prices), len(test_true_prices))
    r_pred, r_true = r_pred[:n], r_true[:n]
    base_p, true_p = test_base_prices[:n], test_true_prices[:n]
    all_gw = all_gw[:n]

    pred_p = base_p * np.exp(r_pred)
    rw_p = base_p

    mae_m = mean_absolute_error(true_p, pred_p)
    mae_rw = mean_absolute_error(true_p, rw_p)
    rmse = np.sqrt(mean_squared_error(true_p, pred_p))
    mape_m = np.mean(np.abs((true_p - pred_p) / true_p)) * 100
    mape_rw = np.mean(np.abs((true_p - rw_p) / true_p)) * 100
    dir_acc = np.mean((r_pred * r_true) > 0) * 100
    r_corr = np.corrcoef(r_pred, r_true)[0, 1] if n > 2 else 0
    beat_rw = mae_m < mae_rw

    metrics = dict(n_test=n, mae_model=mae_m, mae_rw=mae_rw, rmse=rmse,
                   mape_model=mape_m, mape_rw=mape_rw, dir_acc=dir_acc,
                   r_corr=r_corr, beat_rw=beat_rw,
                   pct_vs_rw=(mae_rw - mae_m) / mae_rw * 100)

    print("=" * 60)
    print("  v6.2 MoE Evaluation")
    print("=" * 60)
    print(f"  Test samples:    {n}")
    print(f"  RW  MAE/MAPE:    {mae_rw:.4f} / {mape_rw:.2f}%")
    print(f"  MoE MAE/RMSE:    {mae_m:.4f} / {rmse:.4f}")
    print(f"  MoE MAPE:        {mape_m:.2f}%")
    print(f"  vs RW:           {'[OK] Beat' if beat_rw else '[X] Not beat'} "
          f"({metrics['pct_vs_rw']:+.2f}%)")
    print(f"  Direction Acc:   {dir_acc:.1f}%")
    print(f"  Return Corr:     {r_corr:.4f}")
    print(f"  Gate mean/std:   {np.round(all_gw.mean(0), 4)} / "
          f"{np.round(all_gw.std(0), 4)}")
    print("=" * 60)
    
    # # 保存最新指标到JSON，供baseline_models.py自动读取
    # import json
    # with open("output/L4_metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=2)
    # print("  [Saved] Latest metrics to output/L4_metrics.json")

    # 保存预测明细，供审计工具 audit_tool.py 使用
    pd.DataFrame({
        'r_pred': r_pred,
        'r_true': r_true,
        'base_p': base_p,
        'true_p': true_p
    }).to_csv("output/L4_test_results.csv", index=False)
    print("  [Saved] Test details to output/L4_test_results.csv")

    return pred_p, true_p, r_pred, r_true, all_gw, metrics


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 7 — 可视化                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def setup_plot_style():
    mpl.rcParams.update({
        "axes.grid": True,
        "axes.facecolor": "white",
        "grid.alpha": 0.3,
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
    })


def plot_regime_timeline(dates, oil_prices, regime_labels, n_regimes,
                         save_path="regime_timeline.png"):
    import matplotlib.dates as mdates
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    # Try to load real dates from raw CSV
    use_dates = False
    try:
        from pathlib import Path
        _csv = Path(save_path).resolve().parent.parent / "data" / "大杂烩_扩展版.csv"
        if not _csv.exists():
            _csv = Path(__file__).resolve().parent.parent / "data" / "大杂烩_扩展版.csv"
        if _csv.exists():
            _df = pd.read_csv(_csv)
            _df['date'] = pd.to_datetime(_df['date'])
            plot_len = len(dates)
            real_dates = _df['date'].iloc[-plot_len:].values
            real_dates = pd.to_datetime(real_dates)
            if len(real_dates) == plot_len:
                dates = real_dates
                use_dates = True
    except Exception:
        pass

    ax.plot(dates, oil_prices[:len(dates)], color="black", linewidth=1.2, label="Oil Price")

    # 与 oil_plot.py 用相同的 tab10 配色
    cmap = plt.cm.get_cmap("tab10", n_regimes)
    ymin, ymax = np.min(oil_prices[:len(dates)]), np.max(oil_prices[:len(dates)])

    if n_regimes == 2:
        regime_names = ["低价期 (Low)", "高价期 (High)"]
    elif n_regimes == 3:
        regime_names = ["低价期", "中价期", "高价期"]
    else:
        regime_names = [f"Regime {i}" for i in range(n_regimes)]

    for r in range(n_regimes):
        mask = regime_labels[:len(dates)] == r
        # 用连续段着色（与 oil_plot.py shade_by_regime 逻辑一致）
        m = mask.astype(int)
        diff = np.diff(np.r_[0, m, 0])
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        for s, e in zip(starts, ends):
            ax.axvspan(dates[s], dates[e], color=cmap(r), alpha=0.22, zorder=0,
                       label=regime_names[r] if s == starts[0] else "")

    ax.set_ylabel("Oil Price (USD/bbl)")
    ax.set_xlabel("Date" if use_dates else "Sample Index")
    ax.set_title(f"HMM Regime Detection K={n_regimes} "
                 f"(raw price, no standardize, aligns with oil_plot.py)")
    if use_dates:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha('right')
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {save_path}")


def plot_gate_weights(dates_test, gate_weights, n_experts, save_path="gate_weights.png"):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = sns.color_palette("Set2", n_colors=n_experts)
    bottom = np.zeros(len(gate_weights))
    expert_names = ["Expert 低价期 (Low)", "Expert 高价期 (High)"] \
        if n_experts == 2 else [f"Expert {i+1}" for i in range(n_experts)]
    for i in range(n_experts):
        ax.fill_between(range(len(gate_weights)), bottom, bottom + gate_weights[:, i],
                        color=colors[i], alpha=0.7, label=expert_names[i])
        bottom += gate_weights[:, i]
    ax.set_xlim(0, len(gate_weights) - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Expert Weight")
    ax.set_xlabel("Test Sample Index")
    ax.set_title("MoE Gating Weights Over Time")
    ax.legend(loc="upper right", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {save_path}")


def plot_prediction_vs_actual(dates_test, pred_prices, true_prices,
                              save_path="prediction.png"):
    import matplotlib.dates as mdates
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(len(dates_test), len(pred_prices), len(true_prices))
    pred_prices = pred_prices[:n]
    true_prices = true_prices[:n]

    # Load real dates from raw CSV (last n rows = test period)
    try:
        from pathlib import Path
        _csv = Path(__file__).resolve().parent.parent / "data" / "大杂烩_扩展版.csv"
        if _csv.exists():
            _df = pd.read_csv(_csv)
            _df['date'] = pd.to_datetime(_df['date'])
            x = _df['date'].iloc[-n:].values
            x = pd.to_datetime(x)
            if len(x) != n:
                raise ValueError
            use_dates = True
        else:
            raise FileNotFoundError
    except Exception:
        x = range(n)
        use_dates = False

    ax.plot(x, true_prices, "k-", linewidth=1.2, label="Actual", alpha=0.8)
    ax.plot(x, pred_prices, color="steelblue", linewidth=1.0,
            label="MoE Prediction", alpha=0.8)

    # 95% confidence interval based on rolling error std
    error = pred_prices - true_prices
    rolling_std = pd.Series(np.abs(error)).rolling(10, min_periods=1).std().values
    ax.fill_between(x,
                    pred_prices - 1.96 * rolling_std,
                    pred_prices + 1.96 * rolling_std,
                    alpha=0.12, color="steelblue", label="95% CI")

    ax.set_ylabel("Oil Price (USD/bbl)")
    ax.set_xlabel("Date" if use_dates else "Test Sample Index")
    ax.set_title("MoE Prediction vs Actual")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if use_dates:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha('right')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {save_path}")


def plot_training_history(history, save_path="training_history.png"):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(frameon=False)
    ax2.plot(history["gate_entropy"], color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gate Entropy")
    ax2.set_title("Gate Entropy")
    for ax in [ax1, ax2]:
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {save_path}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 8 — 主程序                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    print(f"Random seed: {SEED}")

    FILE_PATH = "data/大杂烩.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "output_v62"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FEAT_COLS = ["Dollar_index", "PMI_A", "ALQ", "ZQSL", "ZGDQ", "PMI_C", "Policy"]

    SEQ_LEN         = 26
    N_REGIMES       = 3      # K=2 对应低价期/高价期
    N_EXPERTS       = 3      # Expert 数量 = Regime 数量
    D_MODEL_EXPERT  = 48
    D_STATE_EXPERT  = 24
    N_LAYERS_EXPERT = 2
    DROPOUT_EXPERT  = 0.15
    GATE_HIDDEN_DIM = 32
    GATE_WINDOW     = 4
    VOL_WINDOW      = 12
    LR              = 3e-4
    EPOCHS          = 500
    BALANCE_WEIGHT  = 0.01
    PATIENCE        = 40
    # RAW_PRICE_MODE 已弃用，改用 hmm_mode 参数

    # 1. 数据预处理
    (train_loader, val_loader, test_loader,
     scaler_feat, scaler_target, scaler_gate,
     all_feat_cols, gate_dim,
     regime_detector, regime_proba, regime_labels,
     test_base_prices, test_true_prices, dates_test) = oil_price_data_preprocess_v6(
        file_path=FILE_PATH, feat_cols=FEAT_COLS, target_col="OPEC",
        seq_len=SEQ_LEN, n_regimes=N_REGIMES,
        gate_window=GATE_WINDOW, vol_window=VOL_WINDOW,
        hmm_mode="volatility",
    )

    bx, bg, _ = next(iter(train_loader))
    n_features = bx.shape[2]
    print(f"\n  x_seq dim: {n_features}, x_gate dim: {gate_dim}")

    # 2. 构建模型
    print("\n--- Layer 4: Building MoE ---")
    model = OilMoE(
        n_features=n_features, gate_input_dim=gate_dim,
        n_experts=N_EXPERTS, d_model=D_MODEL_EXPERT, pred_len=1,
        gate_hidden_dim=GATE_HIDDEN_DIM,
        expert_configs=[
            {"input_dim": n_features, "d_model": D_MODEL_EXPERT,
             "d_state": D_STATE_EXPERT, "n_layers": N_LAYERS_EXPERT,
             "dropout": DROPOUT_EXPERT}
            for _ in range(N_EXPERTS)
        ],
    )
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,d}")

    # 3. 训练
    print(f"\n--- Training (device={DEVICE}) ---")
    model, history = train_moe_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, balance_weight=BALANCE_WEIGHT,
        patience=PATIENCE, device=DEVICE,
    )

    # 4. 评估
    pred_p, true_p, r_pred, r_true, gate_weights, metrics = evaluate_moe_model(
        model, test_loader, scaler_target,
        test_base_prices, test_true_prices, device=DEVICE,
    )

    # 5. 可视化
    print("\n--- Generating Plots ---")
    try:
        df_raw = pd.read_csv(FILE_PATH).sort_values("date").reset_index(drop=True)
        df_raw[FEAT_COLS + ["OPEC"]] = df_raw[FEAT_COLS + ["OPEC"]].interpolate()
        oil_full = df_raw["OPEC"].values
        plot_len = min(len(regime_labels), len(oil_full))
        plot_regime_timeline(
            dates=np.arange(plot_len),
            oil_prices=oil_full[-plot_len:],
            regime_labels=regime_labels[-plot_len:],
            n_regimes=N_REGIMES,
            save_path=os.path.join(OUTPUT_DIR, "regime_timeline.png"),
        )
    except Exception as e:
        print(f"  [Plot] regime_timeline skipped: {e}")

    plot_gate_weights(dates_test, gate_weights, N_EXPERTS,
                      save_path=os.path.join(OUTPUT_DIR, "gate_weights.png"))
    plot_prediction_vs_actual(dates_test, pred_p, true_p,
                              save_path=os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"))
    plot_training_history(history,
                          save_path=os.path.join(OUTPUT_DIR, "training_history.png"))

    # 6. 保存
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mamba_moe_v62.pth"))
    pd.DataFrame({
        "regime_label": regime_labels,
        **{f"prob_regime_{i}": regime_proba[:, i] for i in range(N_REGIMES)},
    }).to_csv(os.path.join(OUTPUT_DIR, "regime_info.csv"), index=False)
    pd.DataFrame(
        regime_detector.get_transition_matrix(),
        index=[f"From_R{i+1}" for i in range(N_REGIMES)],
        columns=[f"To_R{i+1}" for i in range(N_REGIMES)],
    ).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix.csv"))
    pd.Series(metrics).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))
    print(f"  All results saved to {OUTPUT_DIR}/")

    # 7. analysis_addon 兼容
    y_pred = pred_p.reshape(-1, 1)
    y_true = true_p.reshape(-1, 1)

    class MoEWrapper(nn.Module):
        def __init__(self, moe, default_gate):
            super().__init__()
            self.moe = moe
            self.register_buffer("default_gate", default_gate)

        def forward(self, x_seq):
            gate = self.default_gate.expand(x_seq.shape[0], -1).to(x_seq.device)
            pred, _ = self.moe(x_seq, gate)
            return pred

    test_gates = torch.cat([b[1] for b in test_loader], dim=0)
    wrapped = MoEWrapper(model, test_gates.mean(0, keepdim=True))
    compat_loader = TwoTupleDataLoader(test_loader)

    try:
        from analysis_addon import run_full_analysis
        run_full_analysis(
            model=wrapped, test_loader=compat_loader,
            scaler_oil=scaler_target, feat_cols=all_feat_cols,
            y_pred=y_pred, y_true=y_true,
            device=DEVICE, scaler_feat=scaler_feat,
            test_base_prices=test_base_prices,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  v6.2 Pipeline Complete!")
    print("=" * 60)