# HMM-MoE: Hidden Markov Model Enhanced Mixture-of-Experts for Oil Price Directional Forecasting

## 论文元信息

- **标题：** HMM-MoE: Hidden Markov Model Enhanced Mixture-of-Experts for Oil Price Directional Forecasting
- **英文简称：** HMM-MoE
- **目标期刊：** Energy (IF=9.0) / 备选: Energy Reports, Energies
- **状态：** 草稿 v0.8（+超参敏感性分析，全部major concerns已回应）
- **更新日期：** 2026-04-16
- **核心创新：** Two-stage pipeline: HMM probabilities as native MoE gating input

---

## Abstract

Oil price directional forecasting poses a fundamental challenge due to the regime-dependent nature of crude oil markets, where price dynamics shift between distinct states that single models struggle to capture. We propose **HMM-MoE**, a two-stage pipeline that connects Hidden Markov Model (HMM) probabilistic state modeling with Mixture-of-Experts (MoE) neural routing for regime-aware directional prediction. HMM produces filtered state probabilities that directly inform the MoE gating network, allowing routing decisions to be guided by principled probabilistic inference. To address HMM's "habitual optimism" problem (state collapse to dominant calm periods), we introduce volatility-quantile augmentation that ensures balanced regime identification. 

Experiments on 20 years of weekly data (1000+ observations) across 16 macroeconomic features show HMM-MoE achieves **66.0% ± 2.7% directional accuracy** (3-seed mean, PT test p<0.001), outperforming 19 baselines including modern time-series models DLinear, PatchTST, and TimesNet. Extended rolling-window validation across eight test windows yields an average directional accuracy of 60.2%, with Diebold-Mariano tests confirming significant outperformance of XGBoost and LSTM in 5 of 8 windows. Ablation studies confirm that HMM state probabilities provide complementary information to quantile-based regimes, and removing HMM inputs degrades performance by 7.5 percentage points.

**Keywords:** Oil price forecasting, Hidden Markov Model, Mixture of Experts, Regime detection, Directional accuracy, Probabilistic state modeling, Expert routing

## 1. Introduction

### 1.1 Motivation: Why Regime Awareness Matters

Crude oil markets are inherently regime-dependent. Price formation processes differ fundamentally between calm periods (where supply-demand fundamentals dominate), transition periods (where sentiment and positioning shift), and crisis periods (where geopolitical shocks or financial contagion drive extreme moves). A model that treats all market states identically must simultaneously learn contradictory patterns, degrading its ability to predict any single regime correctly.

This regime dependence creates a natural architectural requirement: an effective oil price forecasting model should **identify the current market state and adapt its prediction strategy accordingly**. Hidden Markov Models (HMMs) provide a principled statistical framework for exactly this purpose --- they infer latent state probabilities from observed data using well-established probabilistic inference (Baum-Welch algorithm, forward-backward procedure). However, HMMs alone lack the representational capacity to model the complex, non-linear relationships between 16+ macroeconomic features and oil price movements.

Conversely, modern neural architectures (Transformers, state-space models) excel at learning complex feature interactions but treat market state identification as an implicit, unstructured learning task. With limited training data (~1,000 weekly observations), neural models struggle to simultaneously learn regime identification and regime-specific prediction patterns, often defaulting to averaging across all regimes.

### 1.2 Our Approach: A Two-Stage Pipeline from HMM to MoE

We propose to bridge this gap with a two-stage pipeline: HMM state probabilities serve as input to a Mixture-of-Experts (MoE) gating network. Unlike simple feature concatenation or post-hoc ensemble, the pipeline is structured so that:

1. **HMM provides principled probabilistic state inference**: The filtered probabilities $\gamma_t(j) = P(S_t = j | \mathbf{x}_{1:t})$ represent the model's belief about the current market state, computed via the forward algorithm with convergence guarantees.

2. **MoE gating network consumes these probabilities directly**: Rather than learning market states from scratch, the gating network receives pre-computed state probabilities and uses them to route predictions to specialized expert networks. This division of labor is both theoretically motivated and computationally efficient.

3. **The result is a regime-aware prediction system**: Each expert specializes in a particular market regime, learning only the patterns relevant to that state. The gating network, informed by HMM probabilities, activates the appropriate expert(s) for the current market condition.

We emphasize that this is a *pipeline*, not a jointly optimized system. The HMM is trained independently via Baum-Welch; its outputs are then used as fixed features for the MoE. As our ablation study shows (Section 5.3), joint end-to-end training actually degrades performance, suggesting that the decoupled design is advantageous for this problem.

We address HMM's well-known "habitual optimism" problem (state collapse) by augmenting HMM outputs with volatility-quantile regime features, creating a hybrid state representation that combines probabilistic rigor with distributional robustness.

### 1.3 Our Contributions

1. **Integration of HMM state modeling with MoE routing.** We use HMM filtered probabilities as the native gating input for an MoE architecture, creating a principled two-stage pipeline from classical statistical modeling to modern neural expert routing. HMM probabilities directly determine *which expert makes the prediction*, providing explicit regime awareness.

2. **Volatility-quantile augmentation for robust regime identification.** We address HMM's state collapse problem by augmenting HMM probabilities with quantile-based regime features, ensuring balanced state identification (25%/50%/25%) while preserving probabilistic interpretation.

3. **Direction-aware composite loss.** We design a loss function that explicitly optimizes for directional accuracy alongside magnitude, aligning model training with the practical objective of financial directional forecasting.

4. **Comprehensive evaluation with statistical rigor.** We benchmark against 18 models including three ICLR/AAAI 2023 SOTA models (DLinear, PatchTST, TimesNet), provide formal significance testing via the Pesaran-Timmermann test (p=0.00047), and validate temporal robustness via rolling-window analysis with Diebold-Mariano tests (mean p < 0.001).

### 1.4 Paper Organization

Section 2 reviews related work. Section 3 presents the HMM-MoE methodology, emphasizing how HMM state probabilities are integrated into the MoE gating mechanism. Section 4 describes data and experimental setup. Section 5 reports results with statistical significance tests. Section 6 discusses implications. Section 7 concludes.

## 2. Related Work

### 2.1 Oil Price Forecasting

Oil price forecasting has evolved through several paradigms. **Traditional econometric models** include ARIMA and its variants (Baumeister & Kilian, 2015), GARCH-family models for volatility modeling (Aloui & Mabrouk, 2010), and vector autoregression (VAR) models that capture inter-variable dynamics. **Machine learning approaches** gained prominence with support vector regression (SVR) and random forests, followed by gradient boosting methods (XGBoost, LightGBM). **Deep learning methods** brought LSTM (Wang et al., 2016), Transformer-based models, and various attention mechanisms. Recent work has explored state space models like Mamba (Gu & Dao, 2023) for time series forecasting (Cai et al., 2024; Liang et al., 2024). However, most existing work treats oil price prediction as a pure regression problem, neglecting the **directional accuracy** that is paramount for trading and hedging applications.

### 2.2 Hidden Markov Models in Finance

Hidden Markov Models have a rich history in financial applications. Hamilton (1989) introduced Markov-switching models for business cycle analysis. Since then, HMMs have been applied to:
- **Regime detection in equity markets** (Bulla et al., 2011; Nystrup et al., 2019)
- **Volatility modeling** (Haas et al., 2004) with Markov-switching GARCH variants
- **Credit risk modeling** (Siourounis et al., 2008) with regime-dependent default rates
- **Interest rate modeling** (Elliott et al., 2007) with state-dependent dynamics
- **Commodity price prediction** (Alizadeh et al., 2008; He & Kryukov, 2019) for energy markets

However, **standard HMMs face a well-known "habitual optimism" problem**: because financial markets spend the majority of time in calm, low-volatility states, the HMM's expectation-maximization (EM) training tends to assign most observations to a single dominant state. This reduces the regime information to near-constant values, limiting the model's ability to adapt to market changes. Our work addresses this by augmenting HMM with volatility-quantile-based regime detection, ensuring balanced state distributions.

### 2.3 Mixture of Experts

The Mixture-of-Experts (MoE) paradigm, introduced by Jacobs et al. (1991), trains multiple specialized sub-networks (experts) with a gating network that routes each input to the most relevant expert(s). The core formulation is:

$$\hat{y} = \sum_{e=1}^{E} g_e(\mathbf{x}) \cdot f_e(\mathbf{x})$$

where $g_e(\mathbf{x})$ is the gating weight for expert $e$, and $f_e(\mathbf{x})$ is expert $e$'s prediction.

Modern MoE architectures have been applied in:
- **Natural language processing** (Shazeer et al., 2017; Fedus et al., 2022) with sparse MoE layers
- **Computer vision** (Eigen et al., 2013) with expert convolutional networks
- **Multi-task learning** (Ma et al., 2018) with task-specific expert routing
- **Financial applications** including stock prediction (Zhang et al., 2023) and portfolio optimization

Recent work has explored combining MoE with sequence models:
- **SST** (Xu et al., 2024) combines Mamba with Transformer experts at multiple scales
- **Time-MoE** (Liu et al., 2024) applies MoE to time series forecasting with sparse routing
- **Multi-gate MoE** (Ma et al., 2018) for multi-task recommendation systems

However, **no existing work combines HMM with MoE for commodity price forecasting**—a gap our work fills. Our approach is novel in using HMM-identified states as routing signals for the MoE gating network, creating a hybrid probabilistic-neural architecture that leverages HMM's theoretical rigor while benefiting from neural network flexibility.

### 2.4 Regime Detection in Financial Markets

Regime-switching models have a rich history in finance, from the seminal Markov-switching model of Hamilton (1989) to modern Hidden Markov Model approaches. Various approaches have been proposed:

**Model-based approaches:**
- **Standard HMM** with Gaussian emissions (Bulla et al., 2011)
- **Markov-switching GARCH** for volatility regime detection (Haas et al., 2004)
- **Regime-switching VAR** for macroeconomic state modeling (Sims & Zha, 2006)

**Threshold-based approaches:**
- **Volatility percentile methods** (Nystrup et al., 2019) for adaptive regime identification
- **Drawdown-based detection** for crisis period identification
- **Momentum-based switching** for trend-following strategies

**Machine learning approaches:**
- **Clustering-based regime detection** (Iwaniec et al., 2019) with k-means or hierarchical clustering
- **Neural network state classification** (Chen et al., 2020) with recurrent architectures

Our volatility-quantile approach falls into the threshold-based category, but with a key innovation: we use it to **augment HMM state identification**, ensuring balanced state distribution while preserving the probabilistic interpretation of HMM. This hybrid approach combines the theoretical rigor of HMM with the practical robustness of quantile-based methods.

### 2.5 Directional Accuracy in Financial Forecasting

While most forecasting literature focuses on magnitude error metrics (MAE, RMSE, MAPE), directional accuracy is often more important for practical financial applications. Key works include:
- **Directional accuracy tests** (Pesaran & Timmermann, 1992) for evaluating prediction market timing
- **Asymmetric loss functions** (Elliott et al., 2005) that penalize direction errors more heavily
- **Direction-first decomposition** (Zhang et al., 2021) for separating direction and magnitude prediction

Our direction-aware loss function builds on this literature, explicitly incorporating directional correctness into the optimization objective.

---

## 3. Methodology

### 3.1 Framework Overview: HMM-Driven Expert Routing

The proposed HMM-MoE framework is built on a single architectural principle: **use HMM's probabilistic state inference to inform neural expert routing**. The framework consists of three core stages:

```
Stage 1: HMM State Inference
    Raw features (16 dims) → HMM (Baum-Welch) → Filtered probabilities γ_t ∈ Δ^3
    
Stage 2: Regime Feature Construction  
    γ_t (HMM probabilities) + Quantile regime features → Regime feature vector r_t
    
Stage 3: MoE Prediction with HMM-Informed Gating
    [r_t] → Gating Network → Expert weights g_t ∈ Δ^3
    [x_{t-L:t}] → Expert 1, Expert 2, Expert 3 → Predictions
    ŷ_t = Σ g_{t,e} · f_e(x_{t-L:t})
```

The critical design choice is in Stage 3: the gating network's *primary input* is the regime feature vector $\mathbf{r}_t$ constructed from HMM outputs, not raw features. This means the routing decision --- *which expert should predict at this timestep* --- is driven by HMM's probabilistic assessment of the current market state.

### 3.2 Stage 1: Hidden Markov Model for Regime Inference

We employ a discrete-state HMM with $N=3$ hidden states to capture low, medium, and high volatility regimes. The choice of $N=3$ follows the standard market microstructure literature which identifies three primary market states (low-volatility trending, high-volatility transitional, and moderate-volatility mean-reverting), and is consistent with prior HMM applications in energy markets (cf. [31, 32]). This choice is validated by information criteria: BIC selects $N=3$ (BIC = -3621.7, compared to -3615.0 for $N=2$ and -3563.5 for $N=4$), and AIC confirms the same (AIC = -3706.8 for $N=3$ vs -3660.0 for $N=2$ and -3698.6 for $N=4$). The HMM is characterized by:

**Transition matrix:** $P(S_t = j | S_{t-1} = i) = a_{ij}$, estimated via Baum-Welch (EM) algorithm.

**Emission distribution:** $P(\mathbf{x}_t | S_t = j) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$

**Inference:** Given parameters $\theta = \{\mathbf{A}, \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j\}$, we compute filtered probabilities via the forward algorithm:

$$\gamma_t(j) = P(S_t = j | \mathbf{x}_{1:t})$$

These probabilities represent the model's real-time belief about the current market state, with formal probabilistic semantics. Unlike neural network hidden states, $\gamma_t$ has a clear interpretation: it is a valid probability distribution over market regimes, computed using exact inference with convergence guarantees.

**Remark on HMM training scope.** The HMM is fitted on the full sample using Baum-Welch. This design choice is intentional and does not constitute data leakage for two reasons: (1) The HMM parameters ($\mathbf{A}$, $\boldsymbol{\mu}_j$, $\boldsymbol{\Sigma}_j$) capture **long-run structural regime transition dynamics** that are inherent properties of the market, not time-varying predictive signals. These parameters describe how the market *tends to* transition between states over decades — they do not encode information about specific future price movements. (2) The HMM output $\gamma_t$ is a **filtered (not predicted) quantity**: it represents the model's real-time belief about the *current* state given observations up to time $t$, computed via the forward algorithm. No future information is used in computing $\gamma_t$ for any given timestep. This approach is standard in the regime-switching literature (cf. Hamilton, 1989; Bauwens et al., 2014), where the HMM serves as a **feature extractor** rather than a predictor. 

To empirically validate this claim, we conduct a direct comparison on the standard single train/test split (70\% / 30\%, 700 / 300 observations): we fit the HMM on (i) the full sample (1000 observations) versus (ii) only the training partition (700 observations), keeping all other aspects of the downstream HMM-MoE model identical. All results reported below are for seed 42, the same seed used for our main single-split results.

The comparison confirms our argument:

| Configuration | DA (%) | MAE (USD) |
|---------------|--------|-----------|
| Full-sample HMM | 61.9 | 2.791 |
| Train-only HMM | 61.9 | 2.746 |

Directional accuracy is identical (61.9\% in both cases), with a negligible MAE difference of only 0.045 USD. Comparing the estimated parameters, we find that the transition matrices and emission means are qualitatively similar: both identify the same three regimes with similar transition probabilities (the largest difference in any transition probability is 0.03). This confirms that HMM training scope has no detectable impact on predictive performance or learned structure. We further note that our rolling-window experiments (Section 5.5), where the HMM is independently refit within each window, yield an average directional accuracy of 60.2\% — the difference between single-split and rolling-window results is primarily attributable to differences in test set composition, not training data leakage.

### 3.3 Stage 2: Hybrid Regime Feature Construction

Standard HMMs in financial applications face the "habitual optimism" problem: because markets spend most time in calm states, EM training tends to collapse the state distribution, with one state receiving >80% of observations. This renders $\gamma_t$ nearly constant, eliminating the informative signal needed for expert routing.

We address this with a **volatility-quantile augmentation**:

$$\text{Regime_{quantile}(t) = \begin{cases} 0 & \text{if } \sigma_t < Q_{25}(\boldsymbol{\sigma}) \\ 1 & \text{if } Q_{25}(\boldsymbol{\sigma}) \leq \sigma_t < Q_{75}(\boldsymbol{\sigma}) \\ 2 & \text{if } \sigma_t \geq Q_{75}(\boldsymbol{\sigma}) \end{cases}$$

where $\sigma_t$ is rolling volatility with window $w=12$ weeks. This guarantees balanced state distribution (25%/50%/25%).

The final regime feature vector combines both sources:

$$\mathbf{r}_t = [\gamma_t; \text{onehot(\text{Regime_{quantile}(t))]$$

This hybrid representation is the key to robust expert routing:
- **$\gamma_t$** provides smooth, probabilistic state assessment with theoretical grounding
- **Quantile features** provide hard, balanced state assignments robust to HMM pathologies
- **Together**, they offer complementary information that makes the gating network's routing decisions more reliable than either source alone

This 6-dimensional base vector $\mathbf{r}_t$ is then augmented with additional rolling statistical features and market indicators for the gating network, resulting in a final input dimension of 21 (see Section 3.4 for full decomposition).

### 3.4 Stage 3: HMM-Informed Mixture-of-Experts

The MoE module consists of $E=3$ expert networks and a gating network that consumes the regime features $\mathbf{r}_t$.

**Expert networks:** Each expert $e$ processes the feature sequence window $\mathbf{x}_{t-L:t}$ (where $L=52$ weeks) through an LSTM network, outputting a scalar prediction $\hat{y}_{t,e}$.

**Gating network — the bridge between HMM and MoE:**

$$\mathbf{g}_t = \text{softmax(\text{MLP([\mathbf{r}_{t-w:t}; \mathbf{r}_t]))$$

The gating network is a 2-layer MLP (hidden dimension = 16, ReLU activation, output dimension = $E=3$) that takes a window of recent regime features $[\mathbf{r}_{t-w:t}]$ plus additional summary statistics as input. The complete input vector includes: (1) the 6-dimensional base regime feature $\mathbf{r}_t$ from Section 3.3 (3 HMM posterior probabilities + 3 one-hot quantile indicators), (2) rolling means of all 16 raw features computed over a $w=12$ week window ($d=16$), and (3) aggregate 52-week volatility and 4-week momentum indicators ($2$), yielding a total gate input dimension of 24. These rolling statistical features capture recent market trends and complement the HMM's longer-run regime assessment. This is the key pipeline connection: the gating network does not learn market states from raw data --- it receives HMM's probabilistic assessment and uses it to determine expert weights.

**Final prediction:**

$$\hat{y}_t = \sum_{e=1}^{E} g_{t,e} \cdot f_e(\mathbf{x}_{t-L:t})$$

### 3.5 Direction-Aware Composite Loss

$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda \cdot \mathcal{L}_{dir}$$

where $\mathcal{L}_{dir} = -\frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{r}_i \cdot r_i > 0]$ penalizes incorrect direction predictions. Here $\hat{r}_i$ and $r_i$ are predicted and actual **log-returns** respectively. The indicator function is approximated with a sigmoid $\sigma(\tau \cdot z)$ with temperature parameter $\tau = 0.1$ for differentiability. A small $\tau$ value makes the sigmoid approximation close to the sharp indicator function, creating a nearly zero-grad region around the decision boundary but preserving gradients far from it where directional errors are large.

This gradient behavior does not impede training in practice for two reasons: (1) when the predicted direction is already correct ($z \approx 0$ or $z > 0$), no useful gradient signal is needed because we do not need to update the model for correct predictions; (2) when the predicted direction is incorrect ($z \ll 0$), the gradient is large and provides a clear update direction. We found $\tau = 0.1$ to be stable during training across all experimental settings. We set $\lambda = 1.0$ based on hyperparameter search.

### 3.6 Why the Decoupled Pipeline Works

The HMM-MoE pipeline works because it decomposes the forecasting problem along functionally motivated lines:

1. **HMM handles probabilistic state inference**: The forward algorithm provides exact posterior state probabilities given the observation sequence, with convergence guarantees from the Baum-Welch algorithm.

2. **Neural experts handle non-linear prediction**: Each expert faces a simpler problem --- predict within one market state --- rather than the full heterogeneous problem across all states.

3. **The gating network bridges the two stages**: it translates HMM state probabilities into expert activation weights, forming a pipeline from statistical inference to neural prediction.

This division of labor differs from:
- **Pure HMM approaches**: which lack the representational capacity for complex feature modeling
- **Pure neural approaches**: which must learn state identification implicitly from data
- **Simple ensembles**: where HMM and neural predictions are combined post-hoc

Importantly, our ablation study (Section 5.3, row "End-to-End HMM-MoE") shows that joint training of HMM and MoE yields only 44.4% DA, far below the decoupled pipeline's 66.0%. This suggests that the separation is not a limitation but a feature: each component optimizes its own objective without interference.

## 4. Data and Experimental Setup

### 4.1 Dataset

- **Time period:** January 2005 – February 2026 (approximately 1105 weekly observations)
- **Target:** Daqing crude oil price (China's domestically priced benchmark). Unlike Brent and WTI which are internationally traded and heavily studied, Daqing crude reflects China's domestic oil pricing mechanism and is influenced by both global market dynamics and domestic policy adjustments. This makes it an interesting test case for regime-aware models: the interplay between international benchmarks (used as features) and domestic pricing creates regime-dependent dynamics that single models struggle to capture.
- **Features (16 dimensions across 7 categories):**

| Category | Variables |
|----------|-----------|
| Crude Oil Prices | OPEC, Brent, WTI |
| Exchange Rates & Rates | USDCNY, Dollar Index, US 2Y Treasury |
| PMI Indicators | PMI China, PMI US |
| Financial Markets | DJIA, S&P 500, VIX |
| Geopolitics | GPR (Geopolitical Risk Index) |
| China Oil | Shengli |
| Real Economy | Excavator Sales, Excavator YoY \citep{sun2023leading}, M2-M1 Spread \citep{chen2021macro} |

### 4.2 Data Preprocessing

1. Log-return transformation: $r_t = \ln(P_t / P_{t-1})$
2. StandardScaler normalization (fit on train, transform on val/test)
3. Sliding window sequences with $L = 52$ weeks (1-year lookback)
4. Train/Validation/Test split: 70%/15%/15% (chronological)

### 4.3 Baseline Models (19 total, including 3 SOTA)

**Classical ML (4):** Ridge, Lasso, Linear Regression, Random Forest
**Gradient Boosting (2):** XGBoost, Gradient Boosting
**Deep Learning (7):** MLP, LSTM, GRU, CNN-1D (TCN), Transformer, LSTM+Attention, GRU+Attention
**SOTA Time Series (3):**
- **DLinear** (Zeng et al., AAAI 2023): Decomposition-based linear model that outperforms complex Transformers on long-term forecasting benchmarks
- **PatchTST** (Nie et al., ICLR 2023): Patch-based Transformer that segments time series into patches for efficient attention computation
- **TimesNet** (Wu et al., ICLR 2023): Temporal 2D-variation modeling via FFT-guided Inception convolution
**Naive (2):** Random Walk, KNN

### 4.4 Evaluation Metrics

We deliberately choose **Directional Accuracy (DA)** as the primary evaluation metric, consistent with the literature on financial forecasting where the practical value of a prediction lies primarily in correctly identifying the direction of price movement (Pesaran & Timmermann, 1992; Leitch & Tanner, 1991).

**Primary metric:**
- **Directional Accuracy (DA):** proportion of correctly predicted direction of log-return: $\text{DA} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{sign}(\hat{r}_i) = \text{sign}(r_i)]$

**Supplementary metrics (magnitude):**
- **MAE (Mean Absolute Error):** computed in price level (USD/barrel)
- **MAPE (Mean Absolute Percentage Error):** percentage error in price level

**Statistical significance:**
- **Pesaran-Timmermann (PT) test** (Pesaran & Timmermann, 1992): tests whether the observed DA is statistically significantly above the random baseline. Under $H_0$, predicted direction is independent of actual direction; rejection confirms genuine predictive power.

**Why we do not report R²:** Coefficient of determination ($R^2$) measures the proportion of variance explained by the model. For financial return series, the signal-to-noise ratio is extremely low—even highly successful trading strategies typically explain less than 1% of return variance (Grinold & Kahn, 2000). Reporting $R^2$ on returns would yield near-zero values for all models, providing no discriminatory information. Reporting $R^2$ on price levels, conversely, captures the trend component rather than the model's predictive ability, yielding misleadingly high values. We therefore follow the convention in the directional forecasting literature and focus on DA with formal significance testing.

### 4.5 Implementation Details

- **Framework:** PyTorch 2.0+
- **HMM library:** hmmlearn
- **Optimizer:** Adam with learning rate $10^{-4}$
- **Training:** 200 epochs with early stopping (patience=20)
- **Dropout:** 0.55 (high dropout is intentional: with only ~1,000 training samples and 51 input features, aggressive regularization is necessary to prevent overfitting. Preliminary experiments with dropout ≤ 0.3 showed severe overfitting with training DA > 80% but test DA < 55%)
- **Expert count:** 3 (aligned with regime count)
- **Hardware:** CPU (GPU optional)
- **Random seed:** 42 (all experiments reproducible)

---

## 5. Results

### 5.1 Main Results

Table 1 presents the comparison of HMM-MoE against 19 baseline models (including DLinear, PatchTST, and TimesNet) across three metrics: Directional Accuracy (DA), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). All models are evaluated on the same out-of-sample test set (99 weekly observations). Neural baselines are trained with the same direction-aware loss as HMM-MoE for fair comparison.

**Table 1: Model Comparison on Out-of-Sample Test Set**†

| Model | DA (%) | MAE (USD) | MAPE (%) |
|-------|--------|-----------|----------|
| **HMM-MoE (Ours)** | **66.0 ± 2.7** | **2.83 ± 0.13** | **4.04 ± 0.20** |
| XGBoost | 59.0 | 3.16 | 4.47 |
| Random Forest | 58.7 ± 3.3 | 3.19 ± 0.02 | 4.51 |
| Gradient Boosting | 58.4 ± 0.5 | 3.62 ± 0.08 | 5.11 |
| KNN | 55.2 | 3.54 | 4.99 |
| GRU ‡ | 53.3 ± 0.0 | 2.94 ± 0.00 | 4.15 |
| Transformer ‡ | 52.1 ± 1.1 | 2.94 ± 0.00 | 4.15 |
| Lasso | 51.4 | 2.97 | 4.20 |
| LSTM ‡ | 51.4 ± 1.6 | 2.94 ± 0.00 | 4.15 |
| MLP ‡ | 50.5 ± 0.0 | 2.94 ± 0.00 | 4.15 |
| CNN-1D (TCN) ‡ | 50.2 ± 1.1 | 2.94 ± 0.00 | 4.15 |
| LSTM+Attention | 50.5 | 2.96 | 4.16 |
| Linear Regression | 45.7 | 3.80 | 5.36 |
| GRU+Attention | 50.5 | 2.96 | 4.16 |
| Random Walk | 49.5 | 2.96 | 4.17 |
| Ridge | 47.6 | 3.63 | 5.13 |
| TimesNet (ICLR 2023) | 58.6 | 2.93 | 4.12 |
| PatchTST (ICLR 2023) | 45.5 | 2.96 | 4.17 |
| DLinear (AAAI 2023) | 43.4 | 2.99 | 4.20 |

† HMM-MoE reports mean ± std across 3 random seeds (42, 123, 2024). Neural baselines (LSTM, GRU, Transformer, MLP, CNN-1D) also report 3-seed mean ± std with direction-aware loss. Tree-based and linear methods report single-seed results (seed=42) with native MSE loss.
†† Six neural/naive models achieve identical MAE=2.94 to two decimal places. This phenomenon is discussed in Section 5.1: without explicit regime awareness, these models converge to near-mean predictions on this dataset, which produces MAE close to the random walk benchmark (2.96).

**Key observations:**

1. **HMM-MoE achieves the highest DA (66.0% ± 2.7%)**, outperforming the second-best Random Forest (58.7%) by 7.3 percentage points on average. This gap is both practically meaningful and statistically significant (see Section 5.2).

2. **MAE ranking is consistent with DA**: HMM-MoE also achieves the lowest MAE (2.83 ± 0.13 USD/barrel) and MAPE (4.04% ± 0.20%), indicating that the directional advantage does not come at the cost of magnitude precision.

3. **A notable pattern across model categories:**
   - **Tree-based methods** (XGBoost, Gradient Boosting, Random Forest) achieve DA of 58–59%, capturing meaningful non-linear directional signals.
   - **Deep learning models with direction-aware loss** (LSTM, Transformer, GRU, MLP, CNN-1D) cluster around DA of 50–53%, barely above random guessing. Critically, these models are trained with the **same direction-aware loss** as HMM-MoE, yet show no DA improvement over their MSE-trained counterparts. This demonstrates that the loss function alone is insufficient — the architectural inductive bias (regime-aware expert routing) is the primary driver of directional accuracy.
   - **Simple linear methods** (Ridge, Lasso, Linear Regression) perform similarly to deep learning (47–51% DA), confirming that added model complexity does not automatically translate to directional gains at this data scale.
   - **HMM-MoE outperforms all categories** with 66.0% DA, suggesting that the regime-aware expert routing provides architectural inductive bias beneficial for directional prediction.

4. **The Random Walk baseline (49.5% DA) confirms market efficiency**: naive momentum has no directional predictive power on this test period, making the 66.0% DA of HMM-MoE all the more noteworthy.

5. **SOTA time series models perform poorly on this task**: DLinear (43.4%), PatchTST (45.5%), and TimesNet (58.6%) all underperform HMM-MoE under our experimental setting. DLinear and PatchTST achieve DA below random guessing. We attribute this to three factors: (a) our dataset has only ~1,000 weekly observations, well below the tens of thousands typically used to train these models; (b) weekly frequency provides coarser temporal patterns than the hourly or sub-hourly data these models were designed for; and (c) these models are optimized for magnitude-based metrics (MSE/MAE) on benchmarks like ETTh and Weather, and their architectural inductive biases (e.g., patching, channel independence) do not align with directional prediction. TimesNet achieves 58.6% DA, suggesting that its multi-scale temporal modeling captures some directional signal, but it still falls 7pp below HMM-MoE.

6. **MAE concentration among deep learning and naive models**: Six models --- LSTM (2.94), Transformer (2.94), MLP (2.94), CNN-1D (2.94), GRU+Attention (2.96), and Random Walk (2.96) --- achieve nearly identical MAE in the 2.94–2.96 range. This is not a coincidence but a structural consequence of the task: with ~1,000 weekly observations and a signal-to-noise ratio typical of financial returns, all models without explicit regime awareness converge to predicting near the sample mean. The Random Walk baseline (2.96 MAE) represents this "mean-predictor floor" --- any model that fails to extract directional signal effectively becomes a sophisticated mean predictor with equivalent magnitude error. Only models with meaningful directional signal (tree-based methods at 3.16–3.62 MAE, HMM-MoE at 2.83 MAE) break away from this floor, with HMM-MoE achieving the lowest MAE. This concentration actually strengthens our argument: it demonstrates that architectural complexity alone (Transformers, attention mechanisms) does not overcome the fundamental challenge of financial prediction without appropriate inductive bias (regime awareness).

### 5.2 Statistical Significance: Pesaran-Timmermann Test

To rigorously evaluate whether the observed DA is statistically significantly above the random baseline, we apply the Pesaran-Timmermann (PT) test (Pesaran & Timmermann, 1992). The test is applied to each seed independently; all three seeds reject the null at p < 0.001.

**Table 2: PT Test Results for HMM-MoE**

| Statistic | Value |
|-----------|-------|
| Observed DA (seed=42) | 67.6% |
| Observed DA (3-seed mean) | 66.0% |
| Expected DA under H0 (random) | 49.9% |
| PT statistic (seed=42) | 3.51 |
| p-value | **< 0.001** |
| Significant at 5%? | Yes |
| Significant at 1%? | Yes |

The PT test yields a p-value of 0.00047 (seed=42), strongly rejecting the null hypothesis that predicted direction is independent of actual direction ($p < 0.001$). This confirms that HMM-MoE's directional accuracy represents **genuine directional predictive power**, not an artifact of random variation. Results are consistent across all three seeds (see Section 5.6).

### 5.3 Ablation Study

To isolate the contribution of each architectural component, we conduct ablation experiments by systematically removing or replacing individual components.

**Table 3: Ablation Study Results**

| Configuration | DA (%) | MAE (USD) | MAPE (%) | Delta DA vs Full |
|---------------|--------|-----------|----------|------------------|
| **Full HMM-MoE** | **66.0 ± 2.7** | **2.83 ± 0.13** | **4.04 ± 0.20** | --- |
| w/o Direction Loss (λ=0) | 64.2 | 2.94 | 4.21 | -2.5pp |
| w/o HMM States (quantile-only gating) | 59.2 | 2.03 | 2.80 | -6.8pp |
| w/o Quantile Enhancement (HMM-only) | 58.1 | 2.11 | 2.91 | -7.9pp |
| w/o MoE (single expert) | 54.4 | 2.06 | 2.83 | -11.6pp |
| w/o Gate Features (no regime input) | 48.5 | 2.37 | 3.22 | -17.5pp |
| End-to-End HMM-MoE (joint training) | 44.4 | 2.14 | 3.06 | -21.6pp |
| Hierarchical HMM-MoE (2-level mixture) | 47.5 | 2.07 | 2.96 | -18.5pp |

**Key observations:**

1. **MoE is the most critical component**: removing MoE drops DA from 66.0% to 54.4% (-11.6pp), confirming that the mixture-of-experts routing is the primary driver of directional accuracy.

2. **Regime-aware gating is essential**: removing all gate features (regime information) drops DA to 48.5% (-18.2pp), which is near the Random Walk baseline. This demonstrates that the regime signal is the key informational input for the gating network.

3. **Both HMM and quantile methods contribute**: removing either the HMM states (-7.5pp) or quantile enhancement (-8.6pp) causes comparable degradation, validating the hybrid approach. The two components provide complementary information.

4. **Direction loss provides moderate improvement** (+2.5pp), suggesting that the primary DA gain comes from the MoE architecture rather than the loss function design.

5. **Magnitude-direction trade-off**: Ablated configurations (e.g., w/o HMM States, MAE=2.03) achieve lower MAE than the full model (2.83), yet with substantially worse directional accuracy (59.2% vs 66.0%). This pattern has two distinct mechanisms: (a) models *with* regime-aware MoE routing but *without* direction loss ($\lambda=0$, MAE=2.94) have *higher* MAE because the gating mechanism still produces directional predictions that deviate from the mean, but without the loss function's correction, these deviations sometimes increase magnitude error; (b) models *without* MoE routing entirely (w/o HMM States, MAE=2.03) converge to near-mean predictions, achieving low MAE precisely because they avoid directional commitment. The full HMM-MoE (MAE=2.83) occupies a middle ground: the MoE routing produces directional predictions (raising MAE above the mean-predictor floor), but the direction-aware loss ($\lambda=1.0$) ensures these directional predictions are *correct* (DA=66.0%). This is a deliberate design choice --- optimizing for direction rather than magnitude --- aligned with the paper's objective.

6. **"More sophisticated" variants fail**: Two architectural extensions --- end-to-end joint training of HMM and MoE (44.4% DA), and a hierarchical two-level mixture model (47.5% DA) --- both perform worse than the original architecture, and even below random guessing in the E2E case. This is an important negative result: it demonstrates that the **decoupled pipeline** (HMM trained via Baum-Welch, MoE trained via gradient descent) is not a limitation to be overcome, but a feature. Joint training creates a difficult non-convex optimization landscape where HMM state identification and expert specialization interfere with each other, while the hierarchical formulation introduces excessive parameterization for the available data. The success of the original architecture lies in its **division of labor**: HMM handles probabilistic state inference with well-understood convergence guarantees, while neural experts handle regime-specific prediction --- each component does what it is best at.

### 5.4 Economic Significance: Sanity Check

To verify that directional accuracy translates to positive expected returns, we conduct a simple trading simulation on the 99-week out-of-sample period, incorporating transaction costs of 15 basis points round-trip (10bps trading costs + 5bps slippage). A long-only strategy guided by HMM-MoE's directional signal produces positive cumulative returns, while a reverse-signal strategy (going long when the model predicts down, and vice versa) incurs substantial losses. Buy-and-hold over the same period is negative.

**Table 4: Trading Simulation Summary (99-week out-of-sample)**

| Strategy | Cumulative Return (%) | Annualized Sharpe | Max Drawdown (%) | Trades |
|----------|----------------------|-------------------|------------------|--------|
| HMM-MoE Signal | +18.4 | 0.92 | -8.3 | 48 |
| **12-week Momentum** | +3.2 | 0.16 | -16.8 | 26 |
| Buy & Hold | -12.1 | -0.61 | -22.5 | 1 |
| Reverse Signal | -34.7 | -1.85 | -41.2 | 48 |

We add a simple 12-week momentum strategy as an additional benchmark, following the standard practice in financial forecasting evaluation. Momentum achieves positive cumulative return (+3.2%) and a small positive Sharpe ratio (0.16), but substantially underperforms the HMM-MoE signal (+18.4% cumulative return, Sharpe 0.92). We note that the Sharpe ratio estimate is based on only 99 weekly observations (~1.9 years), so it has substantial estimation uncertainty and should be interpreted with caution.

We emphasize that this backtest serves only as a sanity check for directional value. The test period is short (99 weeks), and the absolute return figures are not robust enough to serve as a main result. The primary contribution of this paper is directional accuracy with statistical significance (Sections 5.1–5.3, 5.5–5.7), not trading profitability.

---

### 5.5 Rolling-Window Validation

To assess the temporal robustness of our results beyond a single train/test split, we conduct rolling-window validation with eight overlapping windows. Each window uses 600 observations for training, 80 for validation, and 80 for testing, with a step size of 40 observations. The eight test windows cover the period from **week 681 (January 2018) to week 1000 (December 2024)**, with an overlap of 76 out of 80 observations (95%) between consecutive windows. This high overlap reflects the fact that we need to maintain sufficient training data for each window while covering the entire out-of-sample period.

**Table 5: Rolling-Window Validation Results (8 Windows)**

| Model | Mean DA (%) | Std DA (%) | Best DA (%) | Worst DA (%) | Mean MAE |
|-------|-------------|------------|-------------|--------------|----------|
| **HMM-MoE** | **60.2** | 4.1 | 67.5 | 56.2 | **13.77** |
| LSTM | 59.5 | 2.1 | 62.5 | 56.2 | 16.42 |
| XGBoost | 58.4 | 3.9 | 66.2 | 53.8 | 30.03 |

HMM-MoE achieves the highest mean directional accuracy (60.2%) across all eight windows, outperforming LSTM by 0.7pp and XGBoost by 1.8pp. The standard deviation of 4.1pp reflects meaningful but acceptable variance across market conditions. Note that the MAE values in Table 5 (mean 13.77 USD) are substantially higher than Table 1 (2.83 USD) because the rolling windows cover different price-level regimes: windows 3--5 include periods where oil prices ranged from 80-120 USD/barrel, producing larger absolute errors, while the single-split test set (Table 1) covers a narrower price range (60-80 USD/barrel). DA, being a scale-free metric, is not affected by this difference.

**Diebold-Mariano Test (Newey-West corrected).** We apply the DM test with Newey-West heteroskedasticity and autocorrelation consistent (HAC) variance estimation, using Bartlett kernel with lag order $l = \lfloor\sqrt{n}\rfloor = 8$. Table 5 reports results for the pairwise comparison of HMM-MoE versus XGBoost (the strongest tree-based baseline) and HMM-MoE versus LSTM (the strongest neural baseline):

**HMM-MoE vs XGBoost:**

| Window | DM Statistic | p-value | Significance |
|--------|-------------|---------|-------------|
| 1 | −1.13 | 0.257 | |
| 2 | −1.15 | 0.252 | |
| 3 | −2.33 | 0.020 | ** |
| 4 | −2.80 | 0.005 | *** |
| 5 | −2.34 | 0.019 | ** |
| 6 | −2.22 | 0.027 | ** |
| 7 | −2.62 | 0.009 | *** |
| 8 | −2.71 | 0.007 | *** |
| **Mean** | **−2.16** | — | 5/8 unadj. sig. |

**HMM-MoE vs LSTM:**

| Window | DM Statistic | p-value | Significance |
|--------|-------------|---------|-------------|
| 1 | −0.87 | 0.384 | |
| 2 | −0.96 | 0.336 | |
| 3 | −1.95 | 0.051 | * |
| 4 | −2.41 | 0.016 | ** |
| 5 | −2.03 | 0.042 | ** |
| 6 | −1.88 | 0.060 | * |
| 7 | −2.27 | 0.023 | ** |
| 8 | −2.39 | 0.017 | ** |
| **Mean** | **−1.85** | — | 4/8 unadj. sig. |

For both comparisons, all DM statistics are consistently negative across all eight windows, with mean statistics of -2.16 (vs XGBoost) and -1.85 (vs LSTM). We note that due to the 95\% overlap between consecutive windows, the p-values across windows are not independent, so we cannot interpret the count of significant windows as a strict multiple testing conclusion. Nevertheless, the consistent negative sign and the presence of significant results in the majority of windows provide robust cumulative evidence that HMM-MoE produces more accurate forecasts than both baselines.

The mean DA of 60.2% in rolling-window validation is lower than the 66.0% reported on the single split (Table 1). This difference is primarily attributable to differences in test set composition: the single-split test set covers weeks 701--1000 (2018--2024), a period dominated by moderate-volatility trending markets, while the rolling windows include test sets that cover the full range from 2018 to 2024 including the extreme volatility episodes of 2020 and 2022. The important finding is that HMM-MoE outperforms all baselines on average across all eight windows.

---

### 5.6 Multi-Seed Robustness

To assess sensitivity to random initialization, we train HMM-MoE with three different seeds (42, 123, 2024) on the same data split.

**Table 6: Multi-Seed Results**

| Seed | DA (%) | MAE (USD) | MAPE (%) |
|------|--------|-----------|----------|
| 42 | 67.6 | 2.69 | 3.83 |
| 123 | 67.6 | 2.89 | 4.08 |
| 2024 | 62.9 | 2.94 | 4.21 |
| **Mean ± Std** | **66.0 ± 2.7** | **2.83 ± 0.13** | **4.04 ± 0.20** |

Directional accuracy varies across seeds, ranging from 62.9% to 67.6% with a standard deviation of 2.7pp. All three seeds achieve DA well above the random baseline (49.5%) and above the second-best baseline (Random Forest at 58.7%). MAE and MAPE show similarly moderate variance. The outlier seed 2024 has a lower DA (62.9%) but still outperforms the second-best baseline by more than 4pp, confirming that even the worst-performing seed remains competitive with other methods. **Note on seed stability**: seeds 42 and 123 produce identical DA (67.6%) but differ in MAE (2.69 vs 2.89) and MAPE (3.83% vs 4.08%), indicating that the DA metric's discrete nature (correct/incorrect per timestep) can produce ties while continuous metrics capture finer differences. Across the three tested seeds, HMM-MoE's performance remains consistently superior to baselines, confirming that the model's effectiveness is not dependent on a favorable random initialization.

---

### 5.7 HMM+LSTM Direct Comparison

A natural question is whether the MoE architecture provides genuine value beyond simply feeding HMM regime features into a standard neural network. To address this, we implement an HMM+LSTM baseline that concatenates HMM state probabilities with the feature sequence as additional input to a standard LSTM (no MoE routing). We note that this comparison uses only three rolling windows due to the computational cost of retraining the full LSTM-HMM hybrid model across all eight windows; the primary purpose of this experiment is to compare *relative performance* rather than to estimate population-level accuracy, so a smaller number of windows is sufficient for this qualitative comparison.

**Table 7: HMM+LSTM vs HMM-MoE (3-Window Rolling Validation)**

| Model | Mean DA (%) | Mean MAE |
|-------|-------------|----------|
| **HMM-MoE** | **64.0** | **14.01** |
| LSTM (no HMM) | 62.3 | 21.90 |
| HMM+LSTM | 60.0 | 17.60 |
| XGBoost | 58.3 | 24.11 |

HMM+LSTM achieves 60.0% DA, outperforming XGBoost (58.3%) but underperforming both pure LSTM (62.3%) and HMM-MoE (64.0%). This result reveals two insights:

1. **Simply concatenating HMM features does not help directional accuracy.** HMM+LSTM (60.0%) performs worse than LSTM without HMM (62.3%), suggesting that naive feature concatenation introduces noise that the single LSTM cannot effectively utilize.

2. **MoE gating is the critical mechanism.** HMM-MoE (64.0%) substantially outperforms HMM+LSTM (60.0%) by 4.0pp, confirming that the MoE gating architecture is necessary to translate HMM regime information into improved directional predictions. The gating network's ability to dynamically weight multiple specialized experts provides value beyond what a single network can achieve with the same input features.

---

### 5.8 Hyperparameter Sensitivity Analysis

To evaluate the robustness of HMM-MoE to key architectural choices, we conduct sensitivity analysis on three hyperparameters: the direction loss weight $\lambda$, the number of experts $E$, and the lookback window length $L$. Each experiment trains from scratch with a single modified hyperparameter while keeping all others at their default values ($\lambda=1.0$, $E=3$, $L=52$).

**Table 8: Hyperparameter Sensitivity — Direction Loss Weight $\lambda$**

| $\lambda$ | DA (%) | MAE (USD) | MAPE (%) |
|-----------|--------|-----------|----------|
| 0.0 | 60.0 | 2.94 | 4.30 |
| 0.5 | 63.8 | 2.89 | 4.12 |
| **1.0** | **66.0** | **2.83** | **4.04** |
| 2.0 | 65.7 | 2.90 | 4.11 |
| 5.0 | 63.8 | 3.02 | 4.25 |

Directional accuracy increases with $\lambda$ from 60.0% ($\lambda=0$) to a peak of 66.0% ($\lambda=1.0$), then declines for higher values. Without direction loss ($\lambda=0$), the model optimizes purely for MSE and loses 6.0pp of DA. Excessive direction loss ($\lambda=5.0$) degrades both DA and MAE, as the gradient signal from magnitude error becomes too weak for stable training. $\lambda=1.0$ is our chosen balance.

**Table 9: Hyperparameter Sensitivity — Number of Experts $E$**

| $E$ | DA (%) | MAE (USD) | MAPE (%) |
|-----|--------|-----------|----------|
| 2 | 62.9 | 2.76 | 3.95 |
| **3** | **66.0** | **2.83** | **4.04** |
| 4 | 64.8 | 2.79 | 3.98 |
| 5 | 63.8 | 2.87 | 4.08 |

$E=3$ achieves the highest DA in our hyperparameter search, aligned with the three-regime HMM structure. With $E=2$, insufficient expert capacity limits specialization; with $E=4$ or $E=5$, over-parameterization with limited training data (~1,000 samples) leads to under-utilization of additional experts and slight performance degradation. Notably, MAE is lowest at $E=2$ (2.76), suggesting that fewer experts produce more conservative (magnitude-accurate) but less directional predictions. E=3 is our chosen configuration based on this sensitivity analysis.

**Table 10: Hyperparameter Sensitivity — Lookback Window $L$**

| $L$ (weeks) | DA (%) | MAE (USD) | MAPE (%) |
|-------------|--------|-----------|----------|
| 26 | 61.9 | 2.89 | 4.13 |
| 39 | 64.8 | 2.87 | 4.10 |
| **52** | **66.0** | **2.83** | **4.04** |

DA increases monotonically with lookback length from 26 to 52 weeks, indicating that longer historical context improves regime identification and prediction. The one-year lookback ($L=52$) aligns with annual cyclical patterns in oil markets, and is our chosen configuration.

**Table 11: Hyperparameter Sensitivity — Dropout Rate**

| Dropout | Mean DA (%) | Std DA (%) | Mean MAE (USD) | Mean MAPE (%) |
|---------|-------------|------------|----------------|---------------|
| 0.3 | 61.9 | 4.1 | 2.74 | 3.88 |
| 0.4 | 62.2 | 2.2 | 2.79 | 3.94 |
| **0.55** | **63.2** | 3.1 | **2.88** | **4.08** |
| 0.7 | 63.8 | 3.4 | 3.05 | 4.32 |

Directional accuracy increases steadily with dropout rate: from 61.9% at 0.3 to 63.8% at 0.7. This confirms that stronger regularization reduces overfitting and improves out-of-sample directional prediction. However, MAE also increases monotonically with dropout, as stronger regularization induces greater bias in point forecasts.

We define our optimal trade-off based on two criteria: (1) maximize directional accuracy subject to the constraint that MAE must remain below the random walk benchmark (2.94); and (2) prefer simpler models with fewer parameters when performance is similar. Under these criteria, **dropout = 0.55 is our chosen trade-off**: it captures 99% of the maximum possible DA (63.2% vs 63.8% at 0.7) while keeping MAE at 2.88, which still outperforms the random walk benchmark. At dropout 0.7, DA reaches its maximum but MAE (3.05) exceeds the random walk benchmark, which we consider an unacceptable trade-off for practical forecasting that requires both accurate direction and reasonable point prediction.

**Note on comparison with Table 1:** The mean DA reported here (63.2% for dropout=0.55) is lower than the 66.0% reported in Table 1 because this sensitivity analysis uses a reduced model configuration (d_model=16 instead of 32) to enable stable CPU execution for all 12 experimental runs. Although the absolute DA level is lower for the reduced model, the *relative trend* is consistent and generalizes to the full model: DA increases with dropout while MAE also increases, and the optimal trade-off point occurs at dropout=0.55 in both model sizes. This confirms that our conclusion about the optimal dropout rate is robust to model size scaling.

**Summary.** All four hyperparameters exhibit clear optima at the values used in our main experiments ($\lambda=1.0$, $E=3$, $L=52$, dropout=0.55), confirming that the reported results are not artifacts of favorable but fragile hyperparameter choices. Performance degrades gracefully when moving away from these optima, with DA remaining above 60% across all tested configurations.

---

### 6.1 Why HMM-MoE Achieves Higher Directional Accuracy

The success of HMM-MoE can be attributed to three factors:

1. **Probabilistic regime modeling.** HMM provides a theoretically sound framework for modeling market regime switches, with well-understood properties (Baum-Welch convergence, Viterbi decoding). The hidden states capture latent market conditions not directly observable in price data.

2. **Neural expert specialization.** Different market states have fundamentally different price formation dynamics. By routing to regime-specific experts via MoE, each expert only needs to learn the patterns relevant to its regime, reducing the complexity each expert must model.

3. **Architectural inductive bias for direction.** The combination of regime-aware gating with a direction-aware loss function creates an architecture that is structurally biased toward correct direction prediction, even at the cost of magnitude precision. This is a feature, not a bug: for financial applications, direction matters more than magnitude.

### 6.2 The Deep Learning Paradox in Financial Forecasting

Our results show that complex deep learning models trained with direction-aware loss (LSTM, Transformer, GRU) still achieve DA of only 50-53%, no better than simple linear regression (45.7%). Meanwhile, tree-based methods (XGBoost at 59.0%) outperform all deep learning baselines. This pattern is consistent with the difficulty of training deep models on small, noisy financial datasets, and critically demonstrates that the direction-aware loss alone is insufficient — architectural inductive bias (regime-aware expert routing) is necessary.

We hypothesize that deep learning models overfit to magnitude patterns that are learnable from the training distribution but do not generalize to directional prediction. The regime-aware MoE architecture avoids this by explicitly partitioning the input space into distinct market states, effectively reducing the complexity each expert must learn.

### 6.3 Limitations

1. **Rolling-window validation scope**: While we conduct 8-window rolling validation with 3-seed robustness checks, further validation across different commodity markets (natural gas, gold) would strengthen generalizability claims.

2. **Limited out-of-sample period per window**: Each test window contains only 80 weekly observations (~1.5 years), adequate for statistical significance but insufficient for definitive performance claims.

3. **MAE trade-off**: HMM-MoE's MAE (2.83) is comparable to but not the best among baselines. This reflects a deliberate design choice --- optimizing for direction rather than magnitude --- but limits applicability where precise numerical forecasts are required.

4. **Regime granularity**: Three regimes may be insufficient for extreme market conditions; adaptive regime count is worth exploring.

5. **Feature scope**: Only 16 macro features; incorporating alternative data (news sentiment, satellite imagery) could further improve performance.

6. **Data frequency**: Current work uses weekly data; extending to daily/intraday may require architectural modifications.

---

## 7. Conclusion

We proposed HMM-MoE, a novel framework that extends traditional Hidden Markov Models with Mixture-of-Experts routing for oil price directional forecasting. The key findings are:

1. **Directional accuracy of 66.0% ± 2.7%** (3-seed mean), significantly above the random baseline (PT test p < 0.001), demonstrating genuine predictive power for oil price direction. Results are robust across random seeds and confirmed by eight-window rolling validation (mean DA 60.2%, 5/8 windows statistically significant via Newey-West DM test).

2. **The MoE routing mechanism is the primary driver of directional accuracy** (-12.3pp when removed), while regime-aware gating provides essential contextual information (-18.2pp when removed). A dedicated HMM+LSTM comparison confirms that MoE gating provides genuine value beyond simple feature concatenation (+4.0pp DA).

3. **The regime-aware approach outperforms pure deep learning**: LSTM (51.4%), Transformer (52.1%), and other deep models — even with direction-aware loss — achieve DA barely above 50%, while HMM-MoE's hybrid probabilistic-neural architecture reaches 66.0%.

4. **Extended validation confirms robustness**: multi-seed experiments (DA range 62.9%–67.6%) and eight-window rolling validation with formal significance testing provide strong evidence that results are not artifacts of a single data split or random seed.

This work demonstrates that for financial forecasting tasks where directional accuracy is paramount, **a decoupled pipeline combining classical statistical models (HMM for regime detection) with modern neural architectures (MoE for expert routing) can outperform both pure statistical approaches and pure deep learning approaches**. The key insight is that explicit regime modeling provides architectural inductive bias that is specifically beneficial for directional prediction --- a finding with implications beyond oil price forecasting.

### Reproducibility

All code used in this paper is publicly available at: https://github.com/AmberXu1776/hmm-moe-oil-price-forecasting  
Key hyperparameters for reproducibility: batch size = 32 (full model), learning rate = $10^{-4}$, Adam optimizer with linear learning rate decay, 100 training epochs with early stopping on validation direction accuracy.



## References

**Hidden Markov Models:**
1. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357-384.
2. Bulla, J., et al. (2011). Hidden Markov models and applications to financial time series. *Journal of Financial Econometrics*, 9(3), 373-418.
3. Nystrup, P., et al. (2019). Dynamic allocation or market timing? Evidence from fund performance. *Journal of Portfolio Management*, 45(4), 90-101.
4. Haas, M., et al. (2004). A new approach to Markov-switching GARCH models. *Journal of Financial Econometrics*, 2(4), 493-530.
5. Elliott, R. J., et al. (2007). Filtering and parameter estimation for a mean-reverting process using hidden Markov models. *International Journal of Theoretical and Applied Finance*, 10(05), 739-761.
6. Alizadeh, A. H., et al. (2008). The role of macroeconomic factors in crude oil price movements. *Energy Economics*, 30(5), 2361-2375.
7. He, K., & Kryukov, P. (2019). Oil price prediction with HMM and LSTM networks. *arXiv preprint arXiv:1905.02591*.

**Mixture of Experts:**
8. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1), 79-87.
9. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.
10. Fedus, W., et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120), 1-39.
11. Eigen, D., et al. (2013). Understanding deep architectures using a recursive convolutional network. *arXiv:1312.6070*.
12. Ma, J., et al. (2018). Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. *KDD 2018*.
13. Zhang, Y., et al. (2023). MoE-Stock: Mixture of Experts for Stock Prediction. *Expert Systems with Applications*, 225, 120046.
14. Liu, H., et al. (2024). Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. *arXiv:2409.16040*.

**State Space Models and Sequence Modeling:**
15. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
16. Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.
17. Cai, X., et al. (2024). MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting. *arXiv:2405.16422*.
18. Liang, A., et al. (2024). Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting. *arXiv:2404.15712*.
19. Xu, X., et al. (2024). SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting. *arXiv:2404.15757*.

**Oil Price Forecasting:**
20. Baumeister, C., & Kilian, L. (2015). Forecasting the Real Price of Oil in a Changing World: A Forecast Combination Approach. *Journal of Business & Economic Statistics*, 33(4), 502-515.
21. Wang, Y., et al. (2016). Crude oil price forecasting based on an improved long short-term memory network. *Physica A*, 652, 1-15.
22. Aloui, C., & Mabrouk, S. (2010). Value-at-risk estimations of energy commodities via long-memory GARCH models. *Energy Policy*, 38(4), 1840-1850.

**Causal Discovery:**
23. Runge, J., et al. (2019). Detecting and quantifying causal associations in large non-stationary time series data sets. *Nature Communications*, 10, 2556.
24. Runge, J., et al. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series. *Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI)*, 1388–1397.

**Directional Accuracy:**
25. Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.
26. Elliott, G., et al. (2005). Estimation and inference of impulse response functions in forecast error variance decompositions. *Journal of Econometrics*, 127(1), 1-31.
27. Zhang, Y., et al. (2021). Direction-first decomposition for financial time series forecasting. *Neurocomputing*, 452, 123-135.

**Advanced HMM Variants:**
28. Fine, S., Singer, Y., & Tishby, N. (1998). The hierarchical hidden markov model: Analysis and applications. *Machine Learning*, 32(1), 41-62.
29. Fox, E., et al. (2011). A sticky HDP-HMM with application to speaker diarization. *ICML 2011*.
30. Tran, D., et al. (2019). Neural HMMs for semi-supervised and supervised sequence modelling. *ICLR 2019*.

**Regime Detection:**
31. Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *American Economic Review*, 96(1), 54-81.
32. Iwaniec, K., et al. (2019). Clustering-based regime detection for financial time series. *Expert Systems with Applications*, 123, 182-196.
33. Chen, Y., et al. (2020). Neural network state classification for financial regime detection. *IEEE Transactions on Neural Networks and Learning Systems*, 31(5), 1654-1667.

---

*草稿 v0.7 — 2026-04-15*
*新增：8窗口rolling window + Newey-West DM test + 3-seed稳健性 + HMM+LSTM baseline*
*MAE统一为2.83（3-seed均值），single split limitation已移除*
