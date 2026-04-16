# Hierarchical HMM-MoE: 理论框架

**Date:** 2026-04-10
**Status:** v0.1 — 理论推导

---

## 1. 核心思想

**传统做法：** HMM 和 MoE 是串联的"特征提取 → 预测"管道。
**我们的做法：** HMM 的隐状态是生成模型的一阶组件，MoE 专家是二阶组件，两者构成**层次混合模型（Hierarchical Mixture Model）**。

关键区别：HMM 不再是"预处理模块"，而是预测分布的**结构性组成部分**。

---

## 2. 模型定义

### 2.1 观测生成过程

油价收益率的预测分布：

$$P(y_t | \mathbf{x}_{1:t}, \theta) = \sum_{j=1}^{N} \gamma_t^{(j)} \cdot P(y_t | \mathbf{x}_t, S_t=j, \theta)$$

其中 $\gamma_t^{(j)} = P(S_t=j | \mathbf{x}_{1:t})$ 是 HMM 的 filtering probability。

给定 regime j，预测分布是 expert mixture：

$$P(y_t | \mathbf{x}_t, S_t=j, \theta) = \sum_{e=1}^{E} g_{e|j} \cdot \mathcal{N}(y_t; \mu_{e,j}(\mathbf{x}_t), \sigma_{e,j}^2)$$

**展开完整形式：**

$$\boxed{P(y_t | \mathbf{x}_{1:t}, \theta) = \sum_{j=1}^{N} \sum_{e=1}^{E} \underbrace{\gamma_t^{(j)}}_{\text{regime weight}} \cdot \underbrace{g_{e|j}}_{\text{expert weight}} \cdot \underbrace{\mathcal{N}(y_t; f_e^{(j)}(\mathbf{x}_t), \sigma_{e,j}^2)}_{\text{expert output}}}$$

**层次解读：**
- **第一层（外层）：** N 个 regime mixture components，权重由 HMM filtering probability γ_t 决定
- **第二层（内层）：** 每个 regime 下有 E 个 expert mixture components，权重 g_{e|j} 依赖 regime

### 2.2 与标准 MoE 的对比

| 方面 | 标准 MoE | 我们的方法 |
|------|---------|-----------|
| 混合权重 | 单层 softmax | 层次：regime × expert |
| 权重来源 | 门控网络学 | 外层由 HMM 动态计算，内层由 regime-conditioned gating |
| 时序依赖 | 无（或靠 LSTM 隐状态） | 有（HMM filtering 是序列推断） |
| 不确定性 | 单一方差 | regime-expert 特定方差 σ²_{e,j} |
| 理论性质 | 非概率路由 | 完整的概率生成模型 |

### 2.3 Regime-Conditioned Gating

内层门控权重依赖 regime：

$$g_{e|j} = \text{softmax}_e(\mathbf{w}_j \cdot h_t + b_j)$$

其中 $\mathbf{w}_j \in \mathbb{R}^{d_h \times E}$ 是 regime-specific 的门控参数，$h_t$ 是特征编码。

**关键：** 不同 regime 下，同一个 expert 的权重不同。这意味着模型学到的不仅是"什么时候用哪个 expert"，还有"在不同市场状态下，expert 之间如何分工"。

### 2.4 Regime-Conditioned Expert Output

Expert 的输出也受 regime 影响：

$$\mu_{e,j}(\mathbf{x}_t) = f_e^{(j)}(\mathbf{x}_t)$$

两种实现方式：

**方式A：共享骨干 + regime-specific head（轻量）**

$$f_e^{(j)}(\mathbf{x}_t) = \mathbf{w}_{e,out}^{(j)\top} \cdot \text{LSTM}_e(\mathbf{x}_t) + b_{e,out}^{(j)}$$

LSTM 骨干共享，但输出层是 regime-specific。

**方式B：完全独立 expert（重量）**

每个 (regime, expert) 对有独立的 LSTM。参数量大，但灵活度最高。

**推荐方式A** — 参数效率高，且 regime 信息已经通过 γ_t 传入，不需要完全独立。

### 2.5 不确定性建模

每个 (regime, expert) 对有自己的方差参数：

$$\sigma_{e,j}^2 = \text{softplus}(\tilde{\sigma}_{e,j})$$

**物理意义：**
- 同一个 expert 在不同 regime 下的预测不确定性不同
- 例如：趋势跟踪 expert 在趋势 regime 下 σ² 小（自信），在震荡 regime 下 σ² 大（不确定）

**这对金融应用很重要：** 预测的不确定性本身是有价值的信息，可以用于风险管理。

---

## 3. 对数似然与损失函数

### 3.1 对数似然

$$\log P(y_t | \mathbf{x}_{1:t}, \theta) = \log \sum_{j=1}^{N} \sum_{e=1}^{E} \gamma_t^{(j)} \cdot g_{e|j} \cdot \mathcal{N}(y_t; f_e^{(j)}(\mathbf{x}_t), \sigma_{e,j}^2)$$

用 logsumexp 保证数值稳定：

$$= \text{logsumexp}_{j,e}\left[\log \gamma_t^{(j)} + \log g_{e|j} + \log \mathcal{N}(y_t; \mu_{e,j}, \sigma_{e,j}^2)\right]$$

### 3.2 完整训练目标

$$\mathcal{L} = \underbrace{-\frac{1}{T}\sum_{t=1}^{T} \log P(y_t | \mathbf{x}_{1:t}, \theta)}_{\mathcal{L}_{NLL} \text{ — 负对数似然}} + \underbrace{\alpha \cdot \mathcal{L}_{HMM}}_{\text{HMM似然正则}} + \underbrace{\beta \cdot \mathcal{L}_{balance}}_{\text{负载均衡}}$$

**与之前的区别：**
- 之前用 Huber + Direction Loss（point estimate 的损失）
- 现在用 NLL（概率模型的损失），自带方向感知和不确定性估计
- 不需要手动设计方向损失——如果预测分布的均值方向对了且方差合适，NLL 自然就低

### 3.3 NLL 自动包含方向感知

展开 NLL 的梯度：

$$\frac{\partial \mathcal{L}_{NLL}}{\partial \mu_{e,j}} = -\frac{w_{e,j,t}}{\sigma_{e,j}^2}(y_t - \mu_{e,j})$$

其中 $w_{e,j,t} = \gamma_t^{(j)} \cdot g_{e|j}$ 是后验 responsibility。

当 $y_t > 0$ 但 $\mu_{e,j} < 0$ 时，梯度推 $\mu_{e,j}$ 往正方向走。这比手动设计的 direction loss 更自然——方向正确性被编码在概率模型的结构里，不是加个额外的 loss term。

---

## 4. 与 HMM 的梯度耦合

### 4.1 梯度路径

$$\frac{\partial \mathcal{L}_{NLL}}{\partial \theta_{HMM}} = \sum_t \frac{\partial \mathcal{L}_{NLL}}{\partial \log P(y_t|\cdot)} \cdot \frac{\partial \log P(y_t|\cdot)}{\partial \gamma_t} \cdot \frac{\partial \gamma_t}{\partial \theta_{HMM}}$$

关键中间步骤：

$$\frac{\partial \log P(y_t|\cdot)}{\partial \gamma_t^{(j)}} = \frac{\sum_e g_{e|j} \cdot \mathcal{N}(y_t; \mu_{e,j}, \sigma_{e,j}^2)}{P(y_t | \cdot)} = \frac{P(y_t, S_t=j | \cdot)}{P(y_t | \cdot)} = P(S_t=j | y_t, \cdot)$$

**这正好是后验 regime probability。** 物理意义：如果 regime j 下的 expert 预测越接近真实值，梯度越大地鼓励 HMM 增大 γ_t^{(j)}。

### 4.2 这个梯度的直觉

HMM 学到的不是"统计上最优的 regime 划分"，而是"让预测分布最好地解释数据的 regime 划分"。但 α·L_HMM 保证这个划分不能太偏离统计合理性。

**这是一个 soft version 的 EM 算法：**
- E-step：γ_t 由 HMM 前向算法计算
- M-step：梯度更新同时调整 HMM 参数和 expert 参数
- 两步不像传统 EM 那样交替，而是同时发生

---

## 5. 论文叙事（Draft）

### Title Ideas
- "Hierarchical Regime-Expert Mixture for Oil Price Forecasting"
- "A Hierarchical Hidden Markov Mixture-of-Experts Model for Financial Time Series"
- "End-to-End Hierarchical HMM-MoE: Probabilistic Regime-Aware Forecasting"

### 核心贡献（3点）

1. **理论贡献：** 提出 Hierarchical HMM-MoE，一个将隐马尔可夫模型的 regime 动态与混合专家的函数灵活性统一在一个概率生成模型中的层次混合框架。HMM 隐状态不再是预处理步骤，而是预测分布的结构性组件。

2. **方法论贡献：** 推导端到端可微的训练算法，使预测似然的梯度能回传到 HMM 参数，实现 regime discovery 和 expert specialization 的联合优化。

3. **应用贡献：** 在原油价格预测任务上验证，展示层次混合模型相比 flat MoE 和断开 HMM-MoE 的优势，并提供 regime-specific 的不确定性估计。

### 与竞品的方法论差异

| 方法 | HMM 角色 | MoE 角色 | 耦合方式 | 不确定性 |
|------|---------|---------|---------|---------|
| 断开 HMM-MoE | 特征提取器 | 预测器 | 无 | 无 |
| Neural HMM | 发射模型 | 无 | 全耦合 | 无 |
| 标准 MoE | 无 | 预测器 | N/A | 无 |
| **Ours** | **概率组件** | **概率组件** | **层次生成模型** | **有** |

---

## 6. 实验计划

### 6.1 Baseline
1. 纯 LSTM
2. 标准 MoE（flat gating）
3. 断开 HMM-MoE（hmmlearn → MoE）
4. 端到端 HMM-MoE（v0.4，串联架构）
5. **层次 HMM-MoE（本方案）**

### 6.2 Metrics
- DA (Direction Accuracy)
- RMSE / MAE
- NLL（负对数似然）— 概率模型的核心指标
- Calibration（预测区间的覆盖率）— 不确定性估计的质量
- CRPS（Continuous Ranked Probability Score）— 概率预测的综合指标

### 6.3 Ablation
- 去掉 regime-conditioned gating（g_{e|j} → g_e）
- 去掉 regime-conditioned σ²（σ²_{e,j} → σ²_e）
- 去掉 HMM 梯度回传（freeze HMM params）
- 改变 N（regime 数量：2, 3, 4）
- 改变 E（expert 数量：2, 3, 4）

---

*最后更新：2026-04-10*
