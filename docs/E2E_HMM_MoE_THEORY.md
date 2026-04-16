# End-to-End Differentiable HMM-MoE: Complete Theoretical Derivation

**Date:** April 6, 2026

**Authors:** Amber Lord, Amiya

---

## Abstract

本完整版文档从第一性原理出发，推导端到端可微 HMM-MoE 架构的每一个环节：
- **传统断开架构的问题**：为什么 hmmlearn → numpy → PyTorch 的管道会断裂梯度
- **可微前向算法**：在 log-space 中用 PyTorch 原生操作实现完整的 HMM 前向算法
- **梯度链分析**：从预测损失到 HMM 参数（π, A, μ, Σ）的完整反向传播路径
- **参数约束映射**：softmax/exp 保证概率正定约束
- **联合损失函数**：L_pred + λ_dir · L_dir + α · L_hmm + β · L_balance
- **两阶段训练策略**：EM 初始化 → HMM warm-up → 联合微调

---

# Part I: The Problem — Why the Gradient Breaks

## 1.1 断开架构的管道

在传统实现中，HMM 和 MoE 是两个独立模块：

```
hmmlearn.GaussianHMM.fit(X)  →  numpy array γ_t  →  torch.tensor  →  Gate  →  Experts
         ↑                                ↑                              ↑
    EM算法(Baum-Welch)              没有梯度                        MoE梯度
    优化观测似然                   .detach()等价                   传到这里就死了
```

**代码层面：**

```python
# Step 1: HMM 用 hmmlearn 训练（EM算法）
hmm = GaussianHMM(n_components=3).fit(X_train)
# → 参数 π, A, μ, Σ 在这里固定

# Step 2: 计算 regime probability（numpy运算，无梯度）
gamma = hmm.predict_proba(X_all)  # numpy array, shape [T, 3]
# → 梯度在这里死亡

# Step 3: 转为 tensor（等价于 .detach()）
gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
# → PyTorch 计算图无法回传

# Step 4: 喂入 MoE 门控
gate_input = gamma_tensor  # 或 concat 其他特征
gate_weights = softmax(MLP(gate_input))
y_hat = sum(gate_weights * expert_outputs)
# → 梯度只能传回 gate_weights，无法传到 HMM 参数
```

## 1.2 后果：两个不协调的优化目标

| 模块 | 优化目标 | 梯度来源 |
|------|---------|---------|
| HMM (hmmlearn) | 最大化观测似然 $P(\mathbf{X}\|\theta)$ | EM 算法（Baum-Welch） |
| MoE (PyTorch) | 最小化预测损失 $\mathcal{L}_{pred}$ | 反向传播 |

这两个目标之间没有协调机制。HMM 学出的 regime 划分可能"对分类有用，但对预测无用"。

**具体例子：** HMM 可能将数据划分为"价格均值 < 50"和"价格均值 ≥ 50"两组，这种划分在统计意义上使观测似然最大，但对预测油价涨跌方向可能毫无帮助——因为 regime 的边界不一定是预测任务最有用的切分方式。

## 1.3 端到端的核心思想

**目标：** 让预测损失的梯度能回传到 HMM 参数，使 HMM 学出"对预测有用"的 regime 划分。

```
L_pred → ŷ → gate_weights → γ_t → α_t → θ_HMM (π, A, μ, Σ)
```

这需要 HMM 的前向算法完全用可微操作实现。

---

# Part II: Differentiable Forward Algorithm

## 2.1 HMM 的参数化

一个 $N$-状态的 HMM 由以下参数定义：

**初始分布：** $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_N)^\top \in \Delta^{N-1}$

**转移矩阵：** $\mathbf{A} \in \mathbb{R}^{N \times N}$，其中 $A_{ij} = P(S_t = j | S_{t-1} = i)$

**发射分布：** $P(\mathbf{x}_t | S_t = j) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$

约束条件：
- $\sum_j \pi_j = 1, \quad \pi_j \geq 0$
- $\sum_j A_{ij} = 1, \quad A_{ij} \geq 0$ （对每个 $i$）
- $\boldsymbol{\Sigma}_j$ 正定

## 2.2 无约束参数化

为了在梯度下降中自由优化，我们用**无约束参数**加**映射函数**：

### 2.2.1 初始分布

$$\boldsymbol{\pi} = \text{softmax}(\tilde{\boldsymbol{\pi}})$$

其中 $\tilde{\boldsymbol{\pi}} \in \mathbb{R}^N$ 是无约束的 `nn.Parameter`。

**验证：** 对任意 $\tilde{\boldsymbol{\pi}}$，softmax 保证 $\pi_j \geq 0$ 且 $\sum_j \pi_j = 1$。

**展开：**

$$\pi_j = \frac{\exp(\tilde{\pi}_j)}{\sum_{k=1}^{N} \exp(\tilde{\pi}_k)}$$

### 2.2.2 转移矩阵

$$A_{i,:} = \text{softmax}(\tilde{A}_{i,:})$$

其中 $\tilde{\mathbf{A}} \in \mathbb{R}^{N \times N}$ 是无约束的 `nn.Parameter`。

**逐行 softmax：** 每行独立归一化，保证每行是一个概率分布。

$$A_{ij} = \frac{\exp(\tilde{A}_{ij})}{\sum_{k=1}^{N} \exp(\tilde{A}_{ik})}$$

### 2.2.3 发射分布（对角协方差）

$$\boldsymbol{\mu}_j \in \mathbb{R}^D \quad \text{（无约束，直接作为参数）}$$

$$\boldsymbol{\Sigma}_j = \text{diag}(\sigma_{j1}^2, \ldots, \sigma_{jD}^2)$$

其中 $\sigma_{jd} = \exp(\tilde{\sigma}_{jd})$，$\tilde{\sigma}_{jd}$ 是无约束参数。

**为什么用对角协方差？**
1. 参数量：完整协方差 $O(N \cdot D^2)$ vs 对角协方差 $O(N \cdot D)$
2. 数值稳定性：完整协方差容易病态
3. 在我们的场景中（$D=6$），对角假设足够

**exp 保证正定性：** $\sigma_{jd} = \exp(\tilde{\sigma}_{jd}) > 0$，因此 $\Sigma_j$ 正定。

## 2.3 发射概率的计算

### 2.3.1 高斯发射对数概率

$$\log P(\mathbf{x}_t | S_t = j) = -\frac{D}{2}\log(2\pi) - \sum_{d=1}^{D} \log \sigma_{jd} - \frac{1}{2} \sum_{d=1}^{D} \left(\frac{x_{td} - \mu_{jd}}{\sigma_{jd}}\right)^2$$

**PyTorch 实现：**

```python
def _emission_log_prob(self, x):
    # x: [T, D]
    sigma = torch.exp(self.log_sigma).clamp(min=1e-4)  # [N, D]
    diff = x.unsqueeze(1) - self.mu.unsqueeze(0)        # [T, N, D]
    log_prob = -0.5 * (
        D * log_2pi
        + 2 * self.log_sigma.unsqueeze(0)                 # log(σ²)
        + (diff / sigma.unsqueeze(0)) ** 2                # (x-μ)²/σ²
    ).sum(dim=-1)                                         # [T, N]
    return log_prob
```

**每一步都在 PyTorch 计算图内：**
- `self.log_sigma` → `nn.Parameter`，有梯度
- `self.mu` → `nn.Parameter`，有梯度
- 所有运算（减、除、幂、求和）→ 可微

## 2.4 前向算法（Log-Space 数值稳定版）

### 2.4.1 为什么必须用 log-space？

**朴素实现的问题：** 前向概率 $\alpha_t(j)$ 是 $t$ 个概率的连乘积：

$$\alpha_t(j) = \sum_{i} \alpha_{t-1}(i) \cdot A_{ij} \cdot b_j(\mathbf{x}_t)$$

每个因子 $\leq 1$，连乘 $t$ 次后：

$$\alpha_t \sim O(b_{\max}^t)$$

当 $t > 100$ 时，浮点数下溢到 0。对于我们的周频数据（$T = 52$），已经足够严重。

**Log-space 方案：** 用 $\log \alpha_t$ 代替 $\alpha_t$，避免下溢。

### 2.4.2 初始化

$$\log \alpha_1(j) = \log \pi_j + \log b_j(\mathbf{x}_1)$$

**其中：**
- $\log \pi_j$：通过 $\log(\text{softmax}(\tilde{\pi})_j + \epsilon)$ 计算
- $\log b_j(\mathbf{x}_1)$：发射对数概率

### 2.4.3 递推

$$\log \alpha_t(j) = \log b_j(\mathbf{x}_t) + \log \sum_{i=1}^{N} \alpha_{t-1}(i) \cdot A_{ij}$$

**关键技巧：logsumexp**

$$\log \sum_{i=1}^{N} \alpha_{t-1}(i) \cdot A_{ij} = \log \sum_{i=1}^{N} \exp(\log \alpha_{t-1}(i) + \log A_{ij})$$

$$= \text{logsumexp}_i(\log \alpha_{t-1}(i) + \log A_{ij})$$

**PyTorch 实现：**

```python
for t in range(1, T):
    # log_alpha: [N] → [N, 1]
    # log_A: [N, N]
    log_alpha = log_b[t] + torch.logsumexp(
        log_alpha.unsqueeze(1) + log_A,  # [N, N]
        dim=0                             # sum over source state → [N]
    )
```

**这里每一步都可微：**
- `log_alpha.unsqueeze(1) + log_A` → 加法，可微
- `torch.logsumexp` → PyTorch 内建可微操作
- `log_b[t] +` → 加法，可微

### 2.4.4 Filtering Probability γ_t

$$\gamma_t(j) = \frac{\alpha_t(j)}{\sum_{k=1}^{N} \alpha_t(k)} = \frac{\exp(\log \alpha_t(j))}{\sum_{k=1}^{N} \exp(\log \alpha_t(k))} = \text{softmax}(\log \alpha_t)_j$$

**关键：** 这个 γ_t 在计算图内！

```python
gamma = F.softmax(log_alphas, dim=1)  # [T, N], requires_grad=True
```

**这就是端到端的核心：** γ_t 依赖于 HMM 参数（通过 log_alpha 的递推），而后续的门控网络直接消费 γ_t。梯度可以从预测损失一路传回到 π、A、μ、Σ。

### 2.4.5 HMM 对数似然

$$\log P(\mathbf{X} | \theta_{HMM}) = \log \sum_{j=1}^{N} \alpha_T(j) = \text{logsumexp}_j(\log \alpha_T(j))$$

这个量用于联合损失中的 HMM 正则项。

## 2.5 完整的梯度链

从预测损失到 HMM 参数的梯度路径：

$$\frac{\partial \mathcal{L}_{pred}}{\partial \theta_{HMM}} = \frac{\partial \mathcal{L}_{pred}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \gamma_T} \cdot \frac{\partial \gamma_T}{\partial \log\alpha_T} \cdot \frac{\partial \log\alpha_T}{\partial \theta_{HMM}}$$

**逐步展开：**

| 链接 | 操作 | 梯度 |
|------|------|------|
| $\hat{y} \leftarrow \mathbf{g}$ | $\hat{y} = \sum_e g_e \cdot f_e(\mathbf{x})$ | $\frac{\partial \hat{y}}{\partial g_e} = f_e(\mathbf{x})$ |
| $\mathbf{g} \leftarrow \gamma_T$ | $g_e = \text{softmax}_e(\text{MLP}(\gamma_T))$ | 通过 softmax Jacobian 和 MLP 反传 |
| $\gamma_T \leftarrow \log\alpha_T$ | $\gamma_T = \text{softmax}(\log\alpha_T)$ | Softmax Jacobian: $\frac{\partial \gamma_i}{\partial \log\alpha_j} = \gamma_i(\delta_{ij} - \gamma_j)$ |
| $\log\alpha_T \leftarrow \theta_{HMM}$ | 前向递推（$T$ 步 logsumexp） | 通过 $T$ 步递推展开 |

**最后一步的详细分析：**

$\log\alpha_t(j)$ 依赖于 $\log b_j(\mathbf{x}_t)$（发射概率），而发射概率依赖于 $\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j$：

$$\frac{\partial \log b_j(\mathbf{x}_t)}{\partial \mu_{jd}} = \frac{x_{td} - \mu_{jd}}{\sigma_{jd}^2}$$

$$\frac{\partial \log b_j(\mathbf{x}_t)}{\partial \tilde{\sigma}_{jd}} = \frac{\partial \log b_j}{\partial \sigma_{jd}} \cdot \frac{\partial \sigma_{jd}}{\partial \tilde{\sigma}_{jd}} = \left(\frac{(x_{td} - \mu_{jd})^2}{\sigma_{jd}^3} - \frac{1}{\sigma_{jd}}\right) \cdot \sigma_{jd}$$

$\log\alpha_t(j)$ 还依赖于 $\log A_{ij}$（转移概率）：

$$\log\alpha_t(j) = \log b_j(\mathbf{x}_t) + \text{logsumexp}_i(\log\alpha_{t-1}(i) + \log A_{ij})$$

梯度通过 $T$ 步递推的链式法则回传。PyTorch autograd 自动处理这个展开。

---

# Part III: End-to-End Architecture

## 3.1 整体架构

```
Input: x_seq [B, T_seq, D_seq]  +  x_hmm [B, T_hmm, D_hmm]
                |                              |
                |                    ┌─────────┴──────────┐
                |                    │  DifferentiableHMM  │
                |                    │  Forward Algorithm  │
                |                    │  (log-space)        │
                |                    └─────────┬──────────┘
                |                              |
                |                    γ_t [B, N] ← 最后时刻的 filtering probability
                |                              |
                |                    ┌─────────┴──────────┐
                |                    │   Gating Network    │
                |                    │   MLP(γ_t) → softmax│
                |                    └─────────┬──────────┘
                |                              |
                |                    gate_weights g [B, E]
                |                              |
       ┌───────┴──────────────────────────────┴───────┐
       |              |                               |
  ┌────┴────┐   ┌────┴────┐                     ┌────┴────┐
  │ Expert 1 │   │ Expert 2 │     ...            │ Expert E │
  │ (LSTM)   │   │ (LSTM)   │                    │ (LSTM)   │
  └────┬─────┘   └────┬─────┘                    └────┬─────┘
       |              |                               |
       y₁ [B,1]      y₂ [B,1]                   y_E [B,1]
       |              |                               |
       └──────────────┴───────────────────────────────┘
                              |
                    ŷ = Σ g_e · y_e  [B, 1]
```

## 3.2 维度分析

以我们的实际配置为例：

| 符号 | 含义 | 维度 |
|------|------|------|
| $B$ | Batch size | 16 |
| $T_{seq}$ | 时序窗口长度 | 104 (2年) |
| $D_{seq}$ | 时序特征维度 | 51 (16原始 + 35工程) |
| $T_{hmm}$ | HMM 窗口长度 | 52 (1年) |
| $D_{hmm}$ | HMM 特征维度 | 6 (logret + vol + mom + 3 oil returns) |
| $N$ | HMM 状态数 | 3 |
| $E$ | 专家数 | 3 |

## 3.3 HMM 特征选择

HMM 的输入特征不是全部 51 维，而是精选的 6 维：

$$\mathbf{x}_{hmm,t} = [\text{oil\_logret}_t, \text{oil\_vol}_t, \text{oil\_mom}_t, \text{OPEC\_lr}_t, \text{Brent\_lr}_t, \text{WTI\_lr}_t]$$

**为什么只用 6 维？**

1. **HMM 的维度诅咒：** 高维高斯的参数量 $O(N \cdot D^2)$，$D=51$ 时参数太多，在小样本上过拟合
2. **经济直觉：** regime 变化主要由油价自身动态决定，宏观指标是次要信号
3. **数值稳定性：** 低维高斯的协方差矩阵更容易保持正定

## 3.4 Expert 网络结构

每个专家是一个 LSTM 序列模型：

```python
class MambaExpert(nn.Module):
    def __init__(self, input_dim, d_model=32, d_state=16, n_layers=1, dropout=0.55):
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x):  # x: [B, T_seq, D_seq]
        x = self.input_proj(x)           # [B, T_seq, d_model]
        x, _ = self.lstm(x)              # [B, T_seq, d_model]
        x = self.norm(x[:, -1, :])       # [B, d_model] — 取最后时刻
        return self.output_proj(x)        # [B, 1]
```

**关键设计：** 所有专家共享输入 $\mathbf{x}_{seq}$，但有不同的参数。门控网络决定每个专家的权重。

## 3.5 Gating Network

$$\mathbf{g} = \text{softmax}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \boldsymbol{\gamma}_T + \mathbf{b}_1) + \mathbf{b}_2)$$

**输入：** $\boldsymbol{\gamma}_T \in \mathbb{R}^N$（最后时刻的 filtering probability）

**与断开架构的区别：**
- 断开：γ_t 是 numpy array，不参与梯度计算
- 端到端：γ_t 在计算图内，梯度可以回传到 HMM 参数

**简洁设计：** 只用 γ_t（3维）作为门控输入，不用额外的分位数特征。原因是端到端训练中，HMM 会自动学到对预测有用的 regime，不需要分位数辅助。

---

# Part IV: Joint Loss Function

## 4.1 损失函数组成

$$\mathcal{L}_{joint} = \mathcal{L}_{pred} + \lambda_{dir} \cdot \mathcal{L}_{dir} + \alpha \cdot \mathcal{L}_{HMM} + \beta \cdot \mathcal{L}_{balance}$$

### 4.1.1 预测损失（Huber Loss）

$$\mathcal{L}_{pred} = \frac{1}{B} \sum_{i=1}^{B} \text{Huber}_\delta(\hat{y}_i - y_i)$$

其中：

$$\text{Huber}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & \text{if } |r| \leq \delta \\ \delta(|r| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$$

**为什么用 Huber 而非 MSE？** Huber 对异常值更鲁棒（线性惩罚 > 二次惩罚），适合金融数据中的厚尾分布。

### 4.1.2 方向感知损失（Direction-Aware Loss）

$$\mathcal{L}_{dir} = 1 - \frac{1}{B}\sum_{i=1}^{B} \sigma(\tau \cdot \hat{y}_i \cdot y_i)$$

**其中：**
- $\sigma(\cdot)$：sigmoid 函数
- $\tau = 10.0$：温度参数

**直觉：**
- 当 $\hat{y}_i$ 和 $y_i$ 同号时，$\hat{y}_i \cdot y_i > 0$，$\sigma(\tau \cdot \hat{y}_i y_i) \approx 1$，损失 $\approx 0$
- 当 $\hat{y}_i$ 和 $y_i$ 异号时，$\hat{y}_i \cdot y_i < 0$，$\sigma(\tau \cdot \hat{y}_i y_i) \approx 0$，损失 $\approx 1$

**温度参数 $\tau$ 的作用：** 控制 sigmoid 的"软硬"程度。$\tau$ 越大，越接近阶跃函数（硬方向判断）。

### 4.1.3 HMM 似然正则（归一化）

$$\mathcal{L}_{HMM} = -\frac{1}{T_{hmm}} \log P(\mathbf{X}_{hmm} | \theta_{HMM})$$

**归一化：** 除以 $T_{hmm}$（当前=52），使 HMM 损失的量级与预测损失匹配。

**未归一化时的问题：** $\log P(\mathbf{X}) \approx -1775$（$T_{hmm}=52$ 个时间步的概率连乘），而 $\mathcal{L}_{pred} \approx 0.35$。即使 $\alpha = 0.1$，HMM 项贡献 $0.1 \times 1775 = 177.5$，完全主导梯度方向。

**归一化后：** $\mathcal{L}_{HMM} \approx -1775/52 \approx -34.1$，$\alpha \cdot \mathcal{L}_{HMM} \approx 3.41$，与预测损失同量级。

**物理意义：** HMM 似然正则鼓励 HMM 学出"合理的" regime 划分（观测似然不能太低），同时不阻止 regime 对预测任务的适应性调整。

### 4.1.4 负载均衡损失

$$\mathcal{L}_{balance} = D_{KL}(\bar{\mathbf{g}} \| \mathbf{u})$$

其中 $\bar{\mathbf{g}} = \frac{1}{B}\sum_{i=1}^{B} \mathbf{g}_i$ 是 batch 平均门控权重，$\mathbf{u} = (1/E, \ldots, 1/E)$ 是均匀分布。

**展开：**

$$\mathcal{L}_{balance} = \sum_{e=1}^{E} \bar{g}_e \log \frac{\bar{g}_e}{1/E} = \sum_{e=1}^{E} \bar{g}_e (\log \bar{g}_e + \log E)$$

**作用：** 防止门控坍缩到只使用一个专家。

---

# Part V: Training Strategy

## 5.1 两阶段训练

### 5.1.1 为什么需要两阶段？

直接从随机初始化做联合训练很难收敛，原因：

1. **HMM 参数敏感：** 随机的 π、A 会导致 γ_t 接近均匀分布，门控网络收不到有效信号
2. **梯度冲突：** $\mathcal{L}_{pred}$ 想让 HMM 学"对预测有用的 regime"，$\mathcal{L}_{HMM}$ 想让 HMM 学"统计上合理的 regime"，两者梯度方向可能矛盾
3. **学习率不匹配：** HMM 参数（48个）远少于 MoE 参数（~30,000个），但梯度量级可能差很多

### 5.1.2 Stage 1: HMM Warm-up

**目标：** 用 EM 结果初始化，然后梯度微调 HMM 参数。

**损失：** 只用 HMM 似然

$$\mathcal{L}_{S1} = -\log P(\mathbf{X}_{hmm} | \theta_{HMM})$$

**冻结：** MoE 参数不变，只更新 HMM 参数。

**效果：** HMM 从 EM 的好起点出发，在当前 batch 上快速适配。

### 5.1.3 Stage 2: Joint Training

**目标：** 联合优化 HMM + MoE。

**损失：** 完整联合损失

$$\mathcal{L}_{S2} = \mathcal{L}_{pred} + \lambda_{dir} \cdot \mathcal{L}_{dir} + \alpha \cdot \mathcal{L}_{HMM} + \beta \cdot \mathcal{L}_{balance}$$

**差分学习率：**

$$\theta_{HMM} \leftarrow \theta_{HMM} - \eta_{HMM} \cdot \nabla_{\theta_{HMM}} \mathcal{L}$$
$$\theta_{MoE} \leftarrow \theta_{MoE} - \eta_{MoE} \cdot \nabla_{\theta_{MoE}} \mathcal{L}$$

其中 $\eta_{HMM} < \eta_{MoE}$（HMM 走慢一点，避免破坏 EM 初始化的好起点）。

**当前配置：** $\eta_{HMM} = 5 \times 10^{-5}$, $\eta_{MoE} = 3 \times 10^{-4}$

## 5.2 EM 初始化

### 5.2.1 为什么用 EM 初始化而非随机初始化？

| 初始化方式 | γ_t 初始质量 | 收敛速度 | 风险 |
|-----------|------------|---------|------|
| 随机 | 接近均匀 (1/3, 1/3, 1/3) | 极慢 | MoE 收到噪声信号 |
| EM (hmmlearn) | 有意义的 regime 划分 | 快 | 可能有局部最优 |

**我们的策略：** 先用 hmmlearn 的 Baum-Welch 跑 EM（15 次随机重启，取最优），将结果导入 DifferentiableHMM，再做梯度微调。

### 5.2.2 参数映射

```python
def init_from_hmmlearn(self, hmm_model):
    with torch.no_grad():
        # π
        self.log_pi.copy_(log(tensor(hmm_model.startprob_) + ε))
        # A
        self.log_A.copy_(log(tensor(hmm_model.transmat_) + ε))
        # μ
        self.mu.copy_(tensor(hmm_model.means_))
        # Σ（对角）
        self.log_sigma.copy_(0.5 * log(tensor(diag_covs) + ε))
```

**注意：** EM 的参数是在**观测似然**意义下最优的，不保证在**预测损失**意义下最优。这正是 Stage 2 联合训练的价值所在。

---

# Part VI: Hyperparameter Analysis

## 6.1 α 的调节原则

α 控制 HMM 似然正则的强度：

| α 值 | 效果 | 适用场景 |
|------|------|---------|
| ≈ 0 | HMM 完全自由，可能学出"对预测有用但统计上无意义"的 regime | 数据量大，信任 MoE |
| 0.05 | 轻微 HMM 约束 | **推荐起点** |
| 0.1 | 中等约束 | 默认值 |
| 0.3 | 强约束 | HMM regime 必须有统计意义 |
| → ∞ | HMM 退化为标准 EM，MoE 无法影响 regime | 不推荐 |

**建议：** α ∈ {0.05, 0.1, 0.2, 0.3} 网格搜索，用验证集 DA 选最优。

## 6.2 超参敏感性分析（消融维度）

以下超参可以作为论文消融实验的维度：

| 超参 | 默认值 | 搜索范围 | 预期影响 |
|------|--------|---------|---------|
| α | 0.1 | {0.01, 0.05, 0.1, 0.2, 0.3} | DA ±5% |
| N (regimes) | 3 | {2, 3, 4} | 架构变化 |
| T_hmm (HMM窗口) | 52 | {26, 52, 104} | 计算量 |
| lr_hmm | 5e-5 | {1e-5, 5e-5, 1e-4} | 收敛速度 |
| patience | 40 | {20, 30, 40, 50} | 训练时长 |

---

# Part VII: Comparison with Related Approaches

## 7.1 vs Neural HMM (Tran et al., 2019)

| 方面 | Neural HMM | Our E2E HMM-MoE |
|------|-----------|-----------------|
| 任务 | 序列标注（NER, POS） | 金融方向预测 |
| HMM 参数 | 发射概率由神经网络参数化 | 全部参数（π, A, μ, Σ）可微 |
| 门控 | 无 | MoE gating 由 γ_t 驱动 |
| 损失 | 标注损失 + HMM 正则 | 预测损失 + 方向损失 + HMM 正则 + 均衡 |

## 7.2 vs 标准断开 HMM-MoE

| 方面 | 断开架构 | 端到端架构 |
|------|---------|-----------|
| HMM 训练 | EM (hmmlearn) | EM 初始化 + 梯度微调 |
| γ_t 的梯度 | 无（numpy array） | 有（requires_grad=True） |
| HMM 优化目标 | 观测似然 | 观测似然 + 预测准确率 |
| Regime 质量 | 对统计有意义 | 对统计和预测都有意义 |
| 实现复杂度 | 低（调用 hmmlearn） | 中（手写前向算法） |
| 数值稳定性 | hmmlearn 处理 | 需要手动 log-space |

---

# Part VIII: Implementation Notes

## 8.1 参数量分析

| 模块 | 参数量 | 占比 |
|------|--------|------|
| DifferentiableHMM | 48 | 0.16% |
| Expert × 3 (LSTM) | ~30,000 | 97.5% |
| Gating Network | ~750 | 2.4% |
| **Total** | **~30,790** | 100% |

HMM 只有 48 个参数，但通过 γ_t 控制了整个 MoE 的路由决策。这种"少量参数撬动大量参数"的设计正是端到端架构的效率所在。

## 8.2 计算复杂度

**HMM 前向算法：** $O(B \cdot T_{hmm} \cdot N^2)$

- 每个样本独立跑一次前向算法
- $B=16, T_{hmm}=52, N=3$：$16 \times 52 \times 9 = 7,488$ 次基本运算

**MoE 推理：** $O(B \cdot T_{seq} \cdot d_{model})$

- $B=16, T_{seq}=104, d_{model}=32$：主要计算量在 LSTM

**总训练时间：** CPU 上约 5-10 分钟（200 epoch）

---

*最后更新：2026-04-06*
