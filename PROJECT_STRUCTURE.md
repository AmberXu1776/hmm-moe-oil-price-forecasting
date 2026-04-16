# P1_MambaMoE_OilPrice: Project Structure

This is the code and manuscript for the paper **HMM-MoE: Hidden Markov Model Enhanced Mixture-of-Experts for Oil Price Directional Forecasting**.

## 📁 Directory Structure

```
P1_MambaMoE_OilPrice/
├── PAPER.md                 # Full paper manuscript in Markdown (latest version)
├── main.tex                 # LaTeX source in Elsevier elsarticle format (ready for submission)
├── Template.md              # Elsevier LaTeX template reference
├── PROJECT_STRUCTURE.md     # This file — project structure documentation
├── data/
│   └── all_features.csv     # Raw dataset (weekly: 1998-2022, 1105 observations)
├── code/
│   ├── main.py              # Main training and evaluation entry point
│   ├── config.py            # Global configuration (seeds, hyperparameters, paths)
│   ├── multi_seed.py        # Multi-seed (42, 123, 2024) main experiment
│   ├── rolling_window_v2.py # 8-window rolling validation with DM test
│   ├── hyperparam_sensitivity.py # Hyperparameter sensitivity (λ, E, L)
│   ├── hmm_leakage_test.py  # HMM leakage test: full-sample vs train-only HMM
│   ├── fair_baseline_comparison.py # First version fair baseline comparison (all neural 3-seed direction loss)
│   ├── fair_baseline_v2.py  # Fixed version: HMM-MoE first, then baselines (avoids GPU state pollution)
│   ├── dropout_sensitivity.py # Original parallel dropout sensitivity (OOM on GPU)
│   ├── dropout_sensitivity_sequential.py # Sequential version (one seed at a time)
│   ├── run_dropout_part2.py # Part-based version (one dropout per run)
│   ├── run_one_seed.py      # Single seed per process (full memory cleanup)
│   ├── run_one_seed_small.py # Small model (d_model=16) for GPU
│   ├── run_one_seed_cpu.py  # CPU version for stable runs (avoids OOM)
│   ├── hmm_lstm_baseline.py # HMM+LSTM baseline comparison
│   ├── debug_seed42.py      # Debug script for seed=42
│   └── src/
│       ├── data.py          # Data loading and preprocessing
│       ├── hmm.py           # HMM training and filtering
│       ├── model.py         # HMM-MoE model architecture
│       ├── layer4_mamba_moe.py # Core model implementation with Mamba (note: actually uses LSTM experts in final version)
│       └── evaluate.py      # Evaluation metrics (DA, MAE, RMSE, MAPE, DM test)
├── output/                   # Experimental results (CSV)
│   ├── fair_baseline_v2.csv  # Fair baseline comparison results (all 3-seed direction loss)
│   ├── hyperparam_lambda.csv  # λ sensitivity results
│   ├── hyperparam_E.csv      # Number of experts sensitivity
│   ├── hyperparam_L.csv      # Lookback window L sensitivity
│   ├── dropout_0.3.csv       # Dropout=0.3 results (3 seeds)
│   ├── dropout_0.4.csv       # Dropout=0.4 results (3 seeds)
│   ├── dropout_0.55.csv      # Dropout=0.55 results (3 seeds)
│   └── dropout_0.7.csv       # Dropout=0.7 results (3 seeds)
└── figures/                  # Plots (will be generated)
    └── hmm_state_sequence.png # HMM state sequence vs major oil events
```

## 📋 Experiment Reproducibility

All main experiments can be reproduced with the following scripts:

| Experiment | Script | Notes |
|------------|--------|-------|
| Main result (3-seed) | `code/multi_seed.py` | Produces Table 1 results |
| 8-window rolling validation | `code/rolling_window_v2.py` | Produces Section 5.5 results |
| Hyperparameter sensitivity (λ, E, L) | `code/hyperparam_sensitivity.py` | Produces Tables 8-10 |
| Dropout sensitivity (4 values × 3 seeds) | Run `code/run_one_seed_cpu.py` for each combination | Results in `output/dropout_*.csv` → Table 11 |
| HMM leakage test | `code/hmm_leakage_test.py` | Full-sample vs train-only HMM |
| Fair baseline comparison | `code/fair_baseline_v2.py` | All neural baselines with 3-seed direction loss |
| HMM+LSTM baseline | `code/hmm_lstm_baseline.py` | Section 5.7 comparison |

## 🔑 Key Hyperparameters (Main Experiment)

```python
SEEDS = [42, 123, 2024]          # Random seeds for robustness
N_REGIMES = 3                     # HMM number of regimes (selected by BIC/AIC)
N_EXPERTS = 3                     # Number of experts in MoE
SEQ_LEN = 52                      # Lookback window L (one year weekly)
DROPOUT = 0.55                    # Dropout rate (selected by sensitivity)
LOSS_ALPHA = 1.0                  # Direction loss weight λ
LR = 1e-4                         # Learning rate
BATCH_SIZE = 16                   # Batch size (reduced from 32 for memory stability)
D_MODEL = 32                       # Model dimension (full model)
```

## 📝 Revision Status (as of 2026-04-16)

- [x] v0.1-v0.7: Initial draft, main experiments
- [x] v0.7b: First RODEOS review → Major Revision
- [x] Dropout sensitivity experiment completed (all 12 runs) ✓
- [x] HMM leakage experiment completed (DA difference = 0.0pp) ✓
- [x] Fair baseline comparison completed (all neural 3-seed direction loss) ✓
- [ ] Compress Abstract from ~350 words to 150-200 words
- [ ] List 8 rolling window time intervals
- [ ] Add HMM state visualization (vs major oil events)
- [ ] Clarify Table 1 vs Table 11 DA difference (different model size)
- [ ] Clarify gate input dimension (6+16+2 = 21)
- [ ] Add pairwise DM tests vs top baselines
- [ ] Add citations for feature selection

---

*Last updated: 2026-04-16*
