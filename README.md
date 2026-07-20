# Kaggle Playground Series S6E7 — Student Health Risk Prediction

> Competition: https://www.kaggle.com/competitions/playground-series-s6e7

## Overview

| Item | Detail |
|------|--------|
| **Task** | 3-class classification: at-risk / unhealthy / fit |
| **Metric** | Balanced Accuracy Score (macro recall) |
| **Train set** | 690,088 rows, 13 original features (7 numeric + 6 categorical) |
| **Test set** | 295,753 rows |
| **Data** | Synthetic |
| **Class distribution** | at-risk 85.87% / unhealthy 8.36% / fit 5.77% (extreme imbalance) |
| **Current best LB** | **0.95083** |
| **Public leaderboard top** | 0.95102 |
| **Gap** | 0.00021 (21 basis points) |

## Feature Engineering

### Original Features (13d)
- 7 numeric + 6 categorical
- `stress_level × physical_activity_level` is the dominant interaction (nearly determines the target)

### Feature Iteration

| Stage | Dims | Description | LB |
|-------|------|-------------|-----|
| Original | 13d | 7 num + 6 cat | 0.94972 |
| +stress_pal | 14d | Added stress×pal interaction | 0.95004 |
| +num2cat | 17d | Added calorie_cat_, water_cat2_, step_cat_ | 0.95012 |

### Key Findings

- **110d overfits**: 13d → 110d expansion lowered LB; 13d actually beats 110d
- **Target Encoding fails for GBDT**: All GBDT models worsened with TE, but RealMLP can use it internally
- **num2cat marginal gain**: +0.00008 LB
- **NaN missing patterns carry zero signal**: Cramér's V < 0.003 for all NaN-target pairs
- **Best feature set**: 17d = 7 num + 10 cat (original 6 + stress_pal + 3 num2cat)

## Pipeline Evolution

| Version | Config | LB | Note |
|---------|--------|-----|------|
| V1 baseline | 13d + XGB+CB+HGB BA² blend + CA | 0.94958 | Initial baseline |
| 13d baseline | 13d + 3-model blend + CA | 0.94972 | |
| 13d tuned | CB depth 7→5 | 0.94978 | |
| 14d stress_pal | +stress×pal interaction | 0.95004 | **First 0.95+ break** |
| A_num2cat | +3 num2cat features | 0.95012 | |
| RealMLP single | Neural network (yekenot architecture) | 0.95061 | Single model beats GBDT blend |
| RM+GB_A blend | RM0.55+GB_A0.45 | 0.95065 | |
| **champion_v1** | **RM0.57+4way_numte0.43 (per-class weights)** | **0.95083** | **Current best** |

### Champion Architecture (champion_v1.yaml)

```
RM_ref (OOF=0.95060)  ←  RealMLP, epochs=3, seed=63, GP features
GB_4way (OOF=0.95063) ←  XGB:0.02 + CB:0.06 + HGB:0.42 + RM_old:0.50

Per-class weights [at-risk, fit, unhealthy]:
  RM_ref:  [0.60, 0.32, 0.60]
  GB_4way: [0.40, 0.68, 0.40]
```

Reproduce:
```bash
python run_v2.py configs/champion_v1.yaml          # Fast (uses pre-computed OOF)
python run_v2.py configs/champion_v1_full.yaml      # Full training
```

## Core Discoveries

### 1. OOF ↑ ≠ LB ↑
Multiple experiments showed OOF improvement but LB regression. OOF is not a reliable model selection signal.

### 2. CA Calibration Is Ineffective
Historical data proves CA/noca produce identical LB; CA only inflates OOF without improving LB.

### 3. Health Metric
```
health = max(single_model_OOF) - blend_OOF
health < 0 → overfitting signal
```

### 4. Information Asymmetry Kills Blend
Per-model different feature sets → high OOF but low LB. Feature consistency is critical.

### 5. GBDT Models Are Highly Correlated
XGB/CB/HGB correlations >0.99; RealMLP is the only real diversity source.

### 6. GBDT ≠ NN Feature Engineering
| Technique | GBDT | RealMLP |
|-----------|------|---------|
| 70 bins | ✗ | ✓ |
| 271K cardinality interactions | ✗ | ✓ |
| fill=0 | ✗ | ✓ |
| Target Encoding | ✗ | ✓ |
| stress_pal | ✓ helps | ✗ hurts (NTP scaling dilutes signal) |

### 7. LB Prediction Formula v7
```
predicted_LB = blend_OOF + 0.00049 + 1.227 × health
MAE = 0.00014
```

## Failed Experiments

| Experiment | Result | Cause |
|------------|--------|-------|
| Seed bagging (5-seed) | Health stayed negative, no LB gain | Insufficient seed diversity |
| 14d tuned (HGB d8→d5) | Health went negative, LB regressed | Overfitting |
| 15d sleep_stress | OOF inflated (CA), LB=0.94966 | CA pseudo-improvement |
| Triple interaction (64 values) | OOF high, LB=0.94978 | Overfitting |
| Per-model feature sets (PMv1/v2) | Information asymmetry hurts blend | Feature inconsistency |
| numTE blend (all variants) | GBDT feature inconsistency kills LB | TE harmful for GBDT |
| Stacking (LR meta-learner) | Worse | - |
| Prior calibration / NNLS blend | Failed | - |

## Project Structure

```
kaggle-S6E7/
├── pipeline/              # Core pipeline
│   ├── fe.py              # Feature engineering
│   ├── train.py           # GBDT training
│   ├── blend.py           # Model blending
│   ├── realmlp.py         # RealMLP neural network
│   ├── autogluon.py       # AutoGluon integration
│   └── optuna_tune.py     # Hyperparameter tuning
├── configs/               # 29 YAML configs
│   └── champion_v1.yaml   # Best reproducible config
├── docs/                  # 15 detailed technical documents
├── analysis/              # Interaction analysis scripts
├── data/                  # Raw data
├── oof/                   # OOF prediction cache
├── output/                # Pipeline output
├── submissions/           # Submission files
└── notebook/              # Exploration notebooks
```

## Tech Stack

| Component | Version/Spec |
|-----------|-------------|
| Python | 3.13 |
| PyTorch | 2.9.1+cu130 |
| GPU | NVIDIA RTX 3060 12GB |
| GBDT | XGBoost + CatBoost + HistGradientBoosting |
| Neural | RealMLP (yekenot architecture) |
| Pipeline | Config-driven YAML |

## Next Steps

- [ ] General FE architecture refactor (design complete, implementation pending)
- [ ] Multi-family ensemble: AutoGluon + XGB-OvR
- [ ] Per-class CA calibration (top solution uses this)
- [ ] External prediction blending

## Documentation Index

| # | Document | Content |
|---|----------|---------|
| 01 | 项目分析 | Data overview, class distribution, baseline |
| 02 | 特征工程 | Interaction discovery, expansion experiments |
| 03a | 模型训练-baseline | Three-model baseline |
| 03b | 模型训练-调参 | GBDT hyperparameter search |
| 03c | 模型调优-14d调参 | 14d tuning |
| 03d | 模型调优-seed bagging | Seed bagging experiment |
| 03e | 模型调优-增维 | Dimension expansion experiments |
| 03f | NaN交互分析 | NaN missing pattern analysis |
| 04 | 特征工程-迭代调优 | num2cat, TE iteration |
| 05 | 分模型特征集 | Per-model feature experiments |
| 06 | 模型训练-realmlp | RealMLP architecture & training |
| 09 | LB预测公式 | Health metric & LB prediction |
| 10 | 通用FE架构设计 | Refactor design |
| - | RealMLP重构日志 | RealMLP refactor log |
