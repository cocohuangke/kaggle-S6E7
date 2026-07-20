# Kaggle Playground Series S6E7 — 学生健康风险预测

> 竞赛链接：https://www.kaggle.com/competitions/playground-series-s6e7

## 项目概述

| 项目 | 内容 |
|------|------|
| **任务** | 三分类：at-risk / unhealthy / fit |
| **评估指标** | Balanced Accuracy Score（宏平均召回率） |
| **训练集** | 690,088 行，13 原始特征（7 数值 + 6 类别） |
| **测试集** | 295,753 行 |
| **数据性质** | 合成数据 |
| **类别分布** | at-risk 85.87% / unhealthy 8.36% / fit 5.77%（极端不平衡） |
| **当前最佳 LB** | **0.95083** |
| **公开榜 Top1** | 0.95102 |
| **差距** | 0.00021（21 个基点） |

## 特征工程

### 原始特征（13d）
- 7 个数值特征 + 6 个类别特征
- `stress_level × physical_activity_level` 是决定性交互项（几乎能直接确定目标）

### 特征迭代历程

| 阶段 | 维度 | 描述 | LB |
|------|------|------|-----|
| 原始 | 13d | 7 num + 6 cat | 0.94972 |
| +stress_pal | 14d | 加入 stress×pal 交互 | 0.95004 |
| +num2cat | 17d | 加入 calorie_cat_, water_cat2_, step_cat_ | 0.95012 |

### 关键发现

- **110d 过拟合**：13d → 110d 扩维后 LB 反降，13d 实际优于 110d
- **Target Encoding 对 GBDT 无效**：所有 GBDT 模型加 TE 后均变差，但 RealMLP 内部可用
- **num2cat 微增**：+0.00008 LB
- **NaN 缺失模式无信号**：Cramér's V < 0.003，所有 NaN 模式对目标无预测力
- **最佳特征集**：17d = 7 num + 10 cat（原始 6 + stress_pal + 3 num2cat）

## 模型管线演进

| 版本 | 配置 | LB | 备注 |
|------|------|-----|------|
| V1 baseline | 13d + XGB+CB+HGB BA² blend + CA | 0.94958 | 初始基线 |
| 13d baseline | 13d + 3-model blend + CA | 0.94972 | |
| 13d tuned | CB depth 7→5 | 0.94978 | |
| 14d stress_pal | +stress×pal 交互 | 0.95004 | **首次突破 0.95** |
| A_num2cat | +3 num2cat 特征 | 0.95012 | |
| RealMLP single | 神经网络（yekenot 架构） | 0.95061 | 单模型即超 GBDT blend |
| RM+GB_A blend | RM0.55+GB_A0.45 | 0.95065 | |
| **champion_v1** | **RM0.57+4way_numte0.43（逐类权重）** | **0.95083** | **当前最佳** |

### 当前最佳架构（champion_v1.yaml）

```
RM_ref (OOF=0.95060)  ←  RealMLP, epochs=3, seed=63, GP features
GB_4way (OOF=0.95063) ←  XGB:0.02 + CB:0.06 + HGB:0.42 + RM_old:0.50

逐类权重 [at-risk, fit, unhealthy]:
  RM_ref:  [0.60, 0.32, 0.60]
  GB_4way: [0.40, 0.68, 0.40]
```

复现命令：
```bash
python run_v2.py configs/champion_v1.yaml          # 快速（使用预计算 OOF）
python run_v2.py configs/champion_v1_full.yaml      # 完整训练
```

## 核心发现

### 1. OOF ↑ ≠ LB ↑
多次实验表明 OOF 提升但 LB 回归。OOF 不是可靠的模型选择信号。

### 2. CA 校准无效
历史数据证明 CA/noca 产出相同 LB，CA 仅膨胀 OOF 而不改善 LB。

### 3. Health 指标
```
health = max(单模型 OOF) - blend OOF
health < 0 → 过拟合信号
```

### 4. 信息不对称杀死 blend
各模型使用不同特征集 → OOF 高但 LB 低。特征一致性至关重要。

### 5. GBDT 模型高度相关
XGB/CB/HGB 之间相关性 >0.99，RealMLP 是唯一的真正多样性来源。

### 6. GBDT ≠ NN 特征工程
| 技术 | GBDT | RealMLP |
|------|------|---------|
| 70 bins | ✗ | ✓ |
| 271K cardinality 交互 | ✗ | ✓ |
| fill=0 | ✗ | ✓ |
| Target Encoding | ✗ | ✓ |
| stress_pal | ✓ 帮助 | ✗ 伤害（NTP 缩放稀释信号） |

### 7. LB 预测公式 v7
```
predicted_LB = blend_OOF + 0.00049 + 1.227 × health
MAE = 0.00014
```

## 失败实验记录

| 实验 | 结果 | 原因 |
|------|------|------|
| Seed bagging (5-seed) | health 持负，LB 无提升 | 种子多样性不足 |
| 14d tuned (HGB d8→d5) | health 变负，LB 回归 | 过拟合 |
| 15d sleep_stress | OOF 虚高（CA），LB=0.94966 | CA 伪提升 |
| 三阶交互 (64 值) | OOF 高，LB=0.94978 | 过拟合 |
| Per-model 特征集 (PMv1/v2) | 信息不对称伤害 blend | 特征不一致 |
| numTE blend (所有变体) | GBDT 特征不一致杀死 LB | TE 对 GBDT 有害 |
| Stacking (LR meta-learner) | 变差 | - |
| Prior calibration / NNLS blend | 失败 | - |

## 项目结构

```
kaggle-S6E7/
├── pipeline/              # 核心管线
│   ├── fe.py              # 特征工程
│   ├── train.py           # GBDT 训练
│   ├── blend.py           # 模型融合
│   ├── realmlp.py         # RealMLP 神经网络
│   ├── autogluon.py       # AutoGluon 集成
│   └── optuna_tune.py     # 超参调优
├── configs/               # 29 个 YAML 配置
│   └── champion_v1.yaml   # 最佳复现配置
├── docs/                  # 15 篇详细技术文档
├── analysis/              # 交互分析脚本
├── data/                  # 原始数据
├── oof/                   # OOF 预测缓存
├── output/                # 管线输出
├── submissions/           # 提交文件
└── notebook/              # 探索 notebook
```

## 技术栈

| 组件 | 版本/规格 |
|------|-----------|
| Python | 3.13 |
| PyTorch | 2.9.1+cu130 |
| GPU | NVIDIA RTX 3060 12GB |
| GBDT | XGBoost + CatBoost + HistGradientBoosting |
| Neural | RealMLP (yekenot 架构) |
| 管线 | Config-driven YAML |

## 下一步计划

- [ ] 通用 FE 架构重构（设计完成，实现待做）
- [ ] 多族集成：AutoGluon + XGB-OvR
- [ ] 逐类 CA 校准（Top 方案使用此技术）
- [ ] 外部预测融合

## 文档索引

| 编号 | 文档 | 内容 |
|------|------|------|
| 01 | 项目分析 | 数据概览、类别分布、基线 |
| 02 | 特征工程 | 交互项发现、扩维实验 |
| 03a | 模型训练-baseline | 三模型基线建立 |
| 03b | 模型训练-调参 | GBDT 超参搜索 |
| 03c | 模型调优-14d调参 | 14d 维度调参 |
| 03d | 模型调优-seed bagging | 种子袋装实验 |
| 03e | 模型调优-增维 | 维度扩展实验 |
| 03f | NaN交互分析 | NaN 缺失模式分析 |
| 04 | 特征工程-迭代调优 | num2cat、TE 迭代 |
| 05 | 分模型特征集 | Per-model 特征实验 |
| 06 | 模型训练-realmlp | RealMLP 架构与训练 |
| 09 | LB预测公式 | health 指标与 LB 预测 |
| 10 | 通用FE架构设计 | 重构设计方案 |
| - | RealMLP重构日志 | RealMLP 重构过程 |
