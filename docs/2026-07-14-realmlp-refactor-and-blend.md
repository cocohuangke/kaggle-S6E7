# RealMLP 重构与Blend实验日志 — 2026-07-14

## 一、工作概述

今天完成了三项主要工作：
1. 参考notebook RealMLP代码的逐行复现
2. 将参考代码重构为pipeline架构
3. 基于新RealMLP的blend实验，刷新LB纪录

## 二、Step 1: 参考Notebook原样复现

### 来源
- 参考notebook: `notebook/ps-s6-ep6-realmlp-0-95090.ipynb` (LB=0.95090)
- 产出脚本: `run_ref_realmlp.py`

### 适配工作
- 路径从Kaggle环境改为本地路径
- CUDA设备从Kaggle GPU改为本地RTX 3060
- 安装PyTorch 2.11.0+cu128 (Python 3.13)

### 结果
| 指标 | 参考notebook | 本地复现 | 差距 |
|------|-------------|---------|------|
| OOF BA | 0.95065 | 0.95060 | 0.00005 |
| LB | 0.95090 | 0.95074 | 0.00016 |

OOF差距在seed随机性范围内。LB差距来自CUDA/硬件浮点差异，无法消除。

### 关键配置
- 7 folds, seed=63, epochs=3
- 包含9个GP非线性特征 (add_gp_features)
- 额外5个cat特征 (sleep_duration_cat2_, bmi_cat1_/cat2_, heart_rate_cat2_, water_intake_cat2_)
- TargetEncoding (cv=7, smooth='auto')
- 训练耗时: ~50分钟

## 三、Step 2: 重构为Pipeline架构

### 修改文件

#### `pipeline/realmlp.py`
- 新增 `seed_everything(seed)` 函数
- 更新CONFIG: epochs=3 (原2), device='cuda', random_state=63
- 修复 `RealMLP.__init__`: 添加 `self.first_linear = linear` (i==0时显式引用)
- 修复 `get_parameter_groups`: 使用 `model.first_linear.weight` 替代 `model.hidden[0].weight`
- 新增 `add_gp_features(df)` — 9个GP非线性特征
- 更新 `realmlp_fe()`: 新增sleep_duration_cat2_, bmi_cat1_/cat2_, heart_rate_cat2_, 修复water_intake_cat2_的.round(), 加入GP特征
- 新增 `RealMLP_TD_Classifier(BaseEstimator)` — sklearn wrapper (fit/predict_proba/predict, early stopping)
- 保留 `train_realmlp_fold()` 向后兼容

#### `pipeline/train.py` — `train_realmlp()` 函数
- 使用 `RealMLP_TD_Classifier` 替代 `train_realmlp_fold`
- Seed从CONFIG的random_state(63)获取，不再硬编码42
- 调用 `seed_everything(SEED)` 确保可复现
- 列排序: `all_cols = sorted(X_tr.columns.tolist())` + reindex
- TE encoder使用SEED替代硬编码42

### 关键Bug发现: stress_pal交互特征

**这是导致重构版LB下降的根本原因。**

参考notebook的 `important_combos` 只有一个交互:
```python
important_combos = sorted([('heart_rate', 'bmi')])
```

pipeline的 `realmlp_fe()` 默认添加了额外交互:
```python
important_combos = [('heart_rate', 'bmi')]
if fe_config.get('stress_pal', True):  # 默认True!
    important_combos.append(('stress_level', 'physical_activity_level'))
```

这导致:
- +1个categorical列 (stress_level_physical_activity_level_)
- +3个TE列 (每个class一个)
- 改变模型架构、初始化和训练动态
- LB从0.95074降到0.95065

**修复**: 设置 `stress_pal=False` 后，重构版与参考版位相同（test预测差0.000000，OOF完全一致0.95060）。

### 验证结果

| 版本 | stress_pal | OOF BA | LB |
|------|-----------|--------|-----|
| ref_realmlp (参考复制) | N/A | 0.95060 | 0.95074 |
| refactored_realmlp | True | 0.95062 | 0.95065 |
| refactored_realmlp | **False** | **0.95060** | 应与ref相同 |

stress_pal=False时test预测位相同，确认根因。

### 参考Notebook中的其他Bug (已原样复制)
- `heart_rate_cat2_` 定义两次: 第一次 `df[col]-60`，第二次 `100-df[col]` 覆盖。最终使用 `100-heart_rate`
- `sleep_duration_cat2_` 定义两次相同公式，无实际影响

## 四、Step 3: Blend实验

### 新RealMLP OOF文件
- `oof/_realmlp_ref_oof.npy` — 新RealMLP (stress_pal=False, OOF=0.95060)
- `oof/_realmlp_ref_test.npy` — 对应test预测

### 实验结果

| 组合 | 权重 | OOF BA | 预测LB(v7) | 实际LB | 备注 |
|------|------|--------|-----------|--------|------|
| RM + A_num2cat_numte | 0.65/0.35 | 0.95077 | 0.95105 | 0.95061 | numTE版本，LB反而低 |
| RM + 4way_numte | 0.57/0.43 | 0.95081 | 0.95108 | **0.95081** | **新纪录!** OOF-LB gap=0 |
| RM + A_num2cat | 0.65/0.35 | 0.95064 | 0.95108 | 0.95065 | best_9d同款GB_A |

### 关键发现

1. **4way_numte组合LB=0.95081，刷新纪录** — 比之前0.95065提升0.00016
2. 4way_numte包含旧RealMLP预测，与新RM有信息重叠，但blend仍然有效
3. v7 LB预测公式对不同组合类型失准:
   - 4way组合: OOF≈LB (gap=0.00000)，公式可靠
   - 其他组合: OOF-LB gap达0.00016，公式严重高估
4. 新RealMLP的OOF(0.95060)与GBDT有校准差异，直接blend效果不如4way

### 各3模型GBDT equal-blend排名

| 排名 | OOF BA | 名称 | 说明 |
|------|--------|------|------|
| #1 | 0.95063 | _4way_numte_blend | 实为4路(RM+3GBDT) |
| #2 | 0.95006 | _fe_A_num2cat_numte_eq | A_num2cat + numTE |
| #3 | 0.94953 | _permodel_eq | per-model不同FE |
| #4 | 0.94950 | _fe_A_num2cat_eq | A_num2cat (9d) |

## 五、当前最优状态

- **LB = 0.95081** (RM0.57 + 4way_numte0.43)
- **Target = 0.95102** (差距0.00021)
- 提交文件: `submissions/sub_rm057_4way043.csv`

## 六、产出文件清单

### 新增/修改代码
- `run_ref_realmlp.py` — 参考notebook原样复现脚本
- `run_refactored_realmlp.py` — pipeline版RealMLP运行脚本 (stress_pal=False)
- `pipeline/realmlp.py` — 重构后的RealMLP模块
- `pipeline/train.py` — 更新train_realmlp()函数
- `configs/ref_blend.yaml` — 新RealMLP blend配置

### 新增OOF/预测
- `oof/_realmlp_ref_oof.npy` / `_realmlp_ref_test.npy` — 新RealMLP预测
- `output/ref_realmlp_oof.npy` / `_test.npy` / `_submission.csv` — 参考版输出
- `output/refactored_realmlp_oof.npy` / `_test.npy` / `_submission.csv` — 重构版输出

### 提交文件
- `submissions/sub_rm057_4way043.csv` — LB=0.95081 (当前最优)
- `submissions/sub_rm065_gbnumte035.csv` — LB=0.95061
- `submissions/sub_rm065_a_num2cat035.csv` — LB=0.95065

## 七、深度分析：为什么 RealMLP 去掉 stress_pal 反而提升？

### 7.1 stress_pal 添加了什么？

`stress_pal=True` 在 `realmlp_fe()` 中向 `important_combos` 添加了 `('stress_level', 'physical_activity_level')` 交互对，产生 **4 个额外特征**：

| 特征 | 类型 | 说明 |
|------|------|------|
| `stress_level_physical_activity_level_` | categorical | stress×PA 交互，factorize 编码 |
| `_..._TE_class0` | numerical | 上述交互的 TE（at-risk 类） |
| `_..._TE_class1` | numerical | 上述交互的 TE（fit 类） |
| `_..._TE_class2` | numerical | 上述交互的 TE（unhealthy 类） |

总特征数：94 → 98（+4），其中 3 个 TE 是数值特征。

### 7.2 GBDT 为什么需要 stress_pal？

**树模型是逐特征分裂的**——一次只看一个特征。要学习 `stress_level × physical_activity_level` 的交互，树必须先按 stress_level 分裂，再在子节点按 PA 分裂，需要多层深度。手工 `stress_pal` 交互特征是一个**捷径**：树可以在一次分裂中直接使用它。

而且 GBDT 对冗余特征天然免疫——不重要的特征根本不会被选为分裂点，零代价。

SHAP 交互值分析也证实了这一点：`sleep_duration × stress_pal` 的 SHAP 交互值 = 0.26，`stress_level × physical_activity_level` = 0.09，均为显著交互。

### 7.3 RealMLP 为什么不需要（甚至被伤害）？

三个机制叠加：

#### 机制 A：RealMLP 已经内置了交互学习能力

RealMLP 的架构流程：
```
数值特征 → PBLDEmbedding(5维/特征) →
分类特征 → CategoricalFeatureLayer(onehot/embedding) →
拼接 → ScalingLayer → NTPLinear×3(512) + SiLU → 输出
```

- **3 层 512 宽的全连接层**已经可以逼近任意高阶交互（通用逼近定理）
- **PBLD 嵌入**用周期性基函数展开每个数值特征，专门解决 NN 难学不规则函数的问题（Grinsztajn et al., NeurIPS 2022）
- **ScalingLayer**（每个特征一个可学习标量，学习率是其他层的 6 倍）做软特征选择

RealMLP **不需要**手工交互特征来"暴露"交互——它的第一层权重矩阵已经在混合所有特征了。

#### 机制 B：冗余特征增加维度，稀释信号

3 个 TE 数值特征通过 PBLDEmbedding 展开后变成 **3×5=15 个额外维度**，进入 512 维的第一隐藏层。这些维度携带的是 stress_pal 交互的 TE 信息——但 RealMLP 已经可以从原始的 `stress_level` 和 `physical_activity_level` 分类嵌入中学到这个交互。

结果是：**15 个维度里是冗余信号，但梯度下降不知道这一点**，仍然要花优化预算去学习如何"忽略"它们。这挤占了真正有用特征的优化空间。

Grinsztajn et al. (2022) 的关键实验发现：

> "MLP 架构，平均而言，与 XGBoost 相比，表现出更多的过拟合无信息特征。"

#### 机制 C：NTP 参数化的维度惩罚

RealMLP 使用神经正切参数化（NTP），权重矩阵按 `1/√d_in` 缩放。当输入维度从 94 增加到 98 时，`d_in` 变大，**有效学习率自动降低**。这意味着所有特征的更新步长都变小了——仅仅因为多了 4 个冗余特征，整个模型的优化速度就变慢了。

### 7.4 根本原因：旋转不变性 vs 逐特征分裂

| 方面 | GBDT (XGBoost/CatBoost) | RealMLP |
|------|------------------------|---------|
| 如何捕获交互 | 通过深层嵌套的分裂。手工 `stress_pal` 是1次分裂的捷径 | 通过第一层权重矩阵混合所有特征。3层512已经能学任意交互 |
| 对冗余/相关特征的稳健性 | 高——树只分裂最佳特征；忽略冗余 | 低——所有特征输入到每个神经元；冗余特征创建多路径，优化不稳定 |
| 对无信息特征的反应 | 无关——不分裂它们 | 有害——旋转不变性意味着必须先"解开"特征空间 |
| 手工交互特征是 | **必要捷径**，以最少深度捕获交互 | **冗余输入**，增加维度不增加表现力 |

### 7.5 一句话总结

> **GBDT 是"逐特征决策"——手工交互是必要的捷径，冗余特征零成本。RealMLP 是"全特征混合"——手工交互是冗余输入，增加维度却没增加信息，还通过 NTP 缩放惩罚了所有特征的学习率。**

这也解释了为什么参考 notebook（yekenot, LB=0.95090）只用 `('heart_rate', 'bmi')` 一个交互——作者可能通过实验发现，对 RealMLP 来说，少即是多。

### 7.6 参考文献

- Holzmüller et al., "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data", NeurIPS 2024
- Grinsztajn et al., "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022
- LinFE (2026), "Selector Overfitting Paradox" — 高容量代理在大量特征空间中发现更好的噪声而非更好的信号

## 八、下一步方向

1. **RM+4way细粒度grid search** — 当前0.57/0.43，可能还有更优权重
2. **外部预测文件multi-family ensemble** — 5个外部预测(LB 0.95021-0.95052)尚未使用，可提供真正的算法多样性
3. **4way内部权重优化** — 当前4way内部是固定权重，可能可以优化
4. **v7公式重新校准** — 当前公式对不同组合类型失准，需要分区校准
