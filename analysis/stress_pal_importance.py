#!/usr/bin/env python
"""Prove stress_pal is important, programmatically.

Compares GB models WITH vs WITHOUT stress_pal on the full 13d feature set.
Uses SHAP to rank stress_pal among all features.

Approach:
  - A_median:  num2cat (calorie_cat_, water_cat2_, step_cat_), median fill, NO stress_pal
  - AP_median: same + stress_pal (stress_level × physical_activity_level interaction)

For each model (XGB, CB, HGB):
  1. Train 5-fold CV on both configs
  2. Compare OOF BA
  3. SHAP feature importance on AP_median to see where stress_pal ranks

Output:
  - Terminal: per-model comparison + SHAP ranks
  - analysis/stress_pal_comparison.csv: raw OOF BA comparison
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# === Load data ===
train = pd.read_csv(os.path.join(BASE, 'data', 'train.csv'))

# FE from pipeline
from pipeline.fe import get_or_create_features, build_tag

# Target encoding
le = LabelEncoder()
y_all = le.fit_transform(train['health_condition'].values)

# === Configs ===
cfg_without_pal = {
    'num2cat': True,
    'stress_pal': False,
    'num_fill': 'median',
    'te': False,
}

cfg_with_pal = {
    'num2cat': True,
    'stress_pal': True,
    'num_fill': 'median',
    'te': False,
}

# === Get FE data ===
print("=== Feature Engineering ===")
X_no, X_test_no, cat_no, feat_no, tag_no = get_or_create_features(cfg_without_pal, y_all)
X_yes, X_test_yes, cat_yes, feat_yes, tag_yes = get_or_create_features(cfg_with_pal, y_all)

print(f"  NO  stress_pal: tag={tag_no}, dim={X_no.shape[1]}, features={feat_no}")
print(f"  YES stress_pal: tag={tag_yes}, dim={X_yes.shape[1]}, features={feat_yes}")

# Identify stress_pal column
stress_pal_col = [f for f in feat_yes if f not in feat_no]
print(f"  stress_pal column(s): {stress_pal_col}")

# === Train & compare for each model ===
N_SPLITS = 5
RANDOM = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM)

results = []

# ============ XGB ============
print("\n=== XGBoost ===")
for cfg_name, X, cat_idx in [("WITHOUT", X_no, cat_no), ("WITH", X_yes, cat_yes)]:
    oof = np.zeros((len(X), 3))
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
        import xgboost as xgb
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=1500, learning_rate=0.05, max_depth=7,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', device='cuda',
            early_stopping_rounds=50, random_state=RANDOM, verbosity=0,
            enable_categorical=True,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = model.predict_proba(X_val)
        fold_scores.append(balanced_accuracy_score(y_val, oof[val_idx].argmax(1)))

    oof_ba = balanced_accuracy_score(y_all, oof.argmax(1))
    fold_ba = np.mean(fold_scores)
    print(f"  XGB {cfg_name:>7s}: OOF BA={oof_ba:.5f}, mean fold BA={fold_ba:.5f}, folds={[f'{s:.5f}' for s in fold_scores]}")
    results.append({'model': 'XGB', 'config': cfg_name, 'oof_ba': oof_ba, 'mean_fold_ba': fold_ba, 'folds': fold_scores})

# ============ CatBoost ============
print("\n=== CatBoost ===")
for cfg_name, X, cat_idx in [("WITHOUT", X_no, cat_no), ("WITH", X_yes, cat_yes)]:
    oof = np.zeros((len(X), 3))
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
        from catboost import CatBoostClassifier
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]
        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.05, depth=5,
            l2_leaf_reg=1.0, auto_class_weights='Balanced',
            task_type='GPU', random_seed=RANDOM,
            early_stopping_rounds=100, verbose=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  cat_features=cat_idx, verbose=False)
        oof[val_idx] = model.predict_proba(X_val)
        fold_scores.append(balanced_accuracy_score(y_val, oof[val_idx].argmax(1)))

    oof_ba = balanced_accuracy_score(y_all, oof.argmax(1))
    fold_ba = np.mean(fold_scores)
    print(f"  CB  {cfg_name:>7s}: OOF BA={oof_ba:.5f}, mean fold BA={fold_ba:.5f}, folds={[f'{s:.5f}' for s in fold_scores]}")
    results.append({'model': 'CB', 'config': cfg_name, 'oof_ba': oof_ba, 'mean_fold_ba': fold_ba, 'folds': fold_scores})

# ============ HistGradientBoosting ============
print("\n=== HistGradientBoosting ===")
for cfg_name, X, cat_idx in [("WITHOUT", X_no, cat_no), ("WITH", X_yes, cat_yes)]:
    oof = np.zeros((len(X), 3))
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]
        model = HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.05, max_depth=8,
            min_samples_leaf=50, l2_regularization=1.0,
            max_leaf_nodes=31, class_weight='balanced',
            early_stopping=True, random_state=RANDOM,
            categorical_features=cat_idx,
        )
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)
        fold_scores.append(balanced_accuracy_score(y_val, oof[val_idx].argmax(1)))

    oof_ba = balanced_accuracy_score(y_all, oof.argmax(1))
    fold_ba = np.mean(fold_scores)
    print(f"  HGB {cfg_name:>7s}: OOF BA={oof_ba:.5f}, mean fold BA={fold_ba:.5f}, folds={[f'{s:.5f}' for s in fold_scores]}")
    results.append({'model': 'HGB', 'config': cfg_name, 'oof_ba': oof_ba, 'mean_fold_ba': fold_ba, 'folds': fold_scores})

# === Summary comparison ===
print(f"\n{'='*80}")
print("Summary: stress_pal marginal contribution per model")
print(f"{'='*80}")
print(f"{'Model':<6s} {'WITHOUT':>10s} {'WITH':>10s} {'Delta':>10s} {'% gain':>10s}")
print(f"{'-'*46}")

results_df = pd.DataFrame(results)
for model in ['XGB', 'CB', 'HGB']:
    sub = results_df[results_df['model'] == model]
    without = sub[sub['config'] == 'WITHOUT']['oof_ba'].values[0]
    with_ = sub[sub['config'] == 'WITH']['oof_ba'].values[0]
    delta = with_ - without
    pct = (delta / without) * 100 if without > 0 else 0
    print(f"  {model:<6s} {without:10.5f} {with_:10.5f} {delta:+10.5f} {pct:+9.3f}%")

# === SHAP analysis: stress_pal importance ranking ===
print(f"\n{'='*80}")
print("SHAP Analysis: where does stress_pal rank among all features?")
print(f"{'='*80}")

# Use XGBoost (best SHAP support)
fold0 = 0
tr_idx, val_idx = list(skf.split(X_yes, y_all))[fold0]
X_tr, X_val = X_yes.iloc[tr_idx], X_yes.iloc[val_idx]
y_tr, y_val = y_all[tr_idx], y_all[val_idx]

import xgboost as xgb
model_xgb = xgb.XGBClassifier(
    n_estimators=1500, learning_rate=0.05, max_depth=7,
    min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', device='cuda',
    early_stopping_rounds=50, random_state=RANDOM, verbosity=0,
    enable_categorical=True,
)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

try:
    import shap
    # Sample for efficiency
    n_sample = min(5000, len(X_val))
    X_sample = X_val.sample(n=n_sample, random_state=RANDOM)

    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape: (n_samples, n_features, n_classes) for multiclass
    # Average |shap| across samples and classes
    if isinstance(shap_values, list):
        # Multiple outputs (one per class)
        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0).mean(axis=0)

    # Rank features by SHAP importance
    feature_ranks = sorted(
        zip(X_yes.columns, shap_importance),
        key=lambda x: x[1], reverse=True
    )

    print(f"\n  Feature importance ranking (XGBoost, SHAP mean|value|):")
    print(f"  {'Rank':<5s} {'Feature':<30s} {'SHAP':>10s} {'Type':<12s}")
    print(f"  {'-'*60}")
    for rank, (feat, imp) in enumerate(feature_ranks, 1):
        ftype = "*** STRESS_PAL ***" if feat == 'stress_pal' else (
            "num2cat" if feat.endswith('_cat_') or feat.endswith('_cat2_') else
            "original"
        )
        marker = " <<<" if feat == 'stress_pal' else ""
        print(f"  {rank:<5d} {feat:<30s} {imp:10.6f} {ftype:<12s}{marker}")

    print("\n  Conclusion: stress_pal rank and SHAP value above.")

except ImportError:
    print("  SHAP not installed. Install with: pip install shap")

# === Save results ===
out_path = os.path.join(BASE, 'analysis', 'stress_pal_comparison.csv')
results_df.to_csv(out_path, index=False)
print(f"\n  Results saved: {out_path}")
print("Done.")
