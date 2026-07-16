"""Replicate reference notebook: 0-95043-lgbm-xgb-ens-0-95021-solo-lgbm-no-tune.ipynb

Exact copy of notebook logic adapted for local paths and CUDA.
Reference: LGBM(0.85) + XGB(0.15) soft voting → LB=0.95043

Key differences from our pipeline:
1. Ordinal encoding (health-ordered, not alphabetical LabelEncoder)
2. 13 domain features (healthy_score, sleep_score, calorie_score, etc.)
3. LightGBM as primary model
4. Per-model different FE (LGBM gets 26 features, XGB gets 22)
5. Default params (no tuning), class_weight='balanced'
6. 5-fold CV, seed=42
7. No TE, no num2cat, no GP features
8. No NaN imputation (models handle natively)
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin

import lightgbm as lgb
import xgboost as xgb

# ============================================================
# Config
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OOF_DIR = os.path.join(os.path.dirname(__file__), 'oof')
SUB_DIR = os.path.join(os.path.dirname(__file__), 'submissions')

# ============================================================
# Load data
# ============================================================
print('Loading data...', flush=True)
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

y = train_df['health_condition']
X_train = train_df.drop(['id', 'health_condition'], axis=1)
X_test = test_df.drop(['id'], axis=1)

le = LabelEncoder()
y_train = le.fit_transform(train_df['health_condition'])
# le.classes_ = ['at-risk', 'fit', 'unhealthy'] → 0, 1, 2

print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print(f'Classes: {le.classes_} → {list(range(len(le.classes_)))}')

# ============================================================
# Ordinal encoding (health-ordered, NOT alphabetical)
# ============================================================
cat_cols = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

# First cast to category (for baseline test)
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Ordinal maps: higher value = healthier/better
diet_map = {'veg': 0, 'non-veg': 1, 'balanced': 2}
stress_map = {'high': 0, 'medium': 1, 'low': 2}
sleep_map = {'poor': 0, 'average': 1, 'good': 2}
activity_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
smoking_map = {'yes': 0, 'occasional': 1, 'no': 2}
gender_map = {'male': 0, 'female': 1, 'other': 2}

maps = [diet_map, stress_map, sleep_map, activity_map, smoking_map, gender_map]

for col, m in zip(cat_cols, maps):
    X_train[col] = X_train[col].map(m)
    X_test[col] = X_test[col].map(m)

# Cast to float (reference notebook does this)
for col in cat_cols:
    X_train[col] = X_train[col].astype("float")
    X_test[col] = X_test[col].astype("float")

# ============================================================
# Feature engineering
# ============================================================
def engineer_features(df):
    """LGBM version: 13 new features (26 total)."""
    res = df.copy()
    
    health_cols = ['diet_type', 'stress_level', 'sleep_quality',
                   'physical_activity_level', 'smoking_alcohol']
    res['healthy_score'] = res[health_cols].mean(axis=1)

    def gauss(x):
        return np.exp(-(x - 8) ** 2 / 4.5)
    
    sleep_score_map = {0: 0.5, 1: 1.0, 2: 1.5}
    res['sleep_score'] = gauss(res['sleep_duration']) * res['sleep_quality'].map(sleep_score_map)

    res['calorie_score'] = res['calorie_expenditure'] / res['bmi']
    res['steps_score'] = res['step_count'] / res['bmi']

    activity_map = {0: 0.5, 1: 1.0, 2: 1.5}
    res['sport_score'] = np.log1p(
        res['step_count'] * res['exercise_duration'] * res['physical_activity_level'].map(activity_map)
    )

    res['speed_score'] = res['step_count'] / res['exercise_duration']
    res['hydration_score'] = res['water_intake'] / res['calorie_expenditure']
    res['depression_score'] = res[['smoking_alcohol', 'stress_level']].mean(axis=1)

    res['no_exercises'] = np.where(res['exercise_duration'].isna(), np.nan,
                                   (res['exercise_duration'] == 0).astype(int))
    res['bradycardia'] = np.where(res['heart_rate'].isna(), np.nan,
                                  (res['heart_rate'] < 60).astype(int))
    res['tachycardia'] = np.where(res['heart_rate'].isna(), np.nan,
                                  (res['heart_rate'] > 100).astype(int))
    res['underweight'] = np.where(res['bmi'].isna(), np.nan,
                                  (res['bmi'] < 18.5).astype(int))
    res['overweight'] = np.where(res['bmi'].isna(), np.nan,
                                 (res['bmi'] > 25).astype(int))
    
    return res


def engineer_features_XGB(df):
    """XGB version: 9 new features (22 total, omits speed/hydration/depression/no_exercises)."""
    res = df.copy()
    
    health_cols = ['diet_type', 'stress_level', 'sleep_quality',
                   'physical_activity_level', 'smoking_alcohol']
    res['healthy_score'] = res[health_cols].mean(axis=1)

    def gauss(x):
        return np.exp(-(x - 8) ** 2 / 4.5)
    
    sleep_score_map = {0: 0.5, 1: 1.0, 2: 1.5}
    res['sleep_score'] = gauss(res['sleep_duration']) * res['sleep_quality'].map(sleep_score_map)

    res['calorie_score'] = res['calorie_expenditure'] / res['bmi']
    res['steps_score'] = res['step_count'] / res['bmi']

    activity_map = {0: 0.5, 1: 1.0, 2: 1.5}
    res['sport_score'] = np.log1p(
        res['step_count'] * res['exercise_duration'] * res['physical_activity_level'].map(activity_map)
    )

    res['bradycardia'] = np.where(res['heart_rate'].isna(), np.nan,
                                  (res['heart_rate'] < 60).astype(int))
    res['tachycardia'] = np.where(res['heart_rate'].isna(), np.nan,
                                  (res['heart_rate'] > 100).astype(int))
    res['underweight'] = np.where(res['bmi'].isna(), np.nan,
                                  (res['bmi'] < 18.5).astype(int))
    res['overweight'] = np.where(res['bmi'].isna(), np.nan,
                                 (res['bmi'] > 25).astype(int))
    
    return res


X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

X_train_lgb = engineer_features(X_train_copy)
X_test_lgb = engineer_features(X_test_copy)

X_train_xgb = engineer_features_XGB(X_train_copy)
X_test_xgb = engineer_features_XGB(X_test_copy)

print(f'LGBM features: {X_train_lgb.shape[1]}d')
print(f'XGB features: {X_train_xgb.shape[1]}d')
print(f'LGBM new cols: {[c for c in X_train_lgb.columns if c not in X_train_copy.columns]}')
print(f'XGB new cols: {[c for c in X_train_xgb.columns if c not in X_train_copy.columns]}')

# ============================================================
# Cross-validation (5-fold, matching reference)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- LGBM CV ---
print('\n=== LGBM Cross-Validation ===', flush=True)
lgbm_model = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
lgbm_scores = cross_val_score(lgbm_model, X_train_lgb, y_train, cv=skf,
                               scoring='balanced_accuracy')
print(f'LGBM CV: {np.mean(lgbm_scores):.5f} (± {np.std(lgbm_scores):.4f})')
print(f'  Per-fold: {[f"{s:.5f}" for s in lgbm_scores]}')

# --- XGB CV ---
print('\n=== XGB Cross-Validation ===', flush=True)

class XGBBalancedFloat(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs, tree_method="hist")
        
    def fit(self, X, y):
        weights = compute_sample_weight(class_weight='balanced', y=y)
        self.model.fit(X, y, sample_weight=weights, verbose=False)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

xgb_model = XGBBalancedFloat(random_state=42, n_jobs=-1)
xgb_scores = cross_val_score(xgb_model, X_train_xgb, y_train, cv=skf,
                              scoring='balanced_accuracy')
print(f'XGB CV: {np.mean(xgb_scores):.5f} (± {np.std(xgb_scores):.4f})')
print(f'  Per-fold: {[f"{s:.5f}" for s in xgb_scores]}')

# ============================================================
# OOF predictions (for blending with our pipeline)
# ============================================================
print('\n=== OOF Predictions ===', flush=True)

# LGBM OOF
lgbm_oof = np.zeros((len(X_train_lgb), 3))
lgbm_test = np.zeros((len(X_test_lgb), 3))

t0 = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_lgb, y_train), 1):
    m = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
    m.fit(X_train_lgb.iloc[trn_idx], y_train[trn_idx])
    lgbm_oof[val_idx] = m.predict_proba(X_train_lgb.iloc[val_idx])
    lgbm_test += m.predict_proba(X_test_lgb) / 5
    ba = balanced_accuracy_score(y_train[val_idx], lgbm_oof[val_idx].argmax(1))
    print(f'  LGBM fold {fold}/5: BA={ba:.5f}', flush=True)

lgbm_oof_ba = balanced_accuracy_score(y_train, lgbm_oof.argmax(1))
print(f'  LGBM OOF BA: {lgbm_oof_ba:.5f} ({time.time()-t0:.0f}s)')

# XGB OOF
xgb_oof = np.zeros((len(X_train_xgb), 3))
xgb_test = np.zeros((len(X_test_xgb), 3))

t0 = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_xgb, y_train), 1):
    weights = compute_sample_weight('balanced', y_train[trn_idx])
    m = xgb.XGBClassifier(random_state=42, n_jobs=-1, tree_method="hist")
    m.fit(X_train_xgb.iloc[trn_idx], y_train[trn_idx],
          sample_weight=weights, verbose=False)
    xgb_oof[val_idx] = m.predict_proba(X_train_xgb.iloc[val_idx])
    xgb_test += m.predict_proba(X_test_xgb) / 5
    ba = balanced_accuracy_score(y_train[val_idx], xgb_oof[val_idx].argmax(1))
    print(f'  XGB fold {fold}/5: BA={ba:.5f}', flush=True)

xgb_oof_ba = balanced_accuracy_score(y_train, xgb_oof.argmax(1))
print(f'  XGB OOF BA: {xgb_oof_ba:.5f} ({time.time()-t0:.0f}s)')

# ============================================================
# Ensemble (matching reference: LGBM 0.85 + XGB 0.15)
# ============================================================
print('\n=== Ensemble ===', flush=True)

# Reference uses XGB feature set for BOTH models in final ensemble
# Full-train fit (no OOF stacking)
lgbm_final = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
lgbm_final.fit(X_train_xgb, y_train)

weights_final = compute_sample_weight(class_weight='balanced', y=y_train)
xgb_final = xgb.XGBClassifier(random_state=42, n_jobs=-1, tree_method="hist")
xgb_final.fit(X_train_xgb, y_train, sample_weight=weights_final, verbose=False)

preds_lgbm = lgbm_final.predict_proba(X_test_xgb)
preds_xgb = xgb_final.predict_proba(X_test_xgb)

ensemble_preds_proba = preds_lgbm * 0.85 + preds_xgb * 0.15
final_predictions = np.argmax(ensemble_preds_proba, axis=1)
final_test_preds = le.inverse_transform(final_predictions)

# Submission
os.makedirs(SUB_DIR, exist_ok=True)
submission = pd.DataFrame({
    'id': test_df['id'],
    'health_condition': final_test_preds
})
sub_path = os.path.join(SUB_DIR, 'sub_ref_gb_085_015.csv')
submission.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')
print(f'Value counts: {submission.health_condition.value_counts().to_dict()}')

# ============================================================
# Save OOF for blending with our pipeline
# ============================================================
os.makedirs(OOF_DIR, exist_ok=True)

np.save(os.path.join(OOF_DIR, '_ref_gb_lgbm_oof.npy'), lgbm_oof)
np.save(os.path.join(OOF_DIR, '_ref_gb_lgbm_test.npy'), lgbm_test)
np.save(os.path.join(OOF_DIR, '_ref_gb_xgb_oof.npy'), xgb_oof)
np.save(os.path.join(OOF_DIR, '_ref_gb_xgb_test.npy'), xgb_test)

# Also save ensemble OOF (0.85/0.15 blend on OOF predictions)
# Note: LGBM OOF uses LGBM features, XGB OOF uses XGB features
# For OOF blend, we use the same 0.85/0.15 weights
ens_oof = lgbm_oof * 0.85 + xgb_oof * 0.15  # BUG — will fix below
ens_oof = lgbm_oof * 0.85 + xgb_oof * 0.15
ens_test = lgbm_test * 0.85 + xgb_test * 0.15
ens_oof_ba = balanced_accuracy_score(y_train, ens_oof.argmax(1))
print(f'\nEnsemble OOF BA (0.85/0.15): {ens_oof_ba:.5f}')

np.save(os.path.join(OOF_DIR, '_ref_gb_ens_oof.npy'), ens_oof)
np.save(os.path.join(OOF_DIR, '_ref_gb_ens_test.npy'), ens_test)

# ============================================================
# LB prediction (v7 formula)
# ============================================================
health = max(lgbm_oof_ba, xgb_oof_ba) - ens_oof_ba
predicted_lb = ens_oof_ba + 0.00049 + 1.227 * health
print(f'\nLB Prediction (v7):')
print(f'  health = max({lgbm_oof_ba:.5f}, {xgb_oof_ba:.5f}) - {ens_oof_ba:.5f} = {health:.5f}')
print(f'  predicted_LB = {ens_oof_ba:.5f} + 0.00049 + 1.227 * {health:.5f} = {predicted_lb:.5f}')
print(f'  Reference LB = 0.95043')

print('\nDone!')
