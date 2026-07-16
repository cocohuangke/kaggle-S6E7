"""Run all 7 drop-feature experiments (XGB+CB only, skip slow HGB)."""
import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.fe import get_or_create_features, build_tag
from pipeline.train import get_or_train_model

DATA_DIR = 'data'

# Load target
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
le = LabelEncoder()
y = le.fit_transform(train['health_condition'].values)

# Baseline: 13d (base_median)
fe_base = {'num2cat': False, 'stress_pal': False, 'num_fill': 'median', 'te': False}
X_train_base, X_test_base, cat_idx_base, feat_names_base, tag_base = get_or_create_features(fe_base, y)
print(f'Baseline: {tag_base}, {len(feat_names_base)}d, features: {feat_names_base}')

# Train baseline XGB+CB
oof_xgb_base, test_xgb_base, ba_xgb_base = get_or_train_model('XGB', tag_base, X_train_base, y, X_test_base, cat_idx_base, n_splits=5)
oof_cb_base, test_cb_base, ba_cb_base = get_or_train_model('CB', tag_base, X_train_base, y, X_test_base, cat_idx_base, n_splits=5)
eq_base = (oof_xgb_base + oof_cb_base) / 2
ba_base = balanced_accuracy_score(y, eq_base.argmax(1))
print(f'Baseline XGB+CB equal: BA={ba_base:.5f} (XGB={ba_xgb_base:.5f}, CB={ba_cb_base:.5f})')

# Drop experiments
drop_features = [
    'heart_rate',
    'calorie_expenditure',
    'water_intake',
    'gender',
    'step_count',
    'diet_type',
    'exercise_duration',
]

results = [('baseline (13d)', ba_base, 0.0, 'BASELINE')]
for feat in drop_features:
    fe_cfg = {'num2cat': False, 'stress_pal': False, 'num_fill': 'median', 'te': False, 'drop_features': [feat]}
    X_tr, X_te, cat_idx, feat_names, tag = get_or_create_features(fe_cfg, y)
    print(f'\n--- Drop {feat}: {tag}, {len(feat_names)}d ---')
    
    oof_xgb, test_xgb, ba_xgb = get_or_train_model('XGB', tag, X_tr, y, X_te, cat_idx, n_splits=5)
    oof_cb, test_cb, ba_cb = get_or_train_model('CB', tag, X_tr, y, X_te, cat_idx, n_splits=5)
    eq = (oof_xgb + oof_cb) / 2
    ba = balanced_accuracy_score(y, eq.argmax(1))
    delta = ba - ba_base
    verdict = 'DELETE' if delta >= -0.0002 else 'KEEP'
    results.append((f'-{feat} ({len(feat_names)}d)', ba, delta, verdict))
    print(f'  XGB={ba_xgb:.5f}, CB={ba_cb:.5f}, eq={ba:.5f}, delta={delta:+.5f} -> {verdict}')

print('\n' + '='*70)
print('SUMMARY: Feature Deletion (XGB+CB equal blend)')
print(f'Baseline: BA={ba_base:.5f}')
print('='*70)
print(f'{"Experiment":<30} {"BA":>8} {"Delta":>8} {"Verdict":>8}')
print('-'*70)
for name, ba, delta, verdict in results:
    print(f'{name:<30} {ba:>8.5f} {delta:>+8.5f} {verdict:>8}')
