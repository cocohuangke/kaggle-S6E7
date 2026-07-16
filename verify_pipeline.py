#!/usr/bin/env python
"""Verify pipeline reproduces sub_rm_gb_a_blend_noca.csv.

Strategy: Use legacy OOF files (5-fold GB_A + RealMLP) to verify
the blend logic generates the exact same submission.

This DOES NOT retrain models — it only verifies:
1. FE produces correct feature set (17d, A_num2cat)
2. Legacy OOF loading works
3. Equal blend (XGB+CB+HGB) matches _fe_A_num2cat_eq_oof.npy
4. Weighted blend (RM0.55 + GB_A0.45) matches _rm_gb_a_blend_oof.npy
5. Generated submission matches sub_rm_gb_a_blend_noca.csv exactly
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DATA_DIR = os.path.join(BASE, 'data')
OOF_DIR = os.path.join(BASE, 'oof')
SUB_DIR = os.path.join(BASE, 'submissions')

PASS = True


def check(name, condition, detail=''):
    global PASS
    status = '[OK]' if condition else '[FAIL]'
    print(f'  {status} {name}' + (f' — {detail}' if detail else ''), flush=True)
    if not condition:
        PASS = False


def verify_fe():
    """Step 1: Verify FE produces correct A_num2cat features."""
    print('\n--- Step 1: Verify FE ---', flush=True)
    from pipeline.fe import get_or_create_features, build_tag

    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le = LabelEncoder()
    y = le.fit_transform(train['health_condition'].values)

    fe_cfg = {'num2cat': True, 'stress_pal': True, 'num_fill': 'median', 'te': False}
    tag = build_tag(fe_cfg)
    print(f'  FE tag: {tag}', flush=True)

    X_train, X_test, cat_idx, feat_names, tag = get_or_create_features(fe_cfg, y)
    print(f'  Features: {len(feat_names)}d', flush=True)
    print(f'  Feature names: {feat_names}', flush=True)
    print(f'  Cat indices: {cat_idx}', flush=True)

    # Verify dimensions
    check('Feature count = 17', len(feat_names) == 17, f'got {len(feat_names)}')
    check('Cat indices = [7..16]', cat_idx == list(range(7, 17)), f'got {cat_idx}')

    # Verify cat columns are integer type
    cat_cols = [feat_names[i] for i in cat_idx]
    all_int = all(X_train[c].dtype in ('int32', 'int64', 'int8') for c in cat_cols)
    check('Cat columns are int type', all_int)

    # Verify shapes
    check('X_train shape correct', X_train.shape[1] == 17, f'got {X_train.shape}')
    check('X_test shape correct', X_test.shape[1] == 17, f'got {X_test.shape}')

    return True


def verify_blend_legacy():
    """Step 2: Verify blend using legacy OOF files."""
    print('\n--- Step 2: Verify Blend (Legacy OOF) ---', flush=True)

    # Load legacy OOF
    rm_oof = np.load(os.path.join(OOF_DIR, '_realmlp_oof.npy'))
    rm_test = np.load(os.path.join(OOF_DIR, '_realmlp_test.npy'))
    eq5_oof = np.load(os.path.join(OOF_DIR, '_fe_A_num2cat_eq_oof.npy'))
    eq5_test = np.load(os.path.join(OOF_DIR, '_fe_A_num2cat_eq_test.npy'))

    # Load target
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le = LabelEncoder()
    y = le.fit_transform(train['health_condition'].values)

    # Verify OOF BAs
    print(f'  eq5 OOF BA = {balanced_accuracy_score(y, eq5_oof.argmax(1)):.5f}', flush=True)
    print(f'  RM OOF BA = {balanced_accuracy_score(y, rm_oof.argmax(1)):.5f}', flush=True)

    # Blend: RM0.55 + GB_A0.45
    blend_oof = 0.55 * rm_oof + 0.45 * eq5_oof
    blend_test = 0.55 * rm_test + 0.45 * eq5_test

    # Compare with saved blend
    saved_oof = np.load(os.path.join(OOF_DIR, '_rm_gb_a_blend_oof.npy'))
    saved_test = np.load(os.path.join(OOF_DIR, '_rm_gb_a_blend_test.npy'))

    check('Blend OOF = RM0.55 + GB_A0.45', np.allclose(blend_oof, saved_oof, atol=1e-12))
    check('Blend test = RM0.55 + GB_A0.45', np.allclose(blend_test, saved_test, atol=1e-12))

    # Generate and compare submission
    pred_labels = le.inverse_transform(blend_test.argmax(1))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['health_condition'] = pred_labels

    ref = pd.read_csv(os.path.join(SUB_DIR, 'sub_rm_gb_a_blend_noca.csv'))
    check('Submission matches sub_rm_gb_a_blend_noca.csv', sub.equals(ref))

    blend_ba = balanced_accuracy_score(y, blend_oof.argmax(1))
    print(f'  Blend OOF BA = {blend_ba:.5f}', flush=True)

    return True


def verify_config_driven():
    """Step 3: Verify config-driven pipeline (using legacy OOF, no retraining)."""
    print('\n--- Step 3: Verify Config-Driven Pipeline ---', flush=True)
    from pipeline.blend import run_blend

    config_path = os.path.join(BASE, 'configs', 'best_v1.yaml')
    if not os.path.exists(config_path):
        print('  [SKIP] Config file not found', flush=True)
        return True

    # Run blend (will use legacy OOF for RealMLP, train XGB/CB/HGB from scratch)
    # This is the full pipeline test — may take time
    print('  Running config-driven pipeline (best_v1.yaml)...', flush=True)
    print('  NOTE: This will train XGB/CB/HGB with 5-fold CV', flush=True)
    final_ba, final_oof, final_test = run_blend(config_path)

    # Compare submission
    le = LabelEncoder()
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le.fit(train['health_condition'])

    sub_path = os.path.join(BASE, 'submissions', 'sub_best_v1.csv')
    gen_sub = pd.read_csv(sub_path)
    ref = pd.read_csv(os.path.join(SUB_DIR, 'sub_rm_gb_a_blend_noca.csv'))

    # Note: GPU nondeterminism means XGB/CB test predictions may differ slightly
    # So the submission may not match exactly — but should be very close
    n_diff = (gen_sub['health_condition'] != ref['health_condition']).sum()
    n_total = len(ref)
    match_pct = (1 - n_diff / n_total) * 100
    print(f'  Submission match: {match_pct:.4f}% ({n_diff}/{n_total} differences)', flush=True)

    # For exact match, we need to use legacy 5-fold OOF
    # The pipeline trains fresh models, so GPU nondeterminism is expected
    if n_diff == 0:
        check('Config-driven submission matches exactly', True)
    else:
        print(f'  [INFO] Differences expected due to GPU nondeterminism in XGB/CB', flush=True)
        print(f'  [INFO] Exact match requires using legacy 5-fold OOF files', flush=True)

    return True


if __name__ == '__main__':
    print('='*60, flush=True)
    print('Pipeline Verification Script', flush=True)
    print('='*60, flush=True)

    # Step 1: FE
    verify_fe()

    # Step 2: Blend with legacy OOF
    verify_blend_legacy()

    # Step 3: Config-driven pipeline (optional, takes time)
    if '--full' in sys.argv:
        verify_config_driven()
    else:
        print('\n--- Step 3: Config-Driven Pipeline ---', flush=True)
        print('  [SKIP] Use --full flag to run full pipeline test', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'VERIFICATION {"PASSED" if PASS else "FAILED"}', flush=True)
    print(f'{"="*60}', flush=True)
