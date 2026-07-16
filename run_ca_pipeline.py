#!/usr/bin/env python
"""Run GB+RM pipeline with per-class prior calibration (coordinate ascent).

Flow:
  1. Load cached model OOF/test predictions
  2. Per-model CA calibration (individual)
  3. GB equal blend (calibrated)
  4. GB_A CA calibration (blend-level)
  5. RM + GB_A blend
  6. Final CA calibration
  7. Generate submission

Usage:
    python run_ca_pipeline.py
    python run_ca_pipeline.py --ca-steps 500 --ca-rounds 10
"""
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.blend import (
    coordinate_ascent_calibration, apply_calibration,
    equal_blend, grid_search_nway, generate_submission,
)
from pipeline.train import get_or_train_model, _check_cache, _cache_key
from pipeline.fe import get_or_create_features

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', 'model')
OOF_DIR = os.path.join(os.path.dirname(__file__), 'oof')
SUB_DIR = os.path.join(os.path.dirname(__file__), 'submissions')

CLASS_NAMES = ['at-risk', 'fit', 'unhealthy']


def load_y():
    """Load target labels."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le = LabelEncoder()
    y = le.fit_transform(train['health_condition'].values)
    return y, le


def load_model_oof(model_name, fe_tag='AP_median', n_splits=5, params=None,
                   cache_key=None):
    """Load model OOF/test from cache by key or by params."""
    if cache_key:
        # Direct key lookup
        cached = _check_cache(cache_key)
        if cached is not None:
            oof, test, meta = cached
            print(f'  [LOAD] {model_name} cache hit (key={cache_key}): BA={meta["oof_ba"]:.5f}', flush=True)
            return oof, test, meta['oof_ba']
        raise FileNotFoundError(f'No cache with key={cache_key}')

    # Try to load from cache by params
    if params is None:
        from pipeline.train import train_xgb, train_cb, train_hgb
        defaults = {
            'XGB': dict(n_estimators=1500, learning_rate=0.05, max_depth=7,
                        min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
                        device='cuda', random_state=123, verbosity=0,
                        early_stopping_rounds=50),
            'CB': dict(iterations=1500, learning_rate=0.05, depth=5, l2_leaf_reg=1.0,
                       auto_class_weights='Balanced', task_type='GPU',
                       random_seed=123, verbose=0, early_stopping_rounds=100),
            'HGB': dict(max_iter=1000, learning_rate=0.05, max_depth=8,
                        min_samples_leaf=50, l2_regularization=1.0, max_leaf_nodes=31,
                        class_weight='balanced', random_state=123, verbose=0,
                        early_stopping=True, n_iter_no_change=50, validation_fraction=0.1),
        }
        params = defaults.get(model_name, {})

    key = _cache_key(model_name, fe_tag, n_splits, params)
    cached = _check_cache(key)
    if cached is not None:
        oof, test, meta = cached
        print(f'  [LOAD] {model_name} cache hit: BA={meta["oof_ba"]:.5f}', flush=True)
        return oof, test, meta['oof_ba']
    raise FileNotFoundError(f'No cache for {model_name}/{fe_tag}/{n_splits}f (key={key})')


def main():
    parser = argparse.ArgumentParser(description='CA calibration pipeline')
    parser.add_argument('--ca-rounds', type=int, default=5)
    parser.add_argument('--ca-steps', type=int, default=200)
    parser.add_argument('--fe-tag', type=str, default='AP_median')
    parser.add_argument('--use-tuned-xgb', action='store_true',
                        help='Use tuned XGB instead of default')
    args = parser.parse_args()

    print('=' * 60, flush=True)
    print('CA Calibration Pipeline', flush=True)
    print('=' * 60, flush=True)

    # Load target
    y, le = load_y()
    n_classes = 3
    print(f'  Train: {len(y)} rows, classes: {np.bincount(y)}', flush=True)

    # ── Step 1: Load model OOF/test ──────────────────────────────────
    print('\n[Step 1] Loading model predictions...', flush=True)

    # XGB (tuned or default) — use known cache keys
    if args.use_tuned_xgb:
        xgb_oof, xgb_test, xgb_ba = load_model_oof(
            'XGB', args.fe_tag, 5, cache_key='XGB_AP_median_5f_0f1e666d')
    else:
        # Default XGB cache was corrupted (wrong test shape), use tuned instead
        print('  [NOTE] Default XGB cache corrupted, using tuned XGB', flush=True)
        xgb_oof, xgb_test, xgb_ba = load_model_oof(
            'XGB', args.fe_tag, 5, cache_key='XGB_AP_median_5f_0f1e666d')

    cb_oof, cb_test, cb_ba = load_model_oof('CB', args.fe_tag, 5)
    hgb_oof, hgb_test, hgb_ba = load_model_oof('HGB', args.fe_tag, 5)

    # RealMLP (legacy)
    rm_oof_path = os.path.join(OOF_DIR, '_realmlp_oof.npy')
    rm_test_path = os.path.join(OOF_DIR, '_realmlp_test.npy')
    rm_oof = np.load(rm_oof_path)
    rm_test = np.load(rm_test_path)
    rm_ba = balanced_accuracy_score(y, rm_oof.argmax(1))
    print(f'  RealMLP: BA={rm_ba:.5f} (legacy)', flush=True)

    # Print baseline summary
    print(f'\n  --- Baseline (uncalibrated) ---', flush=True)
    print(f'  XGB: {xgb_ba:.5f}', flush=True)
    print(f'  CB:  {cb_ba:.5f}', flush=True)
    print(f'  HGB: {hgb_ba:.5f}', flush=True)
    print(f'  RM:  {rm_ba:.5f}', flush=True)

    # ── Step 2: Per-model CA calibration ─────────────────────────────
    print(f'\n[Step 2] Per-model CA calibration (rounds={args.ca_rounds}, steps={args.ca_steps})...',
          flush=True)

    model_oofs = {'XGB': xgb_oof, 'CB': cb_oof, 'HGB': hgb_oof, 'RM': rm_oof}
    model_tests = {'XGB': xgb_test, 'CB': cb_test, 'HGB': hgb_test, 'RM': rm_test}
    cal_scales = {}  # model -> scales
    cal_oofs = {}    # model -> calibrated OOF
    cal_tests = {}   # model -> calibrated test

    for name, oof in model_oofs.items():
        print(f'\n  --- {name} ---', flush=True)
        scales, cal_ba = coordinate_ascent_calibration(
            y, oof, n_classes=n_classes,
            n_rounds=args.ca_rounds, n_steps=args.ca_steps,
            class_names=CLASS_NAMES)
        cal_scales[name] = scales
        cal_oofs[name] = apply_calibration(oof, scales)
        cal_tests[name] = apply_calibration(model_tests[name], scales)
        delta = cal_ba - balanced_accuracy_score(y, oof.argmax(1))
        print(f'  {name}: {balanced_accuracy_score(y, oof.argmax(1)):.5f} -> {cal_ba:.5f} ({delta:+.5f})',
              flush=True)

    # ── Step 3: GB equal blend (calibrated) ──────────────────────────
    print('\n[Step 3] GB equal blend (calibrated)...', flush=True)

    gb_names = ['XGB', 'CB', 'HGB']
    gb_cal_oof = equal_blend([cal_oofs[n] for n in gb_names])
    gb_cal_test_oof = equal_blend([cal_tests[n] for n in gb_names])
    gb_cal_ba = balanced_accuracy_score(y, gb_cal_oof.argmax(1))
    print(f'  GB_A (equal, calibrated): BA={gb_cal_ba:.5f}', flush=True)

    # Compare with uncalibrated equal blend
    gb_raw_oof = equal_blend([model_oofs[n] for n in gb_names])
    gb_raw_ba = balanced_accuracy_score(y, gb_raw_oof.argmax(1))
    print(f'  GB_A (equal, raw):        BA={gb_raw_ba:.5f}', flush=True)

    # ── Step 4: GB_A CA calibration (blend-level) ────────────────────
    print('\n[Step 4] GB_A blend-level CA calibration...', flush=True)

    gb_scales, gb_ca_ba = coordinate_ascent_calibration(
        y, gb_cal_oof, n_classes=n_classes,
        n_rounds=args.ca_rounds, n_steps=args.ca_steps,
        class_names=CLASS_NAMES)
    gb_final_oof = apply_calibration(gb_cal_oof, gb_scales)
    gb_final_test = apply_calibration(gb_cal_test_oof, gb_scales)
    print(f'  GB_A after blend CA: BA={gb_ca_ba:.5f}', flush=True)

    # ── Step 5: RM + GB_A blend ──────────────────────────────────────
    print('\n[Step 5] RM + GB_A blend...', flush=True)

    # Try grid search for optimal blend weights
    blend_names = ['RM', 'GB_A']
    blend_oof_dict = {'RM': cal_oofs['RM'], 'GB_A': gb_final_oof}
    blend_test_dict = {'RM': cal_tests['RM'], 'GB_A': gb_final_test}
    weights, grid_ba = grid_search_nway(y, blend_oof_dict, blend_names, step=1)
    # weights is a dict: {'RM': float, 'GB_A': float}
    print(f'  Grid search: {weights} -> BA={grid_ba:.5f}', flush=True)

    # Also try with uncalibrated RM
    blend_oof_dict2 = {'RM': model_oofs['RM'], 'GB_A': gb_final_oof}
    weights2, grid_ba2 = grid_search_nway(y, blend_oof_dict2, blend_names, step=1)
    print(f'  Grid (uncal RM): {weights2} -> BA={grid_ba2:.5f}', flush=True)

    # Pick the best
    if grid_ba >= grid_ba2:
        best_oof_dict = blend_oof_dict
        best_test_dict = blend_test_dict
        best_weights = weights
        best_blend_ba = grid_ba
        rm_label = 'calibrated'
    else:
        best_oof_dict = blend_oof_dict2
        best_test_dict = {'RM': model_tests['RM'], 'GB_A': gb_final_test}
        best_weights = weights2
        best_blend_ba = grid_ba2
        rm_label = 'raw'

    # Apply best weights (dict with model names as keys)
    final_oof = sum(best_weights[n] * best_oof_dict[n] for n in blend_names)
    final_test = sum(best_weights[n] * best_test_dict[n] for n in blend_names)

    # ── Step 6: Final CA calibration ─────────────────────────────────
    print(f'\n[Step 6] Final CA calibration (RM={rm_label})...', flush=True)

    final_scales, final_ca_ba = coordinate_ascent_calibration(
        y, final_oof, n_classes=n_classes,
        n_rounds=args.ca_rounds, n_steps=args.ca_steps,
        class_names=CLASS_NAMES)
    final_oof = apply_calibration(final_oof, final_scales)
    final_test = apply_calibration(final_test, final_scales)

    # ── Summary ──────────────────────────────────────────────────────
    print('\n' + '=' * 60, flush=True)
    print('SUMMARY', flush=True)
    print('=' * 60, flush=True)

    # Compute all BAs for comparison
    results = {}
    for name in ['XGB', 'CB', 'HGB', 'RM']:
        raw_ba = balanced_accuracy_score(y, model_oofs[name].argmax(1))
        ca_ba = balanced_accuracy_score(y, cal_oofs[name].argmax(1))
        results[name] = {'raw': raw_ba, 'ca': ca_ba, 'delta': ca_ba - raw_ba}
        print(f'  {name}: raw={raw_ba:.5f}  CA={ca_ba:.5f}  ({ca_ba-raw_ba:+.5f})', flush=True)

    print(f'\n  GB_A (raw equal):   {gb_raw_ba:.5f}', flush=True)
    print(f'  GB_A (CA then eq):  {gb_cal_ba:.5f}', flush=True)
    print(f'  GB_A (CA blend CA): {gb_ca_ba:.5f}', flush=True)
    print(f'  Blend (grid):       {best_blend_ba:.5f}  weights={best_weights}', flush=True)
    print(f'  Final (CA):         {final_ca_ba:.5f}', flush=True)

    # ── Generate submission ──────────────────────────────────────────
    os.makedirs(SUB_DIR, exist_ok=True)
    tag = f'ca_pipeline_{"tuned" if args.use_tuned_xgb else "default"}'
    sub_path = os.path.join(SUB_DIR, f'sub_{tag}.csv')
    generate_submission(final_test, le, sub_path)
    print(f'\n  Submission: {sub_path}', flush=True)

    # Save calibration metadata
    meta_path = os.path.join(SUB_DIR, f'sub_{tag}_meta.json')
    meta = {
        'tag': tag,
        'per_model_scales': {k: v.tolist() for k, v in cal_scales.items()},
        'gb_blend_scales': gb_scales.tolist(),
        'final_scales': final_scales.tolist(),
        'blend_weights': {k: float(v) for k, v in best_weights.items()},
        'rm_label': rm_label,
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'gb_raw_ba': float(gb_raw_ba),
        'gb_cal_ba': float(gb_cal_ba),
        'gb_ca_ba': float(gb_ca_ba),
        'blend_ba': float(best_blend_ba),
        'final_ca_ba': float(final_ca_ba),
        'ca_rounds': args.ca_rounds,
        'ca_steps': args.ca_steps,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Metadata: {meta_path}', flush=True)


if __name__ == '__main__':
    main()
