#!/usr/bin/env python
"""Sequential one-parameter-at-a-time Optuna tuning for GB models.

Design: Tune ONE parameter at a time, fix it, move to next.
This is faster than grid-searching all params simultaneously and
shows which parameters actually matter.

Usage:
    python -m pipeline.optuna_tune XGB    A_median    # tune XGB only
    python -m pipeline.optuna_tune CB     A_median    # tune CB only
    python -m pipeline.optuna_tune HGB    A_median    # tune HGB only
    python -m pipeline.optuna_tune all    A_median    # tune all three
    python -m pipeline.optuna_tune XGB    A_median --trials 10 --cv 3

Args:
    model_name: XGB | CB | HGB | all
    fe_tag:     FE config tag (uses cached FE if available)
    --trials:   Optuna trials per parameter (default 15)
    --cv:       CV folds (default 3; use 5 for final eval)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Project imports
from pipeline.fe import get_or_create_features, build_tag

# ---- Paths ----
RANDOM_STATE = 123
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OPTUNA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'optuna')
os.makedirs(OPTUNA_DIR, exist_ok=True)

# ============================================================
# Default (baseline) params — mirrors pipeline/train.py
# ============================================================

DEFAULT_PARAMS = {
    'XGB': {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        # Fixed (not tuned)
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'early_stopping_rounds': 50,
    },
    'CB': {
        'iterations': 1500,
        'learning_rate': 0.05,
        'depth': 5,
        'l2_leaf_reg': 1.0,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        # Fixed (not tuned)
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 100,
    },
    'HGB': {
        'max_iter': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_samples_leaf': 50,
        'l2_regularization': 1.0,
        'max_leaf_nodes': 31,
        # Fixed (not tuned)
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'verbose': 0,
        'early_stopping': True,
        'n_iter_no_change': 50,
        'validation_fraction': 0.1,
    },
}

# ============================================================
# Parameter tuning order & search spaces
# Ordered by impact: structural → learning → regularization
# ============================================================

PARAM_ORDER = {
    'XGB': [
        {'name': 'max_depth',          'type': 'int',       'low': 3,  'high': 10},
        {'name': 'learning_rate',      'type': 'float_log', 'low': 0.01, 'high': 0.2},
        {'name': 'n_estimators',       'type': 'int_step',  'low': 200, 'high': 2000, 'step': 50},
        {'name': 'min_child_weight',   'type': 'int',       'low': 1,  'high': 100},
        {'name': 'subsample',          'type': 'float',     'low': 0.5, 'high': 1.0},
        {'name': 'colsample_bytree',   'type': 'float',     'low': 0.5, 'high': 1.0},
        {'name': 'reg_alpha',          'type': 'float_log', 'low': 1e-3, 'high': 10.0},
        {'name': 'reg_lambda',         'type': 'float_log', 'low': 1e-3, 'high': 10.0},
    ],
    'CB': [
        {'name': 'depth',              'type': 'int',       'low': 3,  'high': 10},
        {'name': 'learning_rate',      'type': 'float_log', 'low': 0.01, 'high': 0.3},
        {'name': 'iterations',         'type': 'int_step',  'low': 200, 'high': 2000, 'step': 50},
        {'name': 'l2_leaf_reg',        'type': 'float_log', 'low': 0.1, 'high': 20.0},
        {'name': 'random_strength',    'type': 'float_log', 'low': 0.1, 'high': 10.0},
        {'name': 'bagging_temperature','type': 'float',     'low': 0.0, 'high': 10.0},
    ],
    'HGB': [
        {'name': 'max_depth',          'type': 'int_null',  'low': 3,  'high': 16},  # 16 → None
        {'name': 'learning_rate',      'type': 'float_log', 'low': 0.01, 'high': 0.3},
        {'name': 'max_iter',           'type': 'int_step',  'low': 200, 'high': 2000, 'step': 50},
        {'name': 'min_samples_leaf',   'type': 'int',       'low': 10, 'high': 200},
        {'name': 'l2_regularization',  'type': 'float_log', 'low': 0.01, 'high': 10.0},
        {'name': 'max_leaf_nodes',     'type': 'int',       'low': 15, 'high': 127},
    ],
}


def _suggest(trial, param_def):
    """Suggest a value for a single parameter."""
    pname = param_def['name']
    ptype = param_def['type']
    if ptype == 'int':
        return trial.suggest_int(pname, param_def['low'], param_def['high'])
    elif ptype == 'int_step':
        return trial.suggest_int(pname, param_def['low'], param_def['high'],
                                 step=param_def.get('step', 1))
    elif ptype == 'float':
        return trial.suggest_float(pname, param_def['low'], param_def['high'])
    elif ptype == 'float_log':
        return trial.suggest_float(pname, param_def['low'], param_def['high'], log=True)
    elif ptype == 'int_null':
        v = trial.suggest_int(pname, param_def['low'], param_def['high'])
        return None if v == param_def['high'] else v
    raise ValueError(f'Unknown param type: {ptype}')


# ============================================================
# Quick CV evaluation per model
# ============================================================

def _eval_xgb(params, X, y, n_splits):
    """3-fold CV for XGB with given params. Returns mean BA."""
    from xgboost import XGBClassifier
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for trn, val in skf.split(X_arr, y):
        sw = compute_sample_weight('balanced', y[trn])
        m = XGBClassifier(**params)
        m.fit(X_arr[trn], y[trn], sample_weight=sw,
              eval_set=[(X_arr[val], y[val])], verbose=False)
        pred = m.predict_proba(X_arr[val])
        scores.append(balanced_accuracy_score(y[val], pred.argmax(1)))
    return float(np.mean(scores))


def _eval_cb(params, X, y, cat_indices, n_splits):
    """3-fold CV for CatBoost with given params. Returns mean BA."""
    from catboost import CatBoostClassifier, Pool
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for trn, val in skf.split(X, y):
        tr_pool = Pool(X.iloc[trn], y[trn], cat_features=cat_indices)
        va_pool = Pool(X.iloc[val], y[val], cat_features=cat_indices)
        m = CatBoostClassifier(**params)
        m.fit(tr_pool, eval_set=va_pool, verbose=0)
        pred = m.predict_proba(va_pool)
        scores.append(balanced_accuracy_score(y[val], pred.argmax(1)))
    return float(np.mean(scores))


def _eval_hgb(params, X, y, n_splits):
    """3-fold CV for HistGradientBoosting with given params. Returns mean BA."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for trn, val in skf.split(X_arr, y):
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_arr[trn], y[trn])
        pred = m.predict_proba(X_arr[val])
        scores.append(balanced_accuracy_score(y[val], pred.argmax(1)))
    return float(np.mean(scores))


def eval_model(model_name, params, X, y, cat_indices, n_splits=3):
    """Run quick n-fold CV and return mean BA."""
    if model_name == 'XGB':
        return _eval_xgb(params, X, y, n_splits)
    elif model_name == 'CB':
        return _eval_cb(params, X, y, cat_indices, n_splits)
    elif model_name == 'HGB':
        return _eval_hgb(params, X, y, n_splits)
    raise ValueError(f'Unknown model: {model_name}')


# ============================================================
# Sequential one-param tuning
# ============================================================

def tune_model(model_name, fe_tag, n_trials=15, n_splits=3, sample_size=None):
    """Tune ONE parameter at a time, fixing best before moving on.

    Args:
        model_name: 'XGB' | 'CB' | 'HGB'
        fe_tag: FE config tag (e.g. 'A_median', 'ASIP_median')
        n_trials: Optuna trials per parameter
        n_splits: CV folds for evaluation (default 3 for speed)
        sample_size: If set, use stratified subsample for tuning phase only.
                     Final eval always uses full data. Default None → full data.

    Returns:
        (best_params, results_dict)
    """
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le = LabelEncoder()
    y = le.fit_transform(train_df['health_condition'].values)

    # Get FE data
    fe_config = _resolve_fe_config(fe_tag)
    X, X_test, cat_indices, feat_names, actual_tag = get_or_create_features(fe_config, y)
    print(f'Features: {actual_tag} ({len(feat_names)}d)', flush=True)

    # ---- Subsample for quick tuning ----
    if sample_size and sample_size < len(y):
        from sklearn.model_selection import train_test_split
        X_tune, _, y_tune, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=RANDOM_STATE)
        if hasattr(X_tune, 'reset_index'):
            X_tune = X_tune.reset_index(drop=True)
        print(f'Tuning subsample: {len(y_tune)}/{len(y)} rows ({100*len(y_tune)/len(y):.0f}%)', flush=True)
    else:
        X_tune, y_tune = X, y
        sample_size = None  # mark as full

    # Baseline params
    base_params = DEFAULT_PARAMS[model_name].copy()
    param_order = PARAM_ORDER[model_name]

    # ---- 1. Baseline (full data, 5f) ----
    print(f'\n{"="*60}')
    print(f'BASELINE: {model_name} on {actual_tag} (5f CV, full data)')
    print(f'{"="*60}')
    t0 = time.time()
    baseline_ba = eval_model(model_name, base_params, X, y, cat_indices, n_splits=5)
    print(f'  {model_name} baseline: BA={baseline_ba:.5f} ({time.time()-t0:.0f}s)', flush=True)

    # ---- 2. Sequential tuning (on subsample) ----
    best_params = base_params.copy()
    param_results = []
    current_ba = baseline_ba  # for reference only; actual tuning scores on subsample

    for i, param_def in enumerate(param_order, 1):
        pname = param_def['name']
        ptype = param_def['type']
        ss_tag = f' ({sample_size} rows)' if sample_size else ''
        print(f'\n  [{i}/{len(param_order)}] Tuning {pname} ({ptype}){ss_tag} ...', flush=True)

        t_param = time.time()

        def objective(trial):
            # Suggest value for THIS one parameter only
            value = _suggest(trial, param_def)

            # Build params: current best + this trial's value
            trial_params = best_params.copy()
            trial_params[pname] = value

            # Handle early_stopping: disable when tuning n_estimators/iterations/max_iter
            # so the exact value matters; otherwise rely on early stopping
            if model_name == 'XGB':
                if pname == 'n_estimators':
                    trial_params.pop('early_stopping_rounds', None)
            elif model_name == 'CB':
                if pname == 'iterations':
                    trial_params.pop('early_stopping_rounds', None)
            elif model_name == 'HGB':
                if pname == 'max_iter':
                    trial_params['early_stopping'] = False

            ba = eval_model(model_name, trial_params, X_tune, y_tune, cat_indices, n_splits)
            return ba

        # Run Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_value = study.best_params[pname]
        old_value = best_params[pname]
        best_params[pname] = best_value
        new_ba = study.best_value

        delta = new_ba - current_ba if current_ba is not None else 0
        sign = '+' if delta >= 0 else ''
        ss_note = '(subsample)' if sample_size else ''
        print(f'    {pname}: {old_value} → {best_value}  |  BA={new_ba:.5f} ({sign}{delta:.5f}) {ss_note}  '
              f'({time.time()-t_param:.0f}s)', flush=True)

        param_results.append({
            'param': pname,
            'old': _serialize(old_value),
            'new': _serialize(best_value),
            'ba': round(new_ba, 5),
            'delta': round(delta, 5),
        })
        current_ba = new_ba

    # ---- 3. Final 5-fold evaluation ----
    print(f'\n{"="*60}')
    print(f'FINAL: {model_name} tuned params (5f CV)')
    print(f'{"="*60}')
    t0 = time.time()
    final_ba = eval_model(model_name, best_params, X, y, cat_indices, n_splits=5)
    delta_final = final_ba - baseline_ba
    sign = '+' if delta_final >= 0 else ''
    print(f'  Baseline: {baseline_ba:.5f}')
    print(f'  Tuned:    {final_ba:.5f} ({sign}{delta_final:.5f})')
    print(f'  Params:   {_fmt_params(best_params, param_order)}')
    print(f'  ({time.time()-t0:.0f}s)', flush=True)

    # ---- 4. Save ----
    results = {
        'model': model_name,
        'fe_tag': actual_tag,
        'n_trials': n_trials,
        'cv_folds_tune': n_splits,
        'cv_folds_final': 5,
        'baseline_ba': round(baseline_ba, 5),
        'final_ba': round(final_ba, 5),
        'delta': round(delta_final, 5),
        'baseline_params': {k: _serialize(v) for k, v in base_params.items()
                            if k in [p['name'] for p in param_order]},
        'tuned_params': {k: _serialize(v) for k, v in best_params.items()
                         if k in [p['name'] for p in param_order]},
        'param_results': param_results,
    }
    out_path = os.path.join(OPTUNA_DIR, f'{model_name}_{actual_tag}_tuned.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}', flush=True)

    return best_params, results


# ============================================================
# Helpers
# ============================================================

def _resolve_fe_config(fe_tag):
    """Convert tag string to FE config dict.
    
    Supports both raw tags and named presets.
    """
    if fe_tag.startswith('{'):
        import yaml
        return yaml.safe_load(fe_tag)
    
    # Known presets (mirrors pipeline/fe.py build_tag logic)
    # A_median = num2cat + stress_pal, median fill
    if fe_tag == 'A_median':
        return {'num2cat': True, 'stress_pal': True, 'num_fill': 'median', 'te': False}
    # base13 = pure baseline, no extra features
    if fe_tag == 'base13':
        return {'stress_pal': False, 'num_fill': 'median', 'te': False}
    
    # Otherwise, try as a raw tag name (already cached)
    # We need to read the meta to get the config back
    CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'fe')
    meta_path = os.path.join(CACHE_DIR, f'{fe_tag}_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get('config', {})
    
    # Fallback: try to reconstruct from tag
    cfg = {
        'num2cat': 'A' in fe_tag,
        'stress_pal': 'P' in fe_tag or 'A' in fe_tag,
        'sleep_cat': 'S' in fe_tag,
        'sleep_interact_with': ['stress_pal', 'stress_level', 'physical_activity_level'] if 'I' in fe_tag else None,
        'stress_bin': 'H' in fe_tag,
        'key_rule3': 'K' in fe_tag,
        'kbins': 'B' in fe_tag,
        'heart_bmi': 'C' in fe_tag,
        'te': 'D' in fe_tag,
        'num_fill': 'median',
    }
    return cfg


def _serialize(v):
    """Convert value to JSON-serializable form."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _fmt_params(params, param_order):
    """Format tuned params for display."""
    tuned_keys = {p['name'] for p in param_order}
    parts = [f'{k}={_serialize(v)}' for k, v in params.items() if k in tuned_keys]
    return '{' + ', '.join(parts) + '}'


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Sequential GB model tuning')
    parser.add_argument('model', help='XGB | CB | HGB | all')
    parser.add_argument('fe_tag', help='FE config tag (e.g. A_median)')
    parser.add_argument('--trials', type=int, default=15, help='Optuna trials per param (default 15)')
    parser.add_argument('--cv', type=int, default=3, help='CV folds for tuning (default 3)')
    parser.add_argument('--sample', type=int, default=100000, help='Subsample rows for tuning (default 100K; 0=full)')
    args = parser.parse_args()

    models = ['XGB', 'CB', 'HGB'] if args.model == 'all' else [args.model]
    sample_size = args.sample if args.sample > 0 else None

    for m in models:
        print(f'\n{"#"*60}')
        print(f'# Tuning {m}')
        print(f'{"#"*60}')
        try:
            best, res = tune_model(m, args.fe_tag, n_trials=args.trials,
                                   n_splits=args.cv, sample_size=sample_size)
        except Exception as e:
            print(f'\nERROR tuning {m}: {e}', flush=True)
            import traceback
            traceback.print_exc()

    print(f'\nDone. Results in: {OPTUNA_DIR}')
    print(f'Models tuned: {len(models)}')


if __name__ == '__main__':
    main()
