"""Model training pipeline for S6E7.

Each model trained independently with cache:
  cache/model/{model}_{fe_tag}_{nfolds}f_oof.npy
  cache/model/{model}_{fe_tag}_{nfolds}f_test.npy

If cache exists with same (model, fe_tag, nfolds, params_hash), skip training.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

RANDOM_STATE = 123
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'model')
OOF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'oof')


def _params_hash(params):
    """Short hash of params dict for cache key."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _decode_fe_config(fe_tag):
    """Decode fe_tag into RealMLP fe_config dict.
    
    Tags:
      baseline      -> original notebook FE (stress_pal, no sleep)
      sleep         -> + sleep_cat + sleep_interact_with [stress_pal, stress_level, pal]
      realmlp_plain -> original _realmlp: no GP, no extra cat features, no water_intake round
      realmlp_ref   -> ref notebook: GP features, stress_pal=False, epochs=3, seed=63
    
    All tags now use format='realmlp' (unified fe.py pipeline).
    The old realmlp_fe() function is deprecated.
    """
    base = {'format': 'realmlp', 'te': True}
    if fe_tag == 'baseline':
        return {**base, 'stress_pal': True, 'sleep_cat': False}
    elif fe_tag == 'sleep':
        return {**base, 'stress_pal': True, 'sleep_cat': True,
                'sleep_interact_with': ['stress_pal', 'stress_level', 'physical_activity_level']}
    elif fe_tag == 'realmlp_plain':
        return {**base, 'stress_pal': False, 'gp_features': False, 'extra_cat_features': False,
                'water_intake_round': False, 'sleep_cat': False}
    elif fe_tag == 'realmlp_ref':
        return {**base, 'stress_pal': False, 'gp_features': True, 'sleep_cat': False}
    else:
        # Unknown tag: return minimal defaults with stress_pal=False (safe for RealMLP)
        return {**base, 'stress_pal': False, 'sleep_cat': False}


def _cache_key(model_name, fe_tag, n_splits, params, fe_config=None):
    """Generate cache filename stem."""
    ph = _params_hash(params)
    key = f'{model_name}_{fe_tag}_{n_splits}f_{ph}'
    # Include fe_config hash to avoid cache collision between RealMLP variants
    if fe_config:
        fch = hashlib.md5(str(sorted(fe_config.items())).encode()).hexdigest()[:6]
        key += f'_fc{fch}'
    return key


def _check_cache(key):
    """Check if OOF/test files exist in cache. Return (oof, test) or None."""
    oof_path = os.path.join(CACHE_DIR, f'{key}_oof.npy')
    test_path = os.path.join(CACHE_DIR, f'{key}_test.npy')
    meta_path = os.path.join(CACHE_DIR, f'{key}_meta.json')
    if os.path.exists(oof_path) and os.path.exists(test_path) and os.path.exists(meta_path):
        oof = np.load(oof_path)
        test = np.load(test_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return oof, test, meta
    return None


def _save_cache(key, oof, test, model_name, fe_tag, n_splits, params, ba):
    """Save OOF/test to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f'{key}_oof.npy'), oof)
    np.save(os.path.join(CACHE_DIR, f'{key}_test.npy'), test)
    meta = {
        'model': model_name, 'fe_tag': fe_tag, 'n_splits': n_splits,
        'params_hash': _params_hash(params), 'params': params,
        'oof_ba': float(ba), 'key': key,
    }
    with open(os.path.join(CACHE_DIR, f'{key}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def _check_legacy(model_name, fe_tag, n_splits):
    """Check legacy oof/ directory for existing files (backward compat).

    Legacy naming: _fe_A_num2cat_{MODEL}_oof.npy, _realmlp_oof.npy, etc.
    """
    # Map (model, fe_tag, nfolds) to legacy filename patterns
    # NOTE: Individual model OOFs in oof/ are ALL 7-fold (from _fe_A_num2cat_7fold.py)
    # 5-fold individual OOFs were never saved — only 5-fold equal blend was saved.
    # So we only map 7-fold to legacy; 5-fold must be trained fresh.
    legacy_map = {
        ('XGB', 'A_median', 7): '_fe_A_num2cat_XGB',
        ('CB', 'A_median', 7): '_fe_A_num2cat_CB',
        ('HGB', 'A_median', 7): '_fe_A_num2cat_HGB',
        ('RealMLP', 'yekenot', 7): '_realmlp',
        # numte variants (A_num2cat + num TE, 7-fold, te=False, num_te=True)
        ('XGB', 'APN_median', 7): '_fe_A_num2cat_numte_XGB',
        ('CB', 'APN_median', 7): '_fe_A_num2cat_numte_CB',
        ('HGB', 'APN_median', 7): '_fe_A_num2cat_numte_HGB',
        # _realmlp_ref variant
        ('RealMLP', 'realmlp_ref', 7): '_realmlp_ref',
    }
    key = (model_name, fe_tag, n_splits)
    if key in legacy_map:
        prefix = legacy_map[key]
        oof_path = os.path.join(OOF_DIR, f'{prefix}_oof.npy')
        test_path = os.path.join(OOF_DIR, f'{prefix}_test.npy')
        if os.path.exists(oof_path) and os.path.exists(test_path):
            return np.load(oof_path), np.load(test_path)
    return None


# ============================================================
# Individual model training functions
# ============================================================

def train_xgb(X, y, X_test, n_splits=5, params=None, use_eval_set=True):
    """Train XGB with n-fold CV. Returns (oof_proba, test_proba).
    
    Args:
        use_eval_set: If True (default), pass eval_set for early stopping.
            Set False to match reference notebooks that don't use eval_set.
            When early_stopping_rounds is None/0, eval_set has no effect on
            training but is still passed — set use_eval_set=False to skip it
            entirely for exact reproducibility.
    """
    # Keep DataFrame if available — XGB handles both
    is_df = hasattr(X, 'iloc')
    if is_df:
        n, n_test = len(X), len(X_test)
    else:
        X_arr = np.asarray(X)
        X_test_arr = np.asarray(X_test)
        n, n_test = len(X_arr), len(X_test_arr)
    
    params = params or dict(
        n_estimators=1500, learning_rate=0.05, max_depth=7,
        min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
        device='cuda', random_state=RANDOM_STATE, verbosity=0,
        early_stopping_rounds=50
    )
    cv_seed = params.get('random_state', RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros((n, 3))
    test = np.zeros((n_test, 3))

    t0 = time.time()
    split_data = X if is_df else X_arr
    for fold, (trn, val) in enumerate(skf.split(split_data, y), 1):
        sw = compute_sample_weight('balanced', y[trn])
        m = XGBClassifier(**params)
        
        if is_df:
            X_trn, X_val = X.iloc[trn], X.iloc[val]
            X_tst = X_test
        else:
            X_trn, X_val = X_arr[trn], X_arr[val]
            X_tst = X_test_arr
        
        fit_kwargs = dict(sample_weight=sw, verbose=False)
        if use_eval_set:
            fit_kwargs['eval_set'] = [(X_val, y[val])]
        m.fit(X_trn, y[trn], **fit_kwargs)
        
        oof[val] = m.predict_proba(X_val)
        test += m.predict_proba(X_tst) / n_splits
        ba = balanced_accuracy_score(y[val], oof[val].argmax(1))
        print(f'  XGB fold {fold}/{n_splits}: BA={ba:.5f}', flush=True)

    ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f'  XGB total: BA={ba:.5f} ({time.time()-t0:.0f}s)', flush=True)
    return oof, test, params, ba


def train_cb(X, y, X_test, cat_indices, n_splits=5, params=None):
    """Train CatBoost with n-fold CV. Returns (oof_proba, test_proba)."""
    if not hasattr(X, 'iloc'):
        X = pd.DataFrame(X)
    if not hasattr(X_test, 'iloc'):
        X_test = pd.DataFrame(X_test)
    params = params or dict(
        iterations=1500, learning_rate=0.05, depth=5, l2_leaf_reg=1.0,
        auto_class_weights='Balanced', task_type='GPU',
        random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=100
    )
    cv_seed = params.get('random_seed', params.get('random_state', RANDOM_STATE))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros((len(X), 3))
    test = np.zeros((len(X_test), 3))

    t0 = time.time()
    for fold, (trn, val) in enumerate(skf.split(X, y), 1):
        m = CatBoostClassifier(**params)
        tr_pool = Pool(X.iloc[trn], y[trn], cat_features=cat_indices)
        va_pool = Pool(X.iloc[val], y[val], cat_features=cat_indices)
        m.fit(tr_pool, eval_set=va_pool, verbose=0)
        oof[val] = m.predict_proba(va_pool)
        test += m.predict_proba(Pool(X_test, cat_features=cat_indices)) / n_splits
        ba = balanced_accuracy_score(y[val], oof[val].argmax(1))
        print(f'  CB fold {fold}/{n_splits}: BA={ba:.5f}', flush=True)

    ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f'  CB total: BA={ba:.5f} ({time.time()-t0:.0f}s)', flush=True)
    return oof, test, params, ba


def train_hgb(X, y, X_test, n_splits=5, params=None):
    """Train HistGradientBoosting with n-fold CV. Returns (oof_proba, test_proba)."""
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    X_test_arr = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
    params = params or dict(
        max_iter=1000, learning_rate=0.05, max_depth=8,
        min_samples_leaf=50, l2_regularization=1.0, max_leaf_nodes=31,
        class_weight='balanced', random_state=RANDOM_STATE, verbose=0,
        early_stopping=True, n_iter_no_change=50, validation_fraction=0.1
    )
    cv_seed = params.get('random_state', RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros((len(X_arr), 3))
    test = np.zeros((len(X_test_arr), 3))

    t0 = time.time()
    for fold, (trn, val) in enumerate(skf.split(X_arr, y), 1):
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_arr[trn], y[trn])
        oof[val] = m.predict_proba(X_arr[val])
        test += m.predict_proba(X_test_arr) / n_splits
        ba = balanced_accuracy_score(y[val], oof[val].argmax(1))
        print(f'  HGB fold {fold}/{n_splits}: BA={ba:.5f}', flush=True)

    ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f'  HGB total: BA={ba:.5f} ({time.time()-t0:.0f}s)', flush=True)
    return oof, test, params, ba


def train_lgb(X, y, X_test, n_splits=5, params=None):
    """Train LightGBM with n-fold CV. Returns (oof_proba, test_proba).
    
    Keeps DataFrame input when possible — LGBM can auto-detect categorical
    columns from dtype when given a DataFrame (vs numpy where everything
    is numeric). This matches reference notebook behavior.
    
    Default params match reference notebook (0.95043): default LGBM params
    with class_weight='balanced'. Works best with ordinal-encoded categoricals
    and domain features.
    """
    # Keep DataFrame if available — LGBM benefits from dtype info
    is_df = hasattr(X, 'iloc')
    params = params or dict(
        random_state=RANDOM_STATE, verbose=-1, class_weight='balanced',
        n_estimators=100, learning_rate=0.1, max_depth=-1,
    )
    cv_seed = params.get('random_state', RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    
    # Use DataFrame indexing when available, else numpy
    if is_df:
        n = len(X)
        n_test = len(X_test)
    else:
        X_arr = np.asarray(X)
        X_test_arr = np.asarray(X_test)
        n = len(X_arr)
        n_test = len(X_test_arr)
    
    oof = np.zeros((n, 3))
    test = np.zeros((n_test, 3))

    t0 = time.time()
    for fold, (trn, val) in enumerate(skf.split(
            X if is_df else X_arr, y), 1):
        m = lgb.LGBMClassifier(**params)
        if is_df:
            m.fit(X.iloc[trn], y[trn])
            oof[val] = m.predict_proba(X.iloc[val])
            test += m.predict_proba(X_test) / n_splits
        else:
            m.fit(X_arr[trn], y[trn])
            oof[val] = m.predict_proba(X_arr[val])
            test += m.predict_proba(X_test_arr) / n_splits
        ba = balanced_accuracy_score(y[val], oof[val].argmax(1))
        print(f'  LGB fold {fold}/{n_splits}: BA={ba:.5f}', flush=True)

    ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f'  LGB total: BA={ba:.5f} ({time.time()-t0:.0f}s)', flush=True)
    return oof, test, params, ba


# ============================================================
# RealMLP training
# ============================================================

def train_realmlp(n_splits=7, fe_config=None, params=None, oof_prefix=None, y=None):
    """Train RealMLP with n-fold CV. Uses unified fe.py for feature engineering.
    
    Uses RealMLP_TD_Classifier (matches reference 0.95090 notebook).
    Cache is based on a hash of (fe_config, params, n_splits).
    
    Args:
        n_splits: number of CV folds
        fe_config: dict of FE options (te, gp_features, stress_pal, etc.)
        params: dict of model hyperparams to override CONFIG
        oof_prefix: if set, save OOF/test to oof/{oof_prefix}_oof.npy and
                    oof/{oof_prefix}_test.npy, and submission to
                    submissions/sub_{oof_prefix}.csv
        y: target array (if None, loaded from train.csv)
    
    Returns: (oof_proba, test_proba, ba)
    """
    from pipeline.realmlp import RealMLP_TD_Classifier, CONFIG, seed_everything
    from pipeline.fe import get_or_create_features, build_tag
    from sklearn.preprocessing import TargetEncoder
    import torch.nn as nn
    
    p = CONFIG.copy()
    if params:
        p.update(params)
    
    # Resolve string activation names to nn.Module classes (YAML can't serialize classes)
    _ACTIVATION_MAP = {
        'SiLU': nn.SiLU, 'GELU': nn.GELU, 'ReLU': nn.ReLU,
        'PReLU': nn.PReLU, 'Mish': nn.Mish, 'Tanh': nn.Tanh,
    }
    for key in ('activation', 'pbld_activation'):
        val = p.get(key)
        if isinstance(val, str) and val in _ACTIVATION_MAP:
            p[key] = _ACTIVATION_MAP[val]
    
    n_splits = n_splits or 7
    SEED = p.get('random_state', 63)
    seed_everything(SEED)
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load y if not provided
    if y is None:
        train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        TARGET = 'health_condition'
        y = train_raw[TARGET].map({'at-risk': 0, 'fit': 1, 'unhealthy': 2})
        del train_raw
    
    # Feature engineering (unified fe.py, format='realmlp')
    print('\nFeature engineering (unified fe.py, format=realmlp)...', flush=True)
    fe_cfg = fe_config.copy() if fe_config else {}
    fe_cfg['format'] = 'realmlp'
    X, X_test, cat_cols, feature_names, fe_tag = get_or_create_features(fe_cfg, y)
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))
    
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []
    
    t_total = time.time()
    for fold, (tr_idx, val_idx) in enumerate(splits, 1):
        print(f'\nFold {fold}/{n_splits} ...', flush=True)
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_tst = X_test.copy()
        
        # Per-fold Target Encoding
        te_config = fe_config or {}
        if te_config.get('te', True):
            te_cols = [col for col in cat_cols if not col.endswith('bin_')]
            encoder = TargetEncoder(cv=n_splits, smooth='auto', shuffle=True, random_state=SEED)
            tr_enc = encoder.fit_transform(X_tr[te_cols], y[tr_idx])
            val_enc = encoder.transform(X_val[te_cols])
            tst_enc = encoder.transform(X_tst[te_cols])
            te_names = [f'_{col}TE_class{cls}' for col in te_cols for cls in range(3)]
            X_tr[te_names] = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc
            if fold == 1:
                print(f'  Features with TE: {len(X_tr.columns)}', flush=True)
        
        # Sort columns consistently (matches reference notebook)
        all_cols = sorted(X_tr.columns.tolist())
        X_tr = X_tr.reindex(all_cols, axis=1)
        X_val = X_val.reindex(all_cols, axis=1)
        X_tst = X_tst.reindex(all_cols, axis=1)
        
        current_cat_cols = sorted([c for c in cat_cols if c in X_tr.columns])
        
        # Train with RealMLP_TD_Classifier
        model = RealMLP_TD_Classifier(**p)
        model.fit(
            X_tr, y[tr_idx],
            X_val, y[val_idx],
            cat_col_names=current_cat_cols,
        )
        oof_preds[val_idx] = model.best_val_probs_
        test_preds += model.predict_proba(X_tst) / n_splits
        
        fold_score = balanced_accuracy_score(y[val_idx], np.argmax(oof_preds[val_idx], axis=1))
        fold_scores.append(fold_score)
        print(f'  Fold {fold} | Score: {fold_score:.5f}', flush=True)
        torch = __import__('torch')
        torch.cuda.empty_cache()
    
    t_total = time.time() - t_total
    oof_ba = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    print(f'\nRealMLP OOF BA: {oof_ba:.5f} ({t_total:.0f}s / {t_total/60:.1f}min)', flush=True)
    
    # Save OOF/test/submission if oof_prefix is specified
    if oof_prefix:
        SUB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'submissions')
        os.makedirs(OOF_DIR, exist_ok=True)
        os.makedirs(SUB_DIR, exist_ok=True)
        
        np.save(os.path.join(OOF_DIR, f'{oof_prefix}_oof.npy'), oof_preds)
        np.save(os.path.join(OOF_DIR, f'{oof_prefix}_test.npy'), test_preds)
        print(f'  Saved OOF: {os.path.join(OOF_DIR, f"{oof_prefix}_oof.npy")}', flush=True)
        print(f'  Saved test: {os.path.join(OOF_DIR, f"{oof_prefix}_test.npy")}', flush=True)
        
        # Generate submission
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        sub = pd.DataFrame({'id': test_df['id']})
        sub['health_condition'] = pd.Series(np.argmax(test_preds, axis=1)).map(
            {0: 'at-risk', 1: 'fit', 2: 'unhealthy'})
        sub_path = os.path.join(SUB_DIR, f'sub_{oof_prefix}.csv')
        sub.to_csv(sub_path, index=False)
        print(f'  Saved submission: {sub_path}', flush=True)
    
    return oof_preds, test_preds, oof_ba


# ============================================================
# Unified model training entry point (with cache)
# ============================================================

def get_or_train_model(model_name, fe_tag, X, y, X_test, cat_indices,
                       n_splits=5, params=None, feature_subset=None, fe_config=None,
                       model_type=None):
    """Train a single model with cache support.

    Checks: 1) new cache  2) legacy oof/  3) train from scratch

    Args:
        model_name: Component alias (e.g. 'RM_ref', 'XGB_v2') — used for display and blend
        fe_tag: feature engineering tag for cache key
        X, y, X_test: feature matrices and target
        cat_indices: categorical column indices
        n_splits: number of CV folds
        params: model hyperparameters dict
        feature_subset: 'all', 'base', or list of column names (default: 'all')
        fe_config: RealMLP fe_config override (ignored for GBDT models)
        model_type: Actual model type ('XGB', 'CB', 'HGB', 'RealMLP').
                    If None, inferred from model_name.

    Returns: (oof, test, ba)
    """
    # Resolve model_type from model_name if not specified
    _KNOWN_TYPES = {'XGB', 'CB', 'HGB', 'LGB', 'RealMLP'}
    if model_type is None:
        if model_name in _KNOWN_TYPES:
            model_type = model_name
        elif 'RM' in model_name.upper() or 'REALMLP' in model_name.upper():
            model_type = 'RealMLP'
        elif 'LGB' in model_name.upper() or 'LIGHTGBM' in model_name.upper():
            model_type = 'LGB'
        else:
            model_type = model_name  # will fail later if truly unknown

    # Apply feature subset before training (GBDT only)
    if model_type != 'RealMLP' and feature_subset is not None and feature_subset != 'all':
        from pipeline.fe import get_feature_subset
        X, X_test, cat_indices = get_feature_subset(
            X, X_test, cat_indices, list(X.columns), feature_subset)
        n_kept = X.shape[1]
        print(f'[TRAIN] Feature subset "{feature_subset}": {n_kept}d', flush=True)
    # Resolve default params and merge with overrides
    defaults = {
        'XGB': dict(n_estimators=1500, learning_rate=0.05, max_depth=7,
                    min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
                    device='cuda', random_state=RANDOM_STATE, verbosity=0,
                    early_stopping_rounds=50),
        'CB': dict(iterations=1500, learning_rate=0.05, depth=5, l2_leaf_reg=1.0,
              auto_class_weights='Balanced', task_type='GPU',
              random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=100),
        'HGB': dict(max_iter=1000, learning_rate=0.05, max_depth=8,
                    min_samples_leaf=50, l2_regularization=1.0, max_leaf_nodes=31,
                    class_weight='balanced', random_state=RANDOM_STATE, verbose=0,
                    early_stopping=True, n_iter_no_change=50, validation_fraction=0.1),
        'LGB': dict(random_state=RANDOM_STATE, verbose=-1, class_weight='balanced',
                    n_estimators=100, learning_rate=0.1, max_depth=-1),
    }
    if params is None:
        params = defaults.get(model_type, {})
    else:
        # Merge: defaults as base, params as overrides
        base = defaults.get(model_type, {})
        merged = base.copy()
        merged.update(params)
        params = merged

    # Check new cache (uses merged params for correct cache key)
    key = _cache_key(model_name, fe_tag, n_splits, params, fe_config=fe_config)
    cached = _check_cache(key)
    if cached is not None:
        oof, test, meta = cached
        print(f'[TRAIN] Cache hit: {key} (BA={meta["oof_ba"]:.5f})', flush=True)
        return oof, test, meta['oof_ba']

    # Check legacy oof/ directory
    legacy = _check_legacy(model_name, fe_tag, n_splits)
    if legacy is not None:
        oof, test = legacy
        ba = balanced_accuracy_score(y, oof.argmax(1))
        print(f'[TRAIN] Legacy hit: {model_name}/{fe_tag}/{n_splits}f (BA={ba:.5f})', flush=True)
        # Copy to new cache
        _save_cache(key, oof, test, model_name, fe_tag, n_splits, params, ba)
        return oof, test, ba

    # Train from scratch
    print(f'[TRAIN] Training: {model_name} ({model_type}) on {fe_tag} {n_splits}f', flush=True)
    if model_type == 'XGB':
        # Extract use_eval_set from params if specified (default: True for backward compat)
        use_eval = params.pop('use_eval_set', True)
        oof, test, params, ba = train_xgb(X, y, X_test, n_splits, params, use_eval_set=use_eval)
    elif model_type == 'CB':
        oof, test, params, ba = train_cb(X, y, X_test, cat_indices, n_splits, params)
    elif model_type == 'HGB':
        oof, test, params, ba = train_hgb(X, y, X_test, n_splits, params)
    elif model_type == 'LGB':
        oof, test, params, ba = train_lgb(X, y, X_test, n_splits, params)
    elif model_type == 'RealMLP':
        # RealMLP loads raw data and runs unified fe.py FE pipeline (format='realmlp')
        if fe_config is None:
            fe_config = _decode_fe_config(fe_tag)
        oof, test, ba = train_realmlp(n_splits=n_splits, fe_config=fe_config, params=params, y=y)
        _save_cache(key, oof, test, model_name, fe_tag, n_splits, params, ba)
        return oof, test, ba
    else:
        raise ValueError(f'Unknown model_type: {model_type} (model_name={model_name})')

    _save_cache(key, oof, test, model_name, fe_tag, n_splits, params, ba)
    return oof, test, ba
