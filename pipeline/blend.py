"""Blend pipeline for S6E7.

Config-driven: reads YAML config, resolves dependencies, executes blend.
Flow: blend → needs model OOF → needs FE → runs as needed.

Supports:
- Single model component
- Equal-blend group (e.g., XGB+CB+HGB equal)
- Weighted blend of components
- Grid search for optimal weights
"""
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'blend')
MODEL_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'model')
OOF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'oof')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SUB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'submissions')


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def equal_blend(arrays):
    """Equal-weight average of probability arrays."""
    return sum(arrays) / len(arrays)


def weighted_blend(arrays, weights):
    """Weighted average of probability arrays."""
    if not arrays:
        raise ValueError("weighted_blend called with empty arrays")
    return sum(w * a for w, a in zip(weights, arrays))


def grid_search_2way(y, oof_a, oof_b, step=1):
    """Grid search for 2-component blend weights."""
    best_ba, best_w = 0, (0.5, 0.5)
    for w_int in range(20, 80, step):
        w = w_int / 100
        blend = w * oof_a + (1 - w) * oof_b
        ba = balanced_accuracy_score(y, blend.argmax(1))
        if ba > best_ba:
            best_ba = ba
            best_w = (w, 1 - w)
    return best_w, best_ba


def grid_search_nway(y, oof_dict, model_names=None, step=2):
    """Grid search for n-component blend weights."""
    if model_names is None:
        model_names = list(oof_dict.keys())
    n = len(model_names)
    if n == 2:
        (w0, w1), ba = grid_search_2way(y, oof_dict[model_names[0]], oof_dict[model_names[1]], step)
        return dict(zip(model_names, [w0, w1])), ba
    if n == 3:
        return _grid3(y, [oof_dict[m] for m in model_names], model_names, step)
    if n == 4:
        return _grid4(y, [oof_dict[m] for m in model_names], model_names, step)
    raise ValueError(f'Grid search for {n} components not implemented')


def _grid3(y, oofs, names, step):
    best_ba, best_w = 0, [1/3]*3
    for w0 in range(20, 70, step):
        for w1 in range(5, 40, step):
            w2 = 100 - w0 - w1
            if w2 < 5: continue
            blend = w0/100*oofs[0] + w1/100*oofs[1] + w2/100*oofs[2]
            ba = balanced_accuracy_score(y, blend.argmax(1))
            if ba > best_ba:
                best_ba = ba
                best_w = [w0/100, w1/100, w2/100]
    return dict(zip(names, best_w)), best_ba


def _grid4(y, oofs, names, step):
    best_ba, best_w = 0, [0.25]*4
    for w0 in range(30, 70, step):
        for w1 in range(0, 15, step):
            for w2 in range(5, 25, step):
                w3 = 100 - w0 - w1 - w2
                if w3 < 5: continue
                blend = (w0/100*oofs[0] + w1/100*oofs[1] +
                         w2/100*oofs[2] + w3/100*oofs[3])
                ba = balanced_accuracy_score(y, blend.argmax(1))
                if ba > best_ba:
                    best_ba = ba
                    best_w = [w0/100, w1/100, w2/100, w3/100]
    return dict(zip(names, best_w)), best_ba


def generate_submission(test_proba, le, output_path):
    """Generate submission CSV from test probabilities."""
    pred_labels = le.inverse_transform(test_proba.argmax(1))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['health_condition'] = pred_labels
    sub.to_csv(output_path, index=False)
    return sub


# ============================================================
# Per-class prior calibration (coordinate ascent)
# ============================================================

def coordinate_ascent_calibration(y_true, oof_proba, n_classes=3,
                                  n_rounds=5, n_steps=200,
                                  range_min=0.1, range_max=3.0,
                                  class_names=None, verbose=True):
    """Find per-class probability multipliers that maximize OOF balanced accuracy.

    For imbalanced 3-class problems, raw argmax systematically under-calls
    minority classes. Per-class scaling corrects this: multiply each class's
    probability column by a scale factor, then renormalize before argmax.

    Algorithm: coordinate ascent — cycle through classes, sweep each scale
    over [range_min, range_max], pick the value that maximizes BA.

    Args:
        y_true: (N,) integer labels
        oof_proba: (N, C) probability matrix
        n_classes: number of classes
        n_rounds: max coordinate ascent rounds
        n_steps: grid points per class per round
        range_min, range_max: scale search range
        class_names: optional list of class names for logging
        verbose: print progress

    Returns:
        scales: (C,) array of per-class multipliers
        best_ba: calibrated OOF BA
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(n_classes)]

    scales = np.ones(n_classes)
    best_ba = balanced_accuracy_score(y_true, np.argmax(oof_proba, axis=1))

    if verbose:
        print(f'  [CA] Before calibration: BA={best_ba:.5f}', flush=True)

    for rnd in range(n_rounds):
        improved = False
        for c in range(n_classes):
            best_scale_c = scales[c]
            best_ba_c = best_ba

            for s in np.linspace(range_min, range_max, n_steps):
                trial = scales.copy()
                trial[c] = s
                # Scale probabilities, then argmax (no renormalization needed
                # for argmax — monotonic transform preserves order per sample)
                preds = np.argmax(oof_proba * trial[np.newaxis, :], axis=1)
                ba = balanced_accuracy_score(y_true, preds)
                if ba > best_ba_c:
                    best_ba_c = ba
                    best_scale_c = s
                    improved = True

            scales[c] = best_scale_c
            best_ba = best_ba_c

        if verbose:
            scale_str = ', '.join(f'{class_names[i]}={scales[i]:.3f}'
                                  for i in range(n_classes))
            print(f'  [CA] Round {rnd+1}: BA={best_ba:.5f}  scales=[{scale_str}]',
                  flush=True)

        if not improved:
            if verbose:
                print(f'  [CA] Converged at round {rnd+1}', flush=True)
            break

    return scales, best_ba


def apply_calibration(proba, scales):
    """Apply per-class calibration scales to probability matrix.

    Scales each class column, then renormalizes so rows sum to 1.
    Renormalization is needed for proper probability interpretation,
    but argmax is invariant to it (monotonic transform).

    Args:
        proba: (N, C) probability matrix
        scales: (C,) per-class multipliers

    Returns:
        calibrated: (N, C) renormalized probability matrix
    """
    calibrated = proba * scales[np.newaxis, :]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-30)
    return calibrated


# ============================================================
# Config-driven blend runner
# ============================================================

def per_class_weighted_blend(arrays, weights_per_class):
    """Per-class weighted average of probability arrays.
    
    Args:
        arrays: list of (N, C) probability arrays
        weights_per_class: list of (C,) weight arrays, one per component.
            weights_per_class[i][c] is the weight of component i for class c.
            Weights are normalized per class so they sum to 1.
    
    Returns:
        (N, C) blended probability array
    """
    n_classes = arrays[0].shape[1]
    result = np.zeros_like(arrays[0])
    for c in range(n_classes):
        w_c = np.array([w[c] for w in weights_per_class])
        w_c = w_c / w_c.sum()  # normalize per class
        for i, a in enumerate(arrays):
            result[:, c] += w_c[i] * a[:, c]
    return result


def run_blend(config_path):
    """Main entry: load config, resolve dependencies, execute blend.

    Config structure (YAML):
        name: experiment_name
        fe:              # FE configs (keyed by tag)
          A_num2cat:
            num2cat: true
            stress_pal: true
            num_fill: median
            te: false
        models:          # Model configs
          - name: XGB
            fe: A_num2cat
            n_splits: 5
            params: {...}  # optional, uses defaults if omitted
          - name: CB
            fe: A_num2cat
            n_splits: 5
          - name: HGB
            fe: A_num2cat
            n_splits: 5
          - name: RealMLP
            fe: yekenot
            n_splits: 7
            legacy_oof: _realmlp  # use existing OOF file
        blend:
          components:
            - name: RM
              models: [RealMLP]
              weight: 0.55
            - name: GB_A
              models: [XGB, CB, HGB]
              inner: equal
              weight: 0.45
          # OR use grid_search: true to auto-find weights
          # OR use per_class_weights for per-class blending:
          #   per_class_weights:
          #     RM: [0.60, 0.32, 0.60]
          #     GB_A: [0.40, 0.68, 0.40]
    """
    from pipeline.fe import get_or_create_features, build_tag
    from pipeline.train import get_or_train_model

    config = load_config(config_path)
    name = config['name']
    print(f'{"="*60}', flush=True)
    print(f'Experiment: {name}', flush=True)
    print(f'{"="*60}', flush=True)

    # Load target
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    le = LabelEncoder()
    y = le.fit_transform(train['health_condition'].values)

    # Resolve FE configs
    fe_configs = config.get('fe', {})
    fe_cache = {}  # fe_name -> (X_train, X_test, cat_idx, feat_names, tag)

    for fe_name, fe_cfg in fe_configs.items():
        X_train, X_test, cat_idx, feat_names, tag = get_or_create_features(fe_cfg, y)
        fe_cache[fe_name] = (X_train, X_test, cat_idx, feat_names, tag)
        print(f'  FE "{fe_name}" -> tag={tag}, {len(feat_names)}d', flush=True)

    # Train models (or load from cache)
    model_oof = {}   # model_name -> oof array
    model_test = {}  # model_name -> test array
    model_ba = {}    # model_name -> OOF BA

    for mcfg in config.get('models', []):
        mname = mcfg['name']
        n_splits = mcfg.get('n_splits', 5)
        params = mcfg.get('params', None)

        # Check if this is a legacy-only model (e.g., RealMLP)
        legacy_key = mcfg.get('legacy_oof', None)

        # Check if this is an AutoGluon model
        ag_cfg = mcfg.get('autogluon', None)

        if legacy_key:
            oof_path = os.path.join(OOF_DIR, f'{legacy_key}_oof.npy')
            test_path = os.path.join(OOF_DIR, f'{legacy_key}_test.npy')
            if os.path.exists(oof_path) and os.path.exists(test_path):
                oof = np.load(oof_path)
                test = np.load(test_path)
                ba = balanced_accuracy_score(y, oof.argmax(1))
                print(f'  [LEGACY] {mname}: BA={ba:.5f}', flush=True)
                model_oof[mname] = oof
                model_test[mname] = test
                model_ba[mname] = ba
                continue
            else:
                print(f'  [LEGACY] {mname}: files not found at {oof_path}, will train', flush=True)

        if ag_cfg:
            # AutoGluon model — run independently, produces OOF + test
            from run_autogluon import run_autogluon as _run_ag
            ag_preset = ag_cfg.get('preset', 'good_quality')
            ag_time = ag_cfg.get('time_limit', 3600)
            ag_exclude = ag_cfg.get('excluded_model_types', None)
            ag_prefix = ag_cfg.get('save_prefix', f'_autogluon_{ag_preset}')
            # Check if OOF already exists
            ag_oof_path = os.path.join(OOF_DIR, f'{ag_prefix}_oof.npy')
            ag_test_path = os.path.join(OOF_DIR, f'{ag_prefix}_test.npy')
            if os.path.exists(ag_oof_path) and os.path.exists(ag_test_path):
                oof = np.load(ag_oof_path)
                test = np.load(ag_test_path)
                ba = balanced_accuracy_score(y, oof.argmax(1))
                print(f'  [AUTOGLUON] {mname} (cached): BA={ba:.5f}', flush=True)
            else:
                print(f'  [AUTOGLUON] {mname}: training with preset={ag_preset}, time_limit={ag_time}s...', flush=True)
                oof, test, ba = _run_ag(
                    preset=ag_preset,
                    time_limit=ag_time,
                    save_prefix=ag_prefix,
                    excluded_model_types=ag_exclude,
                )
            model_oof[mname] = oof
            model_test[mname] = test
            model_ba[mname] = ba
            continue

        # Get FE data
        fe_name = mcfg.get('fe', list(fe_configs.keys())[0] if fe_configs else None)
        if fe_name is None:
            raise ValueError(f'Model {mname} has no fe config and no legacy_oof')
        X_train, X_test, cat_idx, feat_names, fe_tag = fe_cache[fe_name]

        # Train or load from cache
        oof, test, ba = get_or_train_model(
            mname, fe_tag, X_train, y, X_test, cat_idx,
            n_splits=n_splits, params=params,
            feature_subset=mcfg.get('feature_subset', None),
            fe_config=mcfg.get('fe_config', None),
            model_type=mcfg.get('model_type', None)
        )
        model_oof[mname] = oof
        model_test[mname] = test
        model_ba[mname] = ba

    # Print model summary
    print(f'\n--- Model Summary ---', flush=True)
    for mname, ba in model_ba.items():
        print(f'  {mname}: OOF BA={ba:.5f}', flush=True)

    # Execute blend
    blend_cfg = config.get('blend', {})
    components = blend_cfg.get('components', [])

    comp_oof = {}
    comp_test = {}

    for comp in components:
        cname = comp['name']
        models = comp['models']
        inner = comp.get('inner', 'single')  # 'single', 'equal', 'grid'

        if inner == 'single' and len(models) == 1:
            comp_oof[cname] = model_oof[models[0]]
            comp_test[cname] = model_test[models[0]]
        elif inner == 'equal':
            comp_oof[cname] = equal_blend([model_oof[m] for m in models])
            comp_test[cname] = equal_blend([model_test[m] for m in models])
        elif inner == 'weighted':
            # Fixed weights within this component
            w = comp.get('weights', {})
            if not w:
                raise ValueError(f'Component "{cname}" uses inner=weighted but no weights specified')
            comp_oof[cname] = weighted_blend(
                [model_oof[m] for m in models],
                [w[m] for m in models])
            comp_test[cname] = weighted_blend(
                [model_test[m] for m in models],
                [w[m] for m in models])
        elif inner == 'grid':
            # Grid search for optimal weights within this component
            oof_dict = {m: model_oof[m] for m in models}
            weights_inner, inner_ba = grid_search_nway(
                y, oof_dict, models, step=comp.get('grid_step', 2))
            print(f'  Inner grid search for "{cname}": {weights_inner} -> OOF BA={inner_ba:.5f}', flush=True)
            comp_oof[cname] = weighted_blend(
                [model_oof[m] for m in models],
                [weights_inner[m] for m in models])
            comp_test[cname] = weighted_blend(
                [model_test[m] for m in models],
                [weights_inner[m] for m in models])
        else:
            raise ValueError(f'Unknown inner blend: {inner}')

        ba = balanced_accuracy_score(y, comp_oof[cname].argmax(1))
        print(f'  Component "{cname}" ({inner}): OOF BA={ba:.5f}', flush=True)

    # Final blend
    blend_method = blend_cfg.get('method', 'components')  # 'components' or 'individual_grid'
    
    if blend_method == 'individual_grid':
        # Direct grid search over all models (no component grouping)
        model_names = blend_cfg.get('models', list(model_oof.keys()))
        oof_dict = {m: model_oof[m] for m in model_names}
        weights, best_ba = grid_search_nway(
            y, oof_dict, model_names, step=blend_cfg.get('grid_step', 2))
        print(f'\n  Individual grid search: {weights} -> OOF BA={best_ba:.5f}', flush=True)
        final_oof = weighted_blend([model_oof[m] for m in model_names],
                                   [weights[m] for m in model_names])
        final_test = weighted_blend([model_test[m] for m in model_names],
                                    [weights[m] for m in model_names])
        final_ba = best_ba
    elif blend_cfg.get('per_class_weights'):
        # Per-class weighted blend
        pcw = blend_cfg['per_class_weights']
        comp_names = list(pcw.keys())
        weights_per_class = [np.array(pcw[c]) for c in comp_names]
        final_oof = per_class_weighted_blend(
            [comp_oof[c] for c in comp_names], weights_per_class)
        final_test = per_class_weighted_blend(
            [comp_test[c] for c in comp_names], weights_per_class)
        final_ba = balanced_accuracy_score(y, final_oof.argmax(1))
        print(f'\n  Per-class blend: {pcw} -> OOF BA={final_ba:.5f}', flush=True)
        weights = pcw  # store for metadata
    elif blend_cfg.get('grid_search', False):
        comp_names = list(comp_oof.keys())
        weights, best_ba = grid_search_nway(y, comp_oof, comp_names,
                                            step=blend_cfg.get('grid_step', 1))
        print(f'\n  Grid search: {weights} -> OOF BA={best_ba:.5f}', flush=True)
        final_oof = weighted_blend([comp_oof[c] for c in comp_names],
                                   [weights[c] for c in comp_names])
        final_test = weighted_blend([comp_test[c] for c in comp_names],
                                    [weights[c] for c in comp_names])
        final_ba = best_ba
    else:
        # Fixed weights (or single component with no weight needed)
        if len(components) == 1:
            cname = components[0]['name']
            final_oof = comp_oof[cname]
            final_test = comp_test[cname]
            weights = {cname: 1.0}
        else:
            weights = {comp['name']: comp['weight'] for comp in components}
            final_oof = weighted_blend([comp_oof[c] for c in weights],
                                       [weights[c] for c in weights])
            final_test = weighted_blend([comp_test[c] for c in weights],
                                        [weights[c] for c in weights])
        final_ba = balanced_accuracy_score(y, final_oof.argmax(1))

    final_ba = balanced_accuracy_score(y, final_oof.argmax(1))
    print(f'\n  Final blend: OOF BA={final_ba:.5f}', flush=True)

    # Save blend results
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    blend_tag = name
    # Save to both cache/blend/ and oof/ for backward compat
    for save_dir in [CACHE_DIR, OOF_DIR]:
        np.save(os.path.join(save_dir, f'{blend_tag}_oof.npy'), final_oof)
        np.save(os.path.join(save_dir, f'{blend_tag}_test.npy'), final_test)
    meta = {'name': name, 'weights': str(weights), 'oof_ba': float(final_ba),
            'config_path': str(config_path)}
    with open(os.path.join(CACHE_DIR, f'{blend_tag}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Generate submission
    sub_path = os.path.join(SUB_DIR, f'sub_{name}.csv')
    generate_submission(final_test, le, sub_path)
    print(f'  Saved: {sub_path}', flush=True)

    return final_ba, final_oof, final_test
