"""AutoGluon training module for S6E7.

Usage via pipeline:
    python run_v2.py configs/champion_v1_ag.yaml

Standalone:
    python -m pipeline.autogluon --preset good_quality --time_limit 3600

Output:
    oof/{save_prefix}_oof.npy   — (N, 3) OOF probabilities
    oof/{save_prefix}_test.npy  — (M, 3) test probabilities
    submissions/sub_{save_prefix}.csv
    autogluon_output/           — AutoGluon's own save directory
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# Project root
_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_ROOT, 'data')
OOF_DIR = os.path.join(_ROOT, 'oof')
SUB_DIR = os.path.join(_ROOT, 'submissions')
AG_DIR = os.path.join(_ROOT, 'autogluon_output')


def _progress_callback(total_time_sofar, **kwargs):
    """Called by AutoGluon periodically during training."""
    elapsed_min = total_time_sofar / 60
    print(f'  [AG Progress] {elapsed_min:.1f}min elapsed...', flush=True)


def run_autogluon(preset='good_quality', time_limit=3600,
                  label_col='health_condition', eval_metric='balanced_accuracy',
                  save_prefix='_autogluon', excluded_model_types=None,
                  use_fe=False, fe_config=None, num_gpus=1):
    """Train AutoGluon and extract OOF + test predictions.

    Args:
        preset: 'medium_quality', 'good_quality', 'high_quality', 'best_quality'
        time_limit: max training time in seconds
        label_col: target column name
        eval_metric: evaluation metric
        save_prefix: prefix for saved OOF files
        excluded_model_types: list of model types to exclude
        use_fe: if True, apply our FE pipeline before feeding to AutoGluon
        fe_config: fe config dict (same format as YAML fe: section)
        num_gpus: number of GPUs to use (0=CPU only, 1=use GPU for supported models)

    Returns:
        oof_proba: (N, 3) OOF probabilities
        test_proba: (M, 3) test probabilities
        oof_ba: OOF balanced accuracy
    """
    from autogluon.tabular import TabularPredictor

    # Load data
    print(f'[AG] Loading data...', flush=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    le = LabelEncoder()
    y = le.fit_transform(train_df[label_col].values)
    print(f'[AG] Train: {train_df.shape}, Test: {test_df.shape}', flush=True)
    print(f'[AG] Classes: {le.classes_} → {list(range(len(le.classes_)))}', flush=True)

    # Optional: apply our FE pipeline before AutoGluon
    if use_fe and fe_config:
        print(f'[AG] Applying our FE pipeline...', flush=True)
        from pipeline.fe import get_or_create_features
        # Ensure format is set for GBDT (integers + TE features)
        fe_cfg = dict(fe_config)
        fe_cfg.setdefault('format', 'gbdt')
        X_train, X_test, cat_idx, feat_names, tag = get_or_create_features(fe_cfg, y)
        # FE may return DataFrame or numpy — normalize to DataFrame
        if not isinstance(X_train, pd.DataFrame):
            train_fe = pd.DataFrame(X_train, columns=feat_names)
            test_fe = pd.DataFrame(X_test, columns=feat_names)
        else:
            train_fe = X_train.copy()
            test_fe = X_test.copy()
        # Add label column (AutoGluon needs original string labels)
        train_fe[label_col] = le.inverse_transform(y)
        # Set category columns for AutoGluon (only cat_idx columns, NOT TE columns)
        cat_cols_list = [feat_names[i] for i in cat_idx if i < len(feat_names)]
        for col in cat_cols_list:
            if col in train_fe.columns:
                train_fe[col] = train_fe[col].astype('category')
                test_fe[col] = test_fe[col].astype('category')
        ag_train = train_fe
        ag_test = test_fe
        print(f'[AG] FE applied: {len(feat_names)}d, tag={tag}, {len(cat_cols_list)} cat cols', flush=True)
    else:
        # Raw data — AutoGluon does its own FE (MAXIMUM diversity)
        ag_train = train_df.drop(columns=['id'])
        ag_test = test_df.drop(columns=['id'])
        # AutoGluon handles category detection automatically
        print(f'[AG] Using raw data (AutoGluon internal FE)', flush=True)

    # Clean up previous run
    if os.path.exists(AG_DIR):
        import shutil
        print(f'[AG] Cleaning previous output...', flush=True)
        shutil.rmtree(AG_DIR)

    # Train AutoGluon
    print(f'\n{"="*60}', flush=True)
    print(f'AutoGluon Training', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Preset:          {preset}', flush=True)
    print(f'  Time limit:      {time_limit}s ({time_limit/3600:.1f}h)', flush=True)
    print(f'  Eval metric:     {eval_metric}', flush=True)
    print(f'  Input features:  {ag_train.shape[1]-1} ({"our FE" if use_fe else "raw"})', flush=True)
    print(f'  Train samples:   {len(ag_train)}', flush=True)
    print(f'  num_gpus:        {num_gpus}', flush=True)
    if excluded_model_types:
        print(f'  Excluded models: {excluded_model_types}', flush=True)
    print(f'  Start time:      {time.strftime("%H:%M:%S")}', flush=True)
    print(flush=True)

    t0 = time.time()

    # GPU strategy: Ray not available on Python 3.13 Windows → AG detects 0 GPUs.
    # Workaround: set num_gpus=0 (don't trigger AG's GPU management via Ray),
    # but pass GPU directives via model-native hyperparameters instead.
    # XGB/CB GPU params are INDEPENDENT of Ray/torch — they have their own CUDA support.
    # NN_TORCH needs torch cu128 for GPU (separate install).
    # So we ALWAYS pass XGB/CB GPU params, regardless of num_gpus.
    # num_gpus only controls NN_TORCH GPU via AG's internal routing.
    hyperparameters = {
        'GBM': {},   # LightGBM: CPU is faster than GPU for tabular
        'NN_TORCH': {},  # PyTorch: will use CUDA if torch cu128 installed, else CPU
        'XGB': [{'device': 'cuda'}],  # XGB has native CUDA support (independent of torch)
        'CAT': [{'task_type': 'GPU', 'devices': '0'}],  # CatBoost has native GPU support
    }

    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        path=AG_DIR,
    ).fit(
        train_data=ag_train,
        presets=preset,
        time_limit=time_limit,
        excluded_model_types=excluded_model_types,
        num_gpus=0,  # Ray unavailable → set 0 to avoid AssertionError
        hyperparameters=hyperparameters,
        ag_args_fit={'drop_unique': False},  # don't drop unique cols (like id if present)
    )

    train_time = time.time() - t0
    print(f'\n[AG] Training completed in {train_time:.0f}s ({train_time/60:.1f}min)', flush=True)

    # Model leaderboard
    print(f'\n[AG] Model Leaderboard:', flush=True)
    leaderboard = predictor.leaderboard(extra_info=False)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 40)
    print(leaderboard.to_string())
    print(f'[AG] Total models: {len(leaderboard)}', flush=True)

    # Extract OOF predictions
    print(f'\n[AG] Extracting OOF predictions...', flush=True)
    oof_proba = predictor.predict_proba_oof(as_multiclass=True)
    if hasattr(oof_proba, 'values'):
        oof_proba = oof_proba.values
    oof_proba = np.array(oof_proba, dtype=np.float64)

    # Ensure column order matches our pipeline: at-risk=0, fit=1, unhealthy=2
    ag_classes = predictor.class_labels
    print(f'[AG] AutoGluon class order: {ag_classes}', flush=True)
    print(f'[AG] Pipeline class order:  {list(le.classes_)}', flush=True)

    reorder = None
    if list(ag_classes) != list(le.classes_):
        print(f'[AG] WARNING: Class order mismatch! Reordering...', flush=True)
        reorder = [list(ag_classes).index(c) for c in le.classes_]
        oof_proba = oof_proba[:, reorder]

    oof_ba = balanced_accuracy_score(y, oof_proba.argmax(1))
    print(f'[AG] OOF BA: {oof_ba:.5f}', flush=True)

    # Extract test predictions
    print(f'[AG] Extracting test predictions...', flush=True)
    test_proba = predictor.predict_proba(ag_test, as_multiclass=True)
    if hasattr(test_proba, 'values'):
        test_proba = test_proba.values
    test_proba = np.array(test_proba, dtype=np.float64)

    if reorder is not None:
        test_proba = test_proba[:, reorder]

    print(f'[AG] Test shape: {test_proba.shape}', flush=True)

    # Save OOF + test
    os.makedirs(OOF_DIR, exist_ok=True)
    np.save(os.path.join(OOF_DIR, f'{save_prefix}_oof.npy'), oof_proba)
    np.save(os.path.join(OOF_DIR, f'{save_prefix}_test.npy'), test_proba)
    print(f'[AG] Saved: oof/{save_prefix}_oof.npy, oof/{save_prefix}_test.npy', flush=True)

    # Generate submission
    os.makedirs(SUB_DIR, exist_ok=True)
    pred_labels = le.inverse_transform(test_proba.argmax(1))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['health_condition'] = pred_labels
    sub_path = os.path.join(SUB_DIR, f'sub_{save_prefix.strip("_")}.csv')
    sub.to_csv(sub_path, index=False)
    print(f'[AG] Submission: {sub_path}', flush=True)
    print(f'[AG] Value counts: {sub.health_condition.value_counts().to_dict()}', flush=True)

    # Print model-level OOF for diversity analysis
    print(f'\n[AG] Individual Model Diversity Analysis:', flush=True)
    model_names = predictor.model_names()
    for mname in model_names[:15]:
        try:
            m_oof = predictor.predict_proba_oof(model=mname, as_multiclass=True)
            if hasattr(m_oof, 'values'):
                m_oof = m_oof.values
            m_oof = np.array(m_oof, dtype=np.float64)
            if reorder is not None:
                m_oof = m_oof[:, reorder]
            m_ba = balanced_accuracy_score(y, m_oof.argmax(1))
            corr = [np.corrcoef(oof_proba[:, c], m_oof[:, c])[0, 1] for c in range(3)]
            print(f'  {mname:45s} BA={m_ba:.5f}  corr=[{corr[0]:.4f}, {corr[1]:.4f}, {corr[2]:.4f}]', flush=True)
        except Exception as e:
            print(f'  {mname:45s} ERROR: {e}', flush=True)

    # LB prediction (v7 formula)
    predicted_lb = oof_ba + 0.00049
    print(f'\n[AG] LB Prediction (v7, solo, no health correction): {predicted_lb:.5f}', flush=True)

    return oof_proba, test_proba, oof_ba


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AutoGluon for S6E7')
    parser.add_argument('--preset', default='good_quality',
                        choices=['medium_quality', 'good_quality', 'high_quality', 'best_quality'])
    parser.add_argument('--time_limit', type=int, default=3600)
    parser.add_argument('--prefix', default=None)
    parser.add_argument('--exclude', nargs='*', default=None)
    parser.add_argument('--use_fe', action='store_true',
                        help='Apply our FE pipeline before AutoGluon')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs (0=CPU only, default=1)')
    args = parser.parse_args()

    prefix = args.prefix or f'_autogluon_{args.preset}'

    # Default FE config when --use_fe is specified without explicit fe_config
    default_fe_config = {
        'num2cat': True,
        'te': True,
        'num_te': True,
        'format': 'gbdt',
    }

    oof, test, ba = run_autogluon(
        preset=args.preset,
        time_limit=args.time_limit,
        save_prefix=prefix,
        excluded_model_types=args.exclude,
        use_fe=args.use_fe,
        fe_config=default_fe_config if args.use_fe else None,
        num_gpus=args.num_gpus,
    )

    print(f'\nDone. OOF BA={ba:.5f}', flush=True)
