#!/usr/bin/env python
"""AutoGluon training script for S6E7.

Can run standalone or as part of the pipeline via config.

Standalone:
    python run_autogluon.py
    python run_autogluon.py --preset good_quality --time_limit 3600
    python run_autogluon.py --preset best_quality --time_limit 14400

As part of pipeline (via config YAML):
    python run_v2.py configs/champion_ag.yaml
    # where config has:
    #   models:
    #     - name: AutoGluon
    #       autogluon:
    #         preset: good_quality
    #         time_limit: 3600

Output:
    oof/_autogluon_oof.npy   — (N, 3) OOF probabilities
    oof/_autogluon_test.npy  — (M, 3) test probabilities
    submissions/sub_autogluon.csv
    autogluon_output/        — AutoGluon's own save directory
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OOF_DIR = os.path.join(os.path.dirname(__file__), 'oof')
SUB_DIR = os.path.join(os.path.dirname(__file__), 'submissions')
AG_DIR = os.path.join(os.path.dirname(__file__), 'autogluon_output')


def run_autogluon(preset='good_quality', time_limit=3600,
                  label_col='health_condition', eval_metric='balanced_accuracy',
                  save_prefix='_autogluon', excluded_model_types=None,
                  sample_weight=None):
    """Train AutoGluon and extract OOF + test predictions.

    Args:
        preset: 'good_quality', 'high_quality', 'best_quality'
        time_limit: max training time in seconds
        label_col: target column name
        eval_metric: evaluation metric
        save_prefix: prefix for saved OOF files (e.g., '_autogluon_good')
        excluded_model_types: list of model types to exclude
        sample_weight: optional sample weight column name

    Returns:
        oof_proba: (N, 3) OOF probabilities
        test_proba: (M, 3) test probabilities
        oof_ba: OOF balanced accuracy
    """
    from autogluon.tabular import TabularPredictor

    # Load data
    print(f'Loading data...', flush=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    le = LabelEncoder()
    y = le.fit_transform(train_df[label_col].values)
    print(f'Train: {train_df.shape}, Test: {test_df.shape}', flush=True)
    print(f'Classes: {le.classes_} → {list(range(len(le.classes_)))}', flush=True)

    # Clean up previous run
    if os.path.exists(AG_DIR):
        import shutil
        shutil.rmtree(AG_DIR)

    # Train AutoGluon
    print(f'\n=== AutoGluon Training ===', flush=True)
    print(f'  Preset: {preset}', flush=True)
    print(f'  Time limit: {time_limit}s ({time_limit/3600:.1f}h)', flush=True)
    print(f'  Eval metric: {eval_metric}', flush=True)
    if excluded_model_types:
        print(f'  Excluded models: {excluded_model_types}', flush=True)

    t0 = time.time()

    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        path=AG_DIR,
    ).fit(
        train_data=train_df.drop(columns=['id']),
        presets=preset,
        time_limit=time_limit,
        excluded_model_types=excluded_model_types,
    )

    train_time = time.time() - t0
    print(f'\n  Training completed in {train_time:.0f}s ({train_time/60:.1f}min)', flush=True)

    # Model leaderboard
    leaderboard = predictor.leaderboard(extra_info=True)
    print(f'\n  Models trained: {len(leaderboard)}', flush=True)
    print(leaderboard[['model', 'score_test', 'pred_time_test', 'fit_time']].to_string())

    # Extract OOF predictions
    print(f'\n=== Extracting OOF Predictions ===', flush=True)
    oof_proba = predictor.predict_proba_oof(as_multiclass=True)
    if hasattr(oof_proba, 'values'):
        oof_proba = oof_proba.values
    oof_proba = np.array(oof_proba)

    # Ensure column order matches our pipeline: at-risk=0, fit=1, unhealthy=2
    # AutoGluon's class order should match LabelEncoder order
    ag_classes = predictor.class_labels
    print(f'  AutoGluon class order: {ag_classes}', flush=True)
    print(f'  Our class order: {list(le.classes_)}', flush=True)

    if list(ag_classes) != list(le.classes_):
        print(f'  WARNING: Class order mismatch! Reordering columns...', flush=True)
        reorder = [list(ag_classes).index(c) for c in le.classes_]
        oof_proba = oof_proba[:, reorder]

    oof_ba = balanced_accuracy_score(y, oof_proba.argmax(1))
    print(f'  OOF BA: {oof_ba:.5f}', flush=True)

    # Extract test predictions
    print(f'\n=== Extracting Test Predictions ===', flush=True)
    test_data = test_df.drop(columns=['id'])
    test_proba = predictor.predict_proba(test_data, as_multiclass=True)
    if hasattr(test_proba, 'values'):
        test_proba = test_proba.values
    test_proba = np.array(test_proba)

    if list(ag_classes) != list(le.classes_):
        test_proba = test_proba[:, reorder]

    print(f'  Test shape: {test_proba.shape}', flush=True)

    # Save OOF + test
    os.makedirs(OOF_DIR, exist_ok=True)
    np.save(os.path.join(OOF_DIR, f'{save_prefix}_oof.npy'), oof_proba)
    np.save(os.path.join(OOF_DIR, f'{save_prefix}_test.npy'), test_proba)
    print(f'  Saved: oof/{save_prefix}_oof.npy, oof/{save_prefix}_test.npy', flush=True)

    # Generate submission
    os.makedirs(SUB_DIR, exist_ok=True)
    pred_labels = le.inverse_transform(test_proba.argmax(1))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['health_condition'] = pred_labels
    sub_path = os.path.join(SUB_DIR, f'sub_{save_prefix.strip("_")}.csv')
    sub.to_csv(sub_path, index=False)
    print(f'  Submission: {sub_path}', flush=True)
    print(f'  Value counts: {sub.health_condition.value_counts().to_dict()}', flush=True)

    # Print model-level OOF for diversity analysis
    print(f'\n=== Individual Model OOF (for diversity analysis) ===', flush=True)
    model_names = predictor.model_names()
    for mname in model_names[:10]:  # top 10 only
        try:
            m_oof = predictor.predict_proba_oof(model=mname, as_multiclass=True)
            if hasattr(m_oof, 'values'):
                m_oof = m_oof.values
            m_oof = np.array(m_oof)
            if list(ag_classes) != list(le.classes_):
                m_oof = m_oof[:, reorder]
            m_ba = balanced_accuracy_score(y, m_oof.argmax(1))
            # Correlation with our best component
            corr_per_class = [np.corrcoef(oof_proba[:, c], m_oof[:, c])[0, 1]
                              for c in range(3)]
            print(f'  {mname:40s} OOF BA={m_ba:.5f}  corr={corr_per_class[0]:.4f}/{corr_per_class[1]:.4f}/{corr_per_class[2]:.4f}', flush=True)
        except Exception as e:
            print(f'  {mname:40s} ERROR: {e}', flush=True)

    # LB prediction (v7 formula)
    health = oof_ba - oof_ba  # single model, health=0
    predicted_lb = oof_ba + 0.00049 + 1.227 * health
    print(f'\n  LB Prediction (v7, solo): {predicted_lb:.5f}', flush=True)

    return oof_proba, test_proba, oof_ba


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AutoGluon for S6E7')
    parser.add_argument('--preset', default='good_quality',
                        choices=['medium_quality', 'good_quality', 'high_quality', 'best_quality'],
                        help='AutoGluon quality preset')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='Time limit in seconds (default: 3600 = 1h)')
    parser.add_argument('--prefix', default=None,
                        help='OOF file prefix (default: _autogluon_{preset})')
    parser.add_argument('--exclude', nargs='*', default=None,
                        help='Model types to exclude (e.g., NN_TORCH FASTAI)')
    args = parser.parse_args()

    prefix = args.prefix or f'_autogluon_{args.preset}'

    oof, test, ba = run_autogluon(
        preset=args.preset,
        time_limit=args.time_limit,
        save_prefix=prefix,
        excluded_model_types=args.exclude,
    )

    print(f'\nDone. OOF BA={ba:.5f}', flush=True)
