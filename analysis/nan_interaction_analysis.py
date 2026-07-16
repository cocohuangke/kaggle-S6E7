#!/usr/bin/env python
"""NaN Joint Missing Value Analysis for S6E7.

Analyzes pairwise NaN interaction patterns in the training data.
For each feature pair, computes:
  1. Joint NaN pattern frequencies (4 categories: both-present, only-A-NaN, only-B-NaN, both-NaN)
  2. Target distribution per pattern
  3. Cramér's V (association strength between NaN pattern and target)
  4. Mutual Information between NaN pattern and target

Purpose: identify NaN-driven interactions that could become useful explicit features
for tree-based models (capturing information lost after fill_missing).

Output:
  - Console summary: top-ranked pairs
  - analysis/nan_interaction_top.csv: full ranked table
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import chi2_contingency

np.set_printoptions(suppress=True, precision=6)

# === Paths ===
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, 'data')
OUT_DIR = os.path.join(BASE, 'analysis')

# === Load data ===
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
TARGET = 'health_condition'

# Separate features (exclude id and target)
id_col = 'id'
feature_cols = [c for c in train.columns if c not in (id_col, TARGET)]

# === Configuration ===
# Min NaN rate for a feature to be considered
MIN_NAN_RATE = 0.03  # 3%
# Min joint pattern frequency to be considered meaningful
MIN_PATTERN_FREQ = 0.005  # 0.5% of total data

# === Compute NaN rates and filter features ===
nan_rates = {}
for col in feature_cols:
    rate = train[col].isna().mean()
    nan_rates[col] = rate
    print(f"  {col:25s}: NaN={rate:.4f} ({rate*100:.2f}%)")

# Filter features with meaningful NaN rate
candidate_cols = [c for c in feature_cols if nan_rates[c] >= MIN_NAN_RATE]
print(f"\n  Focus on {len(candidate_cols)} features with NaN >= {MIN_NAN_RATE*100:.0f}%\n")

# === Target encoding ===
target_map = {'at-risk': 0, 'fit': 1, 'unhealthy': 2}
y = train[TARGET].map(target_map).values

print('=' * 90)
print(f"{'Feature A':<22s} {'Feature B':<22s} {'both-OK':>7s} {'A-NaN':>7s} {'B-NaN':>7s} {'both-NaN':>7s} {'CramV':>6s} {'MI':>6s} {'BestBA':>7s}")
print('=' * 90)

results = []

for i, col_a in enumerate(candidate_cols):
    for j, col_b in enumerate(candidate_cols):
        if j <= i:
            continue  # only upper triangle

        # Create NaN pattern labels (4 categories)
        na_a = train[col_a].isna()
        na_b = train[col_b].isna()

        pattern = np.full(len(train), 'both-OK', dtype=object)
        pattern[na_a & ~na_b] = f'{col_a[:6]}-NaN'
        pattern[~na_a & na_b] = f'{col_b[:6]}-NaN'
        pattern[na_a & na_b] = 'both-NaN'

        # Pattern frequencies
        pattern_counts = pd.Series(pattern).value_counts()
        n_both_ok = pattern_counts.get('both-OK', 0)
        n_a_nan = pattern_counts.get(f'{col_a[:6]}-NaN', 0)
        n_b_nan = pattern_counts.get(f'{col_b[:6]}-NaN', 0)
        n_both_na = pattern_counts.get('both-NaN', 0)
        total = len(train)

        both_ok_pct = n_both_ok / total
        a_nan_pct = n_a_nan / total
        b_nan_pct = n_b_nan / total
        both_na_pct = n_both_na / total

        # Skip if any pattern is too rare to be meaningful
        if both_na_pct < MIN_PATTERN_FREQ and a_nan_pct < MIN_PATTERN_FREQ and b_nan_pct < MIN_PATTERN_FREQ:
            continue

        # Target distribution per pattern
        pattern_targets = {}
        for p_label, mask in [
            ('both-OK', ~na_a & ~na_b),
            (f'{col_a[:6]}-NaN', na_a & ~na_b),
            (f'{col_b[:6]}-NaN', ~na_a & na_b),
            ('both-NaN', na_a & na_b),
        ]:
            if mask.sum() > 0:
                dist = pd.Series(y[mask]).value_counts(normalize=True).sort_index()
                pattern_targets[p_label] = dist

        # Cramér's V
        try:
            contingency = pd.crosstab(pattern, y)
            chi2, p, dof, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * min(contingency.shape[0] - 1, contingency.shape[1] - 1)))
        except Exception:
            cramers_v = 0.0

        # Mutual Information (approximation via normalized entropy)
        try:
            from sklearn.metrics import mutual_info_score
            mi = mutual_info_score(pattern, y)
            # normalize by entropy of target
            from scipy.stats import entropy
            h_target = entropy(np.bincount(y) / len(y))
            mi_norm = mi / h_target if h_target > 0 else 0
        except Exception:
            mi_norm = 0.0

        # Best balanced accuracy achievable by just using NaN pattern as predictor
        # (naive: predict most common class per NaN pattern)
        pattern_codes = pd.Categorical(pattern).codes
        preds = np.zeros(len(train), dtype=int)
        unique_patterns = np.unique(pattern_codes)
        for p_code in unique_patterns:
            mask = pattern_codes == p_code
            if mask.sum() > 0:
                # predict most frequent class in this pattern
                mode_val = pd.Series(y[mask]).mode()
                if len(mode_val) > 0:
                    preds[mask] = mode_val.iloc[0]
        ba_naive = balanced_accuracy_score(y, preds)

        print(f"  {col_a:<22s} {col_b:<22s} {both_ok_pct:6.1%} {a_nan_pct:6.1%} {b_nan_pct:6.1%} {both_na_pct:6.1%} {cramers_v:6.4f} {mi_norm:6.4f} {ba_naive:7.5f}")

        results.append({
            'feature_a': col_a,
            'feature_b': col_b,
            'nan_rate_a': nan_rates[col_a],
            'nan_rate_b': nan_rates[col_b],
            'both_ok_pct': both_ok_pct,
            'a_nan_only_pct': a_nan_pct,
            'b_nan_only_pct': b_nan_pct,
            'both_nan_pct': both_na_pct,
            'cramers_v': cramers_v,
            'mi_norm': mi_norm,
            'naive_ba': ba_naive,
            'pattern_targets': {k: v.to_dict() for k, v in pattern_targets.items()},
        })

# === Sort and save ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mi_norm', ascending=False)

print(f"\n{'=' * 90}")
print(f"\nTop 10 NaN Interactions by Mutual Information:")
print(f"{'Rank':<5s} {'Feature A':<22s} {'Feature B':<22s} {'MI':>6s} {'CramV':>6s} {'both-NaN%':>8s} {'BestBA':>7s}")
print('-' * 85)
for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"  {rank:<4d} {row['feature_a']:<22s} {row['feature_b']:<22s} {row['mi_norm']:6.4f} {row['cramers_v']:6.4f} {row['both_nan_pct']:7.2%} {row['naive_ba']:7.5f}")

# Save full results
out_path = os.path.join(OUT_DIR, 'nan_interaction_top.csv')
results_df.to_csv(out_path, index=False)
print(f"\n  Full results saved: {out_path}")

# === Bonus: detailed target distribution for top 5 ===
print(f"\n{'=' * 90}")
print("Detailed target distribution for Top 5 pairs:")
print('=' * 90)
target_names = {0: 'at-risk', 1: 'fit', 2: 'unhealthy'}
for rank, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
    print(f"\n  ** Rank {rank}: {row['feature_a']} × {row['feature_b']} (MI={row['mi_norm']:.4f}, CramV={row['cramers_v']:.4f})")
    pts = row['pattern_targets']
    for pattern_label in ['both-OK', f'{row["feature_a"][:6]}-NaN', f'{row["feature_b"][:6]}-NaN', 'both-NaN']:
        if pattern_label in pts:
            dist = pts[pattern_label]
            dist_str = ', '.join(f"{target_names.get(k, k)}={v:.1%}" for k, v in sorted(dist.items()))
            freq = 0
            if pattern_label == 'both-OK': freq = row['both_ok_pct']
            elif pattern_label == f'{row["feature_a"][:6]}-NaN': freq = row['a_nan_only_pct']
            elif pattern_label == f'{row["feature_b"][:6]}-NaN': freq = row['b_nan_only_pct']
            elif pattern_label == 'both-NaN': freq = row['both_nan_pct']
            print(f"    {pattern_label:12s} ({freq:.2%}): [{dist_str}]")

print("\nDone.")
