"""Compare ref_realmlp vs refactored_realmlp outputs"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

# Check submissions
ref_sub = pd.read_csv('output/ref_realmlp_submission.csv')
ract_sub = pd.read_csv('output/refactored_realmlp_submission.csv')

print('ref_realmlp_submission:')
print(f'  shape: {ref_sub.shape}')
print(f'  value_counts: {dict(ref_sub.health_condition.value_counts())}')

print('\nrefactored_realmlp_submission:')
print(f'  shape: {ract_sub.shape}')
print(f'  value_counts: {dict(ract_sub.health_condition.value_counts())}')

# Test predictions comparison
ref_test = np.load('output/ref_realmlp_test.npy')
ract_test = np.load('output/refactored_realmlp_test.npy')
print(f'\nref_test shape: {ref_test.shape}, sum check: {ref_test.sum(axis=1)[:3]}')
print(f'ract_test shape: {ract_test.shape}, sum check: {ract_test.sum(axis=1)[:3]}')

# Per-class correlation
for i, cls in enumerate(['at-risk', 'fit', 'unhealthy']):
    corr = np.corrcoef(ref_test[:, i], ract_test[:, i])[0, 1]
    sp, _ = spearmanr(ref_test[:, i], ract_test[:, i])
    print(f'  Class {cls}: pearson={corr:.6f}, spearman={sp:.6f}')

# Argmax agreement
ref_argmax = np.argmax(ref_test, axis=1)
ract_argmax = np.argmax(ract_test, axis=1)
agree = (ref_argmax == ract_argmax).mean()
n_disagree = (ref_argmax != ract_argmax).sum()
print(f'\nArgmax agreement: {agree:.6f} ({n_disagree} disagreements out of {len(ref_argmax)})')

# Show disagreement distribution
if n_disagree > 0:
    from collections import Counter
    pairs = [(ref_argmax[i], ract_argmax[i]) for i in range(len(ref_argmax)) if ref_argmax[i] != ract_argmax[i]]
    cls_names = {0: 'at-risk', 1: 'fit', 2: 'unhealthy'}
    pair_counts = Counter(pairs)
    print('Disagreement pairs (ref -> ract):')
    for (a, b), cnt in pair_counts.most_common():
        print(f'  {cls_names[a]} -> {cls_names[b]}: {cnt}')

# OOF comparison
ref_oof = np.load('output/ref_realmlp_oof.npy')
ract_oof = np.load('output/refactored_realmlp_oof.npy')
train = pd.read_csv('data/train.csv')
y = train['health_condition'].map({'at-risk': 0, 'fit': 1, 'unhealthy': 2}).values

print(f'\nref OOF BA: {balanced_accuracy_score(y, np.argmax(ref_oof, axis=1)):.5f}')
print(f'ract OOF BA: {balanced_accuracy_score(y, np.argmax(ract_oof, axis=1)):.5f}')

print('OOF correlation per class:')
for i, cls in enumerate(['at-risk', 'fit', 'unhealthy']):
    corr = np.corrcoef(ref_oof[:, i], ract_oof[:, i])[0, 1]
    print(f'  Class {cls}: pearson={corr:.6f}')

# Max probability difference
print(f'\nMax test prob diff: {np.abs(ref_test - ract_test).max():.6f}')
print(f'Mean test prob diff: {np.abs(ref_test - ract_test).mean():.6f}')
print(f'Max OOF prob diff: {np.abs(ref_oof - ract_oof).max():.6f}')
print(f'Mean OOF prob diff: {np.abs(ref_oof - ract_oof).mean():.6f}')
