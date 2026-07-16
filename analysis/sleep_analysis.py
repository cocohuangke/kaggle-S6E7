"""Analyze sleep_duration distribution and thresholds."""
import pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
target_map = {'at-risk': 0, 'fit': 1, 'unhealthy': 2}
y = train['health_condition'].map(target_map).values

s = train['sleep_duration']
print('sleep_duration stats:')
print(f'  NaN: {s.isna().sum()} ({s.isna().mean()*100:.1f}%)')
print(f'  min={s.min():.1f}, max={s.max():.1f}, median={s.median():.1f}, mean={s.mean():.2f}')
print(f'  unique values: {s.dropna().nunique()}')
print()

# Distribution by target
for cls, name in enumerate(['at-risk','fit','unhealthy']):
    mask = y == cls
    vals = s[mask]
    print(f'  {name}: mean={vals.mean():.2f}, median={vals.median():.1f}, std={vals.std():.2f}, NaN={vals.isna().mean()*100:.1f}%')

print()
print('Distribution by 1-hour buckets:')
print(f'  {"Bucket":<12s} {"Total":>8s} {"%total":>7s} {"at-risk%":>9s} {"fit%":>9s} {"unhealthy%":>11s}')
print('  ' + '-'*55)
bins = list(range(0, 13))
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (s >= lo) & (s < hi)
    n = mask.sum()
    if n > 100:
        pcts = [(y[mask] == c).mean()*100 for c in range(3)]
        print(f'  [{lo:2d},{hi:2d})       {n:>8d} {n/len(s):>6.2f}%  {pcts[0]:>8.1f}% {pcts[1]:>8.1f}% {pcts[2]:>10.1f}%')

# Edge cases
mask_top = s >= 12
if mask_top.sum() > 0:
    n = mask_top.sum()
    pcts = [(y[mask_top] == c).mean()*100 for c in range(3)]
    print(f'  [12,inf)     {n:>8d} {n/len(s):>6.2f}%  {pcts[0]:>8.1f}% {pcts[1]:>8.1f}% {pcts[2]:>10.1f}%')

print()
print('Threshold analysis:')
for t in [5, 6, 7, 8]:
    for op, op_fn in [('<', lambda x: x < t), ('>=', lambda x: x >= t)]:
        mask = op_fn(s)
        n = mask.sum()
        if n > 100:
            pcts = [(y[mask] == c).mean()*100 for c in range(3)]
            print(f'  sleep {op} {t}: n={n:>6d}  at-risk={pcts[0]:.1f}%  fit={pcts[1]:.1f}%  unhealthy={pcts[2]:.1f}%')

print()
print('Naive BA for different discretizations:')
for scheme_name, fn in [
    ('2-way(<6,>=6)', lambda x: (x < 6).astype(str)),
    ('2-way(<7,>=7)', lambda x: (x < 7).astype(str)),
    ('3-way(<6,6-7,>=7)', lambda x: pd.cut(x, bins=[-1,6,7,100], labels=['lt6','6to7','ge7'], include_lowest=True)),
    ('3-way(<6,6-8,>=8)', lambda x: pd.cut(x, bins=[-1,6,8,100], labels=['lt6','6to8','ge8'], include_lowest=True)),
    ('4-way(<6,6-7,7-8,>=8)', lambda x: pd.cut(x, bins=[-1,6,7,8,100], labels=['lt6','6to7','7to8','ge8'], include_lowest=True)),
]:
    cats = fn(s).astype(str)
    # Naive BA: predict majority class per bucket
    encoded = pd.Categorical(cats).codes
    preds = np.zeros(len(s), dtype=int)
    for code in np.unique(encoded):
        m = encoded == code
        if m.sum() > 0:
            preds[m] = pd.Series(y[m]).mode().iloc[0]
    ba = balanced_accuracy_score(y, preds)
    print(f'  {scheme_name:<35s} Naive BA={ba:.5f}')
