"""Blend AutoGluon OOF with champion OOF.

Usage:
    python analyze_ag_blend.py

Assumes AutoGluon has already been run and OOF files exist:
    oof/_autogluon_gq_oof.npy  (or _autogluon_bq_oof.npy)
    oof/_autogluon_gq_test.npy
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
OOF_DIR = os.path.join(ROOT, 'oof')
DATA_DIR = os.path.join(ROOT, 'data')
SUB_DIR = os.path.join(ROOT, 'submissions')

# Load ground truth
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train['health_condition'].values)
print(f'Classes: {le.classes_}')  # at-risk=0, fit=1, unhealthy=2

# Load champion OOF (LB=0.95083)
champ_oof = np.load(os.path.join(OOF_DIR, '_realmlp_ref_oof.npy'))
champ_test = np.load(os.path.join(OOF_DIR, '_realmlp_ref_oof.npy'.replace('_ref_oof', '_ref_test')))

# Load 4way_numte_blend OOF
blend_oof_4way = np.load(os.path.join(OOF_DIR, '_4way_numte_blend_oof.npy'))
blend_test_4way = np.load(os.path.join(OOF_DIR, '_4way_numte_blend_test.npy'))

# Reconstruct champion blend with per-class weights
# RM_ref=[0.60, 0.32, 0.60], GB_4way=[0.40, 0.68, 0.40]
rm_w = np.array([0.60, 0.32, 0.60])
gb_w = np.array([0.40, 0.68, 0.40])
champ_full_oof = champ_oof * rm_w + blend_oof_4way * gb_w
champ_full_test = champ_test * rm_w + blend_test_4way * gb_w
champ_ba = balanced_accuracy_score(y, champ_full_oof.argmax(1))
print(f'Champion OOF BA: {champ_ba:.5f} (expected 0.95086)')

# Find AutoGluon OOF file (try multiple prefixes)
ag_prefixes = ['_autogluon_gq', '_autogluon_bq', '_autogluon']
ag_oof = None
ag_test = None
ag_prefix_used = None

for prefix in ag_prefixes:
    oof_path = os.path.join(OOF_DIR, f'{prefix}_oof.npy')
    test_path = os.path.join(OOF_DIR, f'{prefix}_test.npy')
    if os.path.exists(oof_path) and os.path.exists(test_path):
        ag_oof = np.load(oof_path)
        ag_test = np.load(test_path)
        ag_prefix_used = prefix
        break

if ag_oof is None:
    print('\nERROR: No AutoGluon OOF files found!')
    print('Expected one of:')
    for p in ag_prefixes:
        print(f'  oof/{p}_oof.npy')
    print('\nPlease run AutoGluon first:')
    print('  python -u -m pipeline.autogluon --preset good_quality --time_limit 3600')
    exit(1)

print(f'\nAutoGluon prefix: {ag_prefix_used}')
print(f'AG OOF shape: {ag_oof.shape}, test shape: {ag_test.shape}')

# AG OOF BA
ag_ba = balanced_accuracy_score(y, ag_oof.argmax(1))
print(f'AG OOF BA: {ag_ba:.5f}')

# ============================================================
# Correlation Analysis
# ============================================================
print(f'\n{"="*60}')
print('Correlation: Champion vs AutoGluon (per class)')
print(f'{"="*60}')
for c, name in enumerate(le.classes_):
    corr = np.corrcoef(champ_full_oof[:, c], ag_oof[:, c])[0, 1]
    print(f'  {name:12s}: r = {corr:.6f}')

# Per-class disagreement analysis
champ_labels = champ_full_oof.argmax(1)
ag_labels = ag_oof.argmax(1)
agree = (champ_labels == ag_labels).mean()
print(f'\nAgreement: {agree:.4f} ({(champ_labels != ag_labels).sum()} disagreements)')

# Who's right on disagreements?
disagree_mask = champ_labels != ag_labels
champ_right = (champ_labels[disagree_mask] == y[disagree_mask]).sum()
ag_right = (ag_labels[disagree_mask] == y[disagree_mask]).sum()
print(f'  Champion right: {champ_right}, AG right: {ag_right}')

# ============================================================
# 2-way Uniform Grid Search
# ============================================================
print(f'\n{"="*60}')
print('2-way Grid Search: Champion + AG (uniform weights)')
print(f'{"="*60}')

best_ba = 0
best_w = 0
results = []
for w_ag in range(0, 51, 1):
    w_ag_f = w_ag / 100.0
    w_ch_f = 1.0 - w_ag_f
    blend = champ_full_oof * w_ch_f + ag_oof * w_ag_f
    ba = balanced_accuracy_score(y, blend.argmax(1))
    results.append((w_ag_f, ba))
    if ba > best_ba:
        best_ba = ba
        best_w = w_ag_f

print(f'Best: AG={best_w:.2f}, Champ={1-best_w:.2f}, OOF BA={best_ba:.5f}')
print(f'Champion alone: {champ_ba:.5f}')
print(f'AG alone: {ag_ba:.5f}')
delta = best_ba - champ_ba
print(f'Delta vs champion: {delta:+.5f}')

# Print top-5
results.sort(key=lambda x: -x[1])
print('\nTop-5:')
for w, ba in results[:5]:
    print(f'  AG={w:.2f} Champ={1-w:.2f} → BA={ba:.5f}')

# ============================================================
# 2-way Per-class Grid Search
# ============================================================
print(f'\n{"="*60}')
print('2-way Per-class Grid Search: Champion + AG')
print(f'{"="*60}')

best_pc_ba = 0
best_pc_w = None
step = 0.05

# Search AG weight per class [0.0, 0.05, ..., 0.50]
for w0 in np.arange(0, 0.51, step):
    for w1 in np.arange(0, 0.51, step):
        for w2 in np.arange(0, 0.51, step):
            ag_w_arr = np.array([w0, w1, w2])
            ch_w_arr = 1.0 - ag_w_arr
            blend = champ_full_oof * ch_w_arr + ag_oof * ag_w_arr
            ba = balanced_accuracy_score(y, blend.argmax(1))
            if ba > best_pc_ba:
                best_pc_ba = ba
                best_pc_w = ag_w_arr.copy()

print(f'Best per-class AG weights: {best_pc_w}')
print(f'Best per-class Champ weights: {1.0 - best_pc_w}')
print(f'Best OOF BA: {best_pc_ba:.5f}')
delta_pc = best_pc_ba - champ_ba
print(f'Delta vs champion: {delta_pc:+.5f}')

# ============================================================
# LB Prediction (v7 formula)
# ============================================================
print(f'\n{"="*60}')
print('LB Prediction (v7 formula)')
print(f'{"="*60}')

# For 2-way blend
blend_best = champ_full_oof * (1 - best_w) + ag_oof * best_w
health_uniform = max(champ_ba, ag_ba) - best_ba
predicted_lb_uniform = best_ba + 0.00049 + 1.227 * health_uniform
print(f'Uniform: predicted_LB = {predicted_lb_uniform:.5f} (OOF={best_ba:.5f})')

# For per-class
blend_pc = champ_full_oof * (1 - best_pc_w) + ag_oof * best_pc_w
health_pc = max(champ_ba, ag_ba) - best_pc_ba
predicted_lb_pc = best_pc_ba + 0.00049 + 1.227 * health_pc
print(f'Per-class: predicted_LB = {predicted_lb_pc:.5f} (OOF={best_pc_ba:.5f})')
print(f'Current best LB: 0.95083')

# ============================================================
# Generate Submissions
# ============================================================
os.makedirs(SUB_DIR, exist_ok=True)
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

# Best uniform
blend_test_uniform = champ_full_test * (1 - best_w) + ag_test * best_w
pred_labels_uniform = le.inverse_transform(blend_test_uniform.argmax(1))
sub_uniform = sample_sub.copy()
sub_uniform['health_condition'] = pred_labels_uniform
sub_path_uniform = os.path.join(SUB_DIR, f'sub_champ_ag{int(best_w*100):02d}.csv')
sub_uniform.to_csv(sub_path_uniform, index=False)
print(f'\nSaved uniform submission: {sub_path_uniform}')
print(f'Value counts: {sub_uniform.health_condition.value_counts().to_dict()}')

# Best per-class (if different from uniform)
if not np.allclose(best_pc_w, best_w):
    blend_test_pc = champ_full_test * (1 - best_pc_w) + ag_test * best_pc_w
    pred_labels_pc = le.inverse_transform(blend_test_pc.argmax(1))
    sub_pc = sample_sub.copy()
    sub_pc['health_condition'] = pred_labels_pc
    w_str = '_'.join([f'{int(w*100):02d}' for w in best_pc_w])
    sub_path_pc = os.path.join(SUB_DIR, f'sub_champ_ag_pc{w_str}.csv')
    sub_pc.to_csv(sub_path_pc, index=False)
    print(f'Saved per-class submission: {sub_path_pc}')
    print(f'Value counts: {sub_pc.health_condition.value_counts().to_dict()}')

# Also save AG solo submission for reference
ag_pred_labels = le.inverse_transform(ag_test.argmax(1))
sub_ag = sample_sub.copy()
sub_ag['health_condition'] = ag_pred_labels
sub_path_ag = os.path.join(SUB_DIR, f'sub_{ag_prefix_used.strip("_")}.csv')
sub_ag.to_csv(sub_path_ag, index=False)
print(f'Saved AG solo submission: {sub_path_ag}')

print(f'\nDone!')
