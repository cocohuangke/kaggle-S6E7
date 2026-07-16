"""Compare pipeline output with reference submission."""
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import balanced_accuracy_score

class_map = {'at-risk': 0, 'fit': 1, 'unhealthy': 2}
inv_map = {v: k for k, v in class_map.items()}

# 1. Compare submission
pipe_test = np.load('cache/blend/best_v1_test.npy')
ref = pd.read_csv('submissions/sub_rm_gb_a_blend_noca.csv')
pipe_pred = np.argmax(pipe_test, axis=1)
pipe_labels = np.array([inv_map[p] for p in pipe_pred])
ref_labels = ref['health_condition'].values
match = pipe_labels == ref_labels
print(f'Submission matches reference: {match.all()}')
print(f'Differences: {(~match).sum()} / {len(ref_labels)}')

# 2. Pipeline 5-fold model BAs
print('\n--- Pipeline 5-fold Model BAs ---')
for f in sorted(os.listdir('cache/model')):
    if f.endswith('_meta.json'):
        meta = json.load(open(os.path.join('cache/model', f)))
        m = meta['model']
        ba = meta['oof_ba']
        print(f'  {m} 5f BA: {ba:.5f}')

# 3. Pipeline 5-fold eq blend BA
y = np.array([class_map[v] for v in pd.read_csv('data/train.csv')['health_condition'].values])
oofs = []
for name in ['XGB', 'CB', 'HGB']:
    for f in os.listdir('cache/model'):
        if f.startswith(name) and f.endswith('_oof.npy'):
            oofs.append(np.load(os.path.join('cache/model', f)))
            break
eq_pipe = sum(oofs) / 3
ba_pipe_eq = balanced_accuracy_score(y, eq_pipe.argmax(1))
print(f'\nPipeline 5f eq BA: {ba_pipe_eq:.5f}')

# 4. Legacy 5-fold eq
eq_5f = np.load('oof/_fe_A_num2cat_eq_oof.npy')
ba_legacy_eq = balanced_accuracy_score(y, eq_5f.argmax(1))
print(f'Legacy 5f eq BA:   {ba_legacy_eq:.5f}')

# 5. Full blend comparison
rm = np.load('oof/_realmlp_oof.npy')
pipe_blend = 0.55 * rm + 0.45 * eq_pipe
legacy_blend = 0.55 * rm + 0.45 * eq_5f
print(f'\nPipeline blend BA: {balanced_accuracy_score(y, pipe_blend.argmax(1)):.5f}')
print(f'Legacy blend BA:   {balanced_accuracy_score(y, legacy_blend.argmax(1)):.5f}')

# 6. OOF difference between pipeline eq and legacy eq
diff = np.abs(eq_pipe - eq_5f)
print(f'\nPipeline eq vs Legacy eq OOF max diff: {diff.max():.6f}')
print(f'Pipeline eq vs Legacy eq OOF mean diff: {diff.mean():.6f}')
