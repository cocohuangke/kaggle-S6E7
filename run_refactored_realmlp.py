"""Run refactored pipeline RealMLP (reference-matched, stress_pal=False).

Outputs saved automatically via oof_prefix:
  - oof/_realmlp_ref_oof.npy, oof/_realmlp_ref_test.npy
  - submissions/sub_realmlp_ref.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.train import train_realmlp

oof, test, ba = train_realmlp(
    n_splits=7,
    fe_config={'te': True, 'gp_features': True, 'stress_pal': False},
    oof_prefix='_realmlp_ref',
)

print(f"\nDone: OOF BA = {ba:.5f}")
