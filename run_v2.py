#!/usr/bin/env python
"""Unified pipeline entry point for S6E7.

Usage:
    python run_v2.py configs/champion_v1.yaml
    python run_v2.py configs/best_9d.yaml

Flow: config → resolve FE → train/load models → blend → submission
All outputs trace back to the config file name.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.blend import run_blend

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_v2.py <config.yaml>', flush=True)
        print('Available configs:', flush=True)
        config_dir = os.path.join(os.path.dirname(__file__), 'configs')
        for f in sorted(os.listdir(config_dir)):
            if f.endswith('.yaml') or f.endswith('.yml'):
                print(f'  configs/{f}', flush=True)
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    if not os.path.exists(config_path):
        print(f'Config not found: {config_path}', flush=True)
        sys.exit(1)

    ba, oof, test = run_blend(config_path)
    print(f'\nDone. OOF BA={ba:.5f}', flush=True)
