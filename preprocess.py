#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


DATASET_CFG = {
    'assist2009': {
        'preprocess_script': os.path.join('Data', 'preprocess_assist09.py'),
        'intermediate_dir': os.path.join('Data', 'data', 'ASSIST09'),
        'output_dir': os.path.join('datasets', 'assist2009'),
        'dataset_name': 'assist2009',
    },
    'assist2017': {
        'preprocess_script': os.path.join('Data', 'preprocess_assist17.py'),
        'intermediate_dir': os.path.join('Data', 'data', 'ASSIST17'),
        'output_dir': os.path.join('datasets', 'assist2017'),
        'dataset_name': 'assist2017',
    },
    'statics2011': {
        'preprocess_script': os.path.join('Data', 'preprocess_statics2011.py'),
        'intermediate_dir': os.path.join('Data', 'data', 'STATICS2011'),
        'output_dir': os.path.join('datasets', 'statics2011'),
        'dataset_name': 'statics2011',
    },
    'xes3g5m': {
        'preprocess_script': os.path.join('Data', 'preprocess_xes3g5m.py'),
        'intermediate_dir': os.path.join('Data', 'data', 'XES3G5M'),
        'output_dir': os.path.join('datasets', 'xes3g5m'),
        'dataset_name': 'xes3g5m',
    },
}

# Backward-compatible aliases
DATASET_ALIASES = {
    'assist09': 'assist2009',
    'assist17': 'assist2017',
}


def run(cmd):
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Step 1: unified preprocessing for PEBG pipeline')
    parser.add_argument(
        '--dataset',
        required=True,
        choices=sorted(list(DATASET_CFG.keys()) + list(DATASET_ALIASES.keys())),
    )
    parser.add_argument('--python', default=sys.executable, help='Python interpreter path')
    parser.add_argument('--max-len', type=int, default=200)
    parser.add_argument('--min-len', type=int, default=3)
    parser.add_argument('--split', choices=['all', 'train'], default='all')
    args = parser.parse_args()

    canonical_dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    cfg = DATASET_CFG[canonical_dataset]
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 1) Raw dataset -> intermediate format
    run([args.python, cfg['preprocess_script']])

    # 2) Intermediate -> PEBG format
    run([
        args.python,
        'prepare_dataset.py',
        '--input-dir', cfg['intermediate_dir'],
        '--output-dir', cfg['output_dir'],
        '--dataset-name', cfg['dataset_name'],
        '--max-len', str(args.max_len),
        '--min-len', str(args.min_len),
        '--split', args.split,
    ])

    # 3) Build graph similarity files
    run([args.python, 'extract.py', '--data-dir', cfg['output_dir']])

    print('\nStep 1 finished.')
    print(f"Dataset: {canonical_dataset}")
    print(f"Prepared dir: {cfg['output_dir']}")


if __name__ == '__main__':
    main()
