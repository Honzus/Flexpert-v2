#!/usr/bin/env python3
"""Compute per-protein Spearman/Pearson correlations for one or more Flexpert prediction files
vs. ground-truth RMSF labels, and print a comparison table.

Usage:
    python3 evaluate_ablations.py \
        --data_path data/rmsf_atlas_data_prottransready.txt \
        --splits_path data/atlas_splits.json \
        --pred_files "ProT5:prediction_results/foo.txt" "ESM2-650M:prediction_results/bar.txt"
"""
import os
import json
import argparse
import numpy as np
from scipy.stats import spearmanr, pearsonr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='Ground-truth RMSF file (NAME:\\tval1, val2, ... format)')
    parser.add_argument('--splits_path', required=True,
                        help='Path to *_splits.json')
    parser.add_argument('--split', default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--pred_files', nargs='+', required=True,
                        help='One or more "ModelName:path/to/pred.txt" pairs')
    return parser.parse_args()


def normalize_name(name):
    """Convert dot-separated PDB names to underscore (e.g. 1abc.A → 1abc_A)."""
    return name.replace('.', '_')


def load_ground_truth(data_path):
    """Return dict mapping normalized name → list of float RMSF values."""
    gt = {}
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name_part, vals_part = line.split(':\t', 1)
            name = normalize_name(name_part.strip())
            vals = [float(v) for v in vals_part.split(', ')]
            gt[name] = np.array(vals, dtype=np.float32)
    return gt


def load_predictions(pred_path):
    """Return dict mapping normalized name → numpy array of predicted values."""
    preds = {}
    with open(pred_path) as f:
        lines = f.readlines()
    for header, vals_line in zip(lines[::2], lines[1::2]):
        name = normalize_name(header.strip().lstrip('>'))
        vals = np.array(vals_line.strip().split(', '), dtype=np.float32)
        preds[name] = vals
    return preds


def compute_correlations(gt, preds, test_names):
    """Compute per-protein Spearman and Pearson for proteins in test_names."""
    spearmans, pearsons = [], []
    skipped = 0
    for name in test_names:
        norm_name = normalize_name(name)
        if norm_name not in gt or norm_name not in preds:
            skipped += 1
            continue
        g = gt[norm_name]
        p = preds[norm_name]
        # Align lengths (truncate to shorter)
        n = min(len(g), len(p))
        if n < 5:
            skipped += 1
            continue
        g, p = g[:n], p[:n]
        # Filter invalid ground truth (>900 are sentinel missing values)
        valid = g < 900
        if valid.sum() < 5:
            skipped += 1
            continue
        g, p = g[valid], p[valid]
        rho, _ = spearmanr(g, p)
        r, _   = pearsonr(g, p)
        spearmans.append(rho)
        pearsons.append(r)
    return np.array(spearmans), np.array(pearsons), skipped


def main():
    args = parse_args()

    # Load splits
    with open(args.splits_path) as f:
        splits = json.load(f)
    test_names = splits[args.split]
    print(f'Test set: {len(test_names)} proteins from {args.splits_path}')

    # Load ground truth
    gt = load_ground_truth(args.data_path)
    print(f'Ground truth: {len(gt)} proteins from {args.data_path}\n')

    # Parse model name:path pairs
    models = []
    for entry in args.pred_files:
        if ':' not in entry:
            raise ValueError(f'Expected "ModelName:path" format, got: {entry}')
        model_name, pred_path = entry.split(':', 1)
        models.append((model_name, pred_path))

    # Compute and print table
    header = f'{"Model":<22}  {"N":>5}  {"Spearman":>18}  {"Pearson":>18}  {"Skipped":>8}'
    print(header)
    print('-' * len(header))

    for model_name, pred_path in models:
        if not os.path.exists(pred_path):
            print(f'{model_name:<22}  {"—":>5}  {"prediction file not found":>40}')
            continue
        preds = load_predictions(pred_path)
        spearmans, pearsons, skipped = compute_correlations(gt, preds, test_names)
        n = len(spearmans)
        if n == 0:
            print(f'{model_name:<22}  {"0":>5}  {"no matching proteins":>40}')
            continue
        sp_mean, sp_std = spearmans.mean(), spearmans.std()
        pe_mean, pe_std = pearsons.mean(), pearsons.std()
        print(f'{model_name:<22}  {n:>5}  {sp_mean:.4f} ± {sp_std:.4f}  {pe_mean:.4f} ± {pe_std:.4f}  {skipped:>8}')

    print()


if __name__ == '__main__':
    main()
