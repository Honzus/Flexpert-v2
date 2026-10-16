#!/usr/bin/env python3
"""Compute per-protein Spearman/Pearson correlations for one or more Flexpert prediction files
vs. ground-truth RMSF labels, and print a comparison table (mean ± std over chains).

Run from the repository root:
    python3 -m evaluation.evaluate_ablations \
        --data_path data/rmsf_atlas_data_prottransready.txt \
        --splits_path data/atlas_splits.json \
        --pred_files "Flexpert-Seq:outputs/predictions/flexpert_seq_atlas_test.txt" \
                     "Flexpert-3D:outputs/predictions/flexpert_3d_atlas_test.txt" \
        [--per_chain_out outputs/predictions/atlas_test_per_chain.csv]

``--per_chain_out`` writes one row per scored chain with columns
``chain,n_res,spearman_<model>,pearson_<model>,...`` where ``<model>`` is the model name
lower-cased with non-alphanumerics replaced by ``_`` (``Flexpert-Seq`` -> ``flexpert_seq``).
"""
import os
import re
import csv
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
    parser.add_argument('--per_chain_out', default=None,
                        help='Optional CSV with per-chain correlations '
                             '(chain,n_res,spearman_<model>,pearson_<model>,...).')
    return parser.parse_args()


def model_slug(model_name):
    """'Flexpert-Seq' -> 'flexpert_seq' (column suffix in the --per_chain_out CSV)."""
    return re.sub(r'[^0-9a-z]+', '_', model_name.lower()).strip('_')


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


def compute_correlations(gt, preds, test_names, per_chain=None):
    """Compute per-protein Spearman and Pearson for proteins in test_names.

    If ``per_chain`` is a dict, it is filled with ``{name: (n_res, rho, r)}`` for every scored chain.
    """
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
        if per_chain is not None:
            per_chain[name] = (int(valid.sum()), rho, r)
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

    per_chain_tables = []   # (slug, {chain: (n_res, rho, r)})
    for model_name, pred_path in models:
        if not os.path.exists(pred_path):
            print(f'{model_name:<22}  {"—":>5}  {"prediction file not found":>40}')
            continue
        preds = load_predictions(pred_path)
        per_chain = {}
        spearmans, pearsons, skipped = compute_correlations(gt, preds, test_names, per_chain)
        per_chain_tables.append((model_slug(model_name), per_chain))
        n = len(spearmans)
        if n == 0:
            print(f'{model_name:<22}  {"0":>5}  {"no matching proteins":>40}')
            continue
        sp_mean, sp_std = spearmans.mean(), spearmans.std()
        pe_mean, pe_std = pearsons.mean(), pearsons.std()
        print(f'{model_name:<22}  {n:>5}  {sp_mean:.4f} ± {sp_std:.4f}  {pe_mean:.4f} ± {pe_std:.4f}  {skipped:>8}')

    print()

    if args.per_chain_out:
        write_per_chain_csv(args.per_chain_out, test_names, per_chain_tables)
        print(f'Per-chain correlations written to {args.per_chain_out}')


def write_per_chain_csv(path, test_names, per_chain_tables):
    """One row per chain scored by at least one model (split order); blanks where unscored."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = ['chain', 'n_res']
    for slug, _ in per_chain_tables:
        header += [f'spearman_{slug}', f'pearson_{slug}']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for name in test_names:
            hits = [table.get(name) for _, table in per_chain_tables]
            if all(h is None for h in hits):
                continue
            row = [name, next(h[0] for h in hits if h is not None)]
            for h in hits:
                row += ['', ''] if h is None else [repr(float(h[1])), repr(float(h[2]))]
            w.writerow(row)


if __name__ == '__main__':
    main()
