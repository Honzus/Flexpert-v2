#!/usr/bin/env python3
"""Check that predictions from the released Flexpert weights reproduce the reference test metrics.

Expects the prediction files written by ``slurm/verify_core.sh`` (or by ``run_inference.py`` with
the same naming): ``<pred_dir>/<model>_<dataset>_test.txt`` with ``<model>`` in
{flexpert_seq, flexpert_3d, flexpert_seq_prott5} plus an optional ``_mdcath`` / ``_combined``
suffix (the weight file stem without ``_prostt5_aa``/``_saprot``/``_weights``) and ``<dataset>``
in {atlas, mdcath, combined}: 9 models x 3 test sets = 27 cells. Missing files and cells without
a reference value (NOREF) are reported, not fatal, unless ``--require_all`` is set.

Run from the repository root:
    python3 -m evaluation.check_reproduction [--pred_dir outputs/predictions] [--tol 0.005]
        [--reference_dir <dir of older prediction files> ...] [--require_all]

Exit status 0 if every checked mean Spearman/Pearson is within ``--tol`` of the reference.
"""
import os
import json
import argparse
import numpy as np

from evaluation.evaluate_ablations import load_ground_truth, load_predictions, compute_correlations

DATASETS = {  # name -> (labels, splits)
    'atlas': ('data/rmsf_atlas_data_prottransready.txt', 'data/atlas_splits.json'),
    'mdcath': ('data/rmsf_mdCATH_data.txt', 'data/mdCATH_splits.json'),
    'combined': ('data/rmsf_combined_data.txt', 'data/combined_splits.json'),
}

# Mean per-chain (Spearman, Pearson) on each test split, released HF weights (Honzus24/Flexpert-v2).
REFERENCE = {
    'flexpert_seq':          {'atlas': (0.7142, 0.7794), 'mdcath': (0.6678, 0.7197), 'combined': (0.6775, 0.7321)},
    'flexpert_3d':           {'atlas': (0.8240, 0.8748), 'mdcath': (0.8004, 0.8476), 'combined': (0.8053, 0.8533)},
    'flexpert_seq_mdcath':   {'atlas': (0.7254, 0.7825), 'mdcath': (0.7759, 0.8086), 'combined': (0.7654, 0.8032)},
    'flexpert_3d_mdcath':    {'atlas': (0.8110, 0.8438), 'mdcath': (0.8545, 0.8907), 'combined': (0.8455, 0.8810)},
    'flexpert_seq_combined': {'atlas': (0.7553, 0.8046), 'mdcath': (0.7735, 0.8125), 'combined': (0.7697, 0.8109)},
    'flexpert_3d_combined':  {'atlas': (0.8445, 0.8800), 'mdcath': (0.8522, 0.8885), 'combined': (0.8506, 0.8868)},
    'flexpert_seq_prott5':          {'atlas': (0.6907, 0.7607), 'mdcath': (0.6772, 0.7255), 'combined': (0.6800, 0.7328)},
    'flexpert_seq_prott5_mdcath':   {'atlas': (0.6815, 0.7541), 'mdcath': (0.7457, 0.7839), 'combined': (0.7323, 0.7777)},
    'flexpert_seq_prott5_combined': {'atlas': (0.7116, 0.7798), 'mdcath': (0.7489, 0.7803), 'combined': (0.7411, 0.7802)},
}

# Pre-release prediction files (internal naming) used only for the optional per-residue diff;
# {test} is Atlas / MDCATH / Combined.
_REF_FILE = {
    'flexpert_seq': 'verify_atlasProstT5AA_test{test}.txt',
    'flexpert_3d': 'verify_atlasSAprot_test{test}.txt',
    'flexpert_seq_mdcath': 'verify_mdcathProstT5AA_test{test}.txt',
    'flexpert_3d_mdcath': 'verify_mdcathSAprot_test{test}.txt',
    'flexpert_seq_combined': 'verify_combinedProstT5AA_test{test}.txt',
    'flexpert_3d_combined': 'verify_combinedSAprot_test{test}.txt',
    'flexpert_seq_prott5': 'results_atlasFlexpertSeq_test{test}.txt',
    'flexpert_seq_prott5_mdcath': 'results_mdcathFlexpertSeq_test{test}.txt',
    'flexpert_seq_prott5_combined': 'results_combinedFlexpertSeq_test{test}.txt',
}
_REF_TEST = {'atlas': 'Atlas', 'mdcath': 'MDCATH', 'combined': 'Combined'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pred_dir', default='outputs/predictions')
    p.add_argument('--tol', type=float, default=0.005)
    p.add_argument('--reference_dir', action='append', default=[],
                   help='Optional dir with pre-release prediction files (verify_<run>_test<Dataset>.txt '
                        'or results_<train>FlexpertSeq_test<Dataset>.txt); reports the max '
                        'per-residue |difference| to them. May be given more than once.')
    p.add_argument('--require_all', action='store_true',
                   help='Fail if any of the 27 prediction files is missing or has no reference.')
    p.add_argument('--json_out', default=None, help='Optional path for a JSON summary.')
    return p.parse_args()


def max_abs_diff(new, old):
    worst = 0.0
    for name, vals in new.items():
        if name in old:
            a, b = np.asarray(vals, dtype=float), np.asarray(old[name], dtype=float)
            n = min(len(a), len(b))
            if n:
                worst = max(worst, float(np.max(np.abs(a[:n] - b[:n]))))
    return worst


def _fmt(v):
    return '-' if v is None else f'{v:.4f}'


def main():
    args = parse_args()
    rows, failures, missing, noref = [], 0, 0, 0
    for ds, (labels, splits) in DATASETS.items():
        gt = load_ground_truth(labels)
        with open(splits) as f:
            test_names = json.load(f)['test']
        for model, ref in REFERENCE.items():
            path = os.path.join(args.pred_dir, f'{model}_{ds}_test.txt')
            if not os.path.exists(path):
                missing += 1
                rows.append({'model': model, 'dataset': ds, 'status': 'MISSING', 'path': path})
                continue
            preds = load_predictions(path)
            sp, pe, skipped = compute_correlations(gt, preds, test_names)
            row = {'model': model, 'dataset': ds, 'n': int(len(sp)), 'skipped': int(skipped),
                   'spearman': round(float(sp.mean()), 4), 'pearson': round(float(pe.mean()), 4),
                   'ref_spearman': None, 'ref_pearson': None}
            if ref[ds] is None:
                row['status'] = 'NOREF'
                noref += 1
            else:
                ref_sp, ref_pe = ref[ds]
                ok = (abs(sp.mean() - ref_sp) <= args.tol and abs(pe.mean() - ref_pe) <= args.tol
                      and not skipped)
                row.update(ref_spearman=ref_sp, ref_pearson=ref_pe, status='PASS' if ok else 'FAIL')
                failures += not ok
            for ref_dir in args.reference_dir:
                old = os.path.join(ref_dir, _REF_FILE[model].format(test=_REF_TEST[ds]))
                if os.path.exists(old):
                    row['max_abs_diff_vs_reference'] = round(max_abs_diff(preds, load_predictions(old)), 6)
                    break
            rows.append(row)

    print(f'{"model":<30}{"dataset":<10}{"N":>5}{"Spearman":>10}{"ref":>8}{"Pearson":>10}{"ref":>8}'
          f'{"max|d|":>10}  status')
    for r in rows:
        if r['status'] == 'MISSING':
            print(f'{r["model"]:<30}{r["dataset"]:<10}  MISSING  {r["path"]}')
            continue
        d = r.get('max_abs_diff_vs_reference')
        skipped = f' ({r["skipped"]} chain(s) skipped)' if r['skipped'] else ''
        print(f'{r["model"]:<30}{r["dataset"]:<10}{r["n"]:>5}{r["spearman"]:>10.4f}{_fmt(r["ref_spearman"]):>8}'
              f'{r["pearson"]:>10.4f}{_fmt(r["ref_pearson"]):>8}{"" if d is None else f"{d:.2e}":>10}'
              f'  {r["status"]}{skipped}')
    checked = len(rows) - missing - noref
    print(f'RESULT: {checked - failures}/{checked} within ±{args.tol}; {missing} missing; '
          f'{noref} without reference (NOREF)')
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rows, f, indent=1)
    bad = failures or checked == 0 or (args.require_all and (missing or noref))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
