"""Figure 5c: average flexibility profile along the chain on the ATLAS test set.

Three series per chain: crystallographic B-factor and MD RMSF (ATLAS analyses,
``<dir>/<id>/<id>_{Bfactor,RMSF}.tsv``) and the Flexpert prediction. Each series is z-scored
within the chain (NaN-aware), residues of all chains are pooled and binned by relative position
i / (L - 1) (0 = N, 1 = C; 20 bins), and the curves are the per-bin mean ± SEM over residues.
Chains whose three series differ in length (e.g. predictions truncated at 1023 residues) or that
have no B-factors are skipped.

    python3 -m analysis.fig5_termini
    # -> analysis/figures/fig5c_termini.png, fig5c_termini.csv
"""
import argparse
import json
import os

import numpy as np

from analysis.plot_flexibility_universe import read_flexpert_predictions

SERIES = [  # key, legend label, colour (draw order)
    ('bfactor', 'crystallographic B-factor', 'gray'),
    ('rmsf', 'MD RMSF (ATLAS)', 'tab:orange'),
    ('flexpert', 'Flexpert prediction', 'tab:blue'),
]


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.nanmean(x)) / np.nanstd(x)


def relative_bins(length, n_bins):
    """Bin index of each residue by relative position i / (L - 1), 0 = N terminus, 1 = C terminus."""
    pos = np.arange(length) / max(length - 1, 1)
    return np.minimum((pos * n_bins).astype(int), n_bins - 1)


def load_chain(analyses_dir, chain, preds):
    """``{series: values}`` for one chain, or a string saying why it is skipped."""
    from data.scripts.extract_rmsf_labels import extract_bfactor_labels, extract_rmsf_labels

    folder = chain.replace('.', '_')
    base = os.path.join(analyses_dir, folder, folder)
    _, rmsf = extract_rmsf_labels(base + '_RMSF.tsv')
    bfac = extract_bfactor_labels(base + '_Bfactor.tsv')[1].to_numpy(dtype=float)
    if folder not in preds:
        return 'no prediction'
    pred = preds[folder]
    if not len(bfac) == len(rmsf) == len(pred):
        return f'lengths differ: B-factor {len(bfac)}, RMSF {len(rmsf)}, prediction {len(pred)}'
    if not np.isfinite(bfac).any():
        return 'no B-factors'
    return {'bfactor': bfac, 'rmsf': np.asarray(rmsf), 'flexpert': pred}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--predictions', default='outputs/predictions/flexpert_seq_atlas_test.txt')
    ap.add_argument('--splits_path', default='data/atlas_splits.json')
    ap.add_argument('--split', default='test')
    ap.add_argument('--analyses_dir', default='data/atlas/analyses')
    ap.add_argument('--n_bins', type=int, default=20)
    ap.add_argument('--fig_dir', default='analysis/figures')
    args = ap.parse_args()

    with open(args.splits_path) as f:
        chains = json.load(f)[args.split]
    preds = read_flexpert_predictions(args.predictions)

    z = {k: [] for k, _, _ in SERIES}
    bins = []
    for chain in chains:
        data = load_chain(args.analyses_dir, chain, preds)
        if isinstance(data, str):
            print(f'  skip {chain}: {data}')
            continue
        for k in z:
            z[k].append(zscore(data[k]))
        bins.append(relative_bins(len(data['rmsf']), args.n_bins))
    n = len(bins)
    print(f'{n} of {len(chains)} chains used')

    bins = np.concatenate(bins)
    centres = (np.arange(args.n_bins) + 0.5) / args.n_bins
    stats = {}
    for k in z:
        v = np.concatenate(z[k])
        per_bin = [v[(bins == i) & np.isfinite(v)] for i in range(args.n_bins)]
        stats[k] = (np.array([x.mean() for x in per_bin]),
                    np.array([x.std(ddof=1) / np.sqrt(x.size) for x in per_bin]))

    os.makedirs(args.fig_dir, exist_ok=True)
    csv = os.path.join(args.fig_dir, 'fig5c_termini.csv')
    with open(csv, 'w') as f:
        f.write('bin_centre,' + ','.join(f'{k}_mean,{k}_sem' for k, _, _ in SERIES) + '\n')
        for i, c in enumerate(centres):
            f.write(f'{c:.3f},' + ','.join(f'{stats[k][0][i]:.4f},{stats[k][1][i]:.4f}'
                                          for k, _, _ in SERIES) + '\n')
    print(f'  wrote {csv}')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for k, label, colour in SERIES:
        mean, sem = stats[k]
        ax.plot(centres, mean, color=colour, lw=2, label=label)
        ax.fill_between(centres, mean - sem, mean + sem, color=colour, alpha=0.25, lw=0)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('relative position (0 = N → 1 = C)')
    ax.set_ylabel('within-chain z-score (mean over residues)')
    ax.set_title(f'ATLAS test set, n = {n} chains')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', fontsize=9)
    fig.tight_layout()
    out = os.path.join(args.fig_dir, 'fig5c_termini.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
