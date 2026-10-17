"""Figure 5b: how many of a chain's 40 nearest neighbours two embedding spaces share, over all chains.

Per space (``pooled_<model>.npz``): L2-normalise -> PCA(50, seed 42) -> exact cosine k-NN over
all chains (self excluded). For every chain and every pair of spaces we count the shared
neighbours out of k and report the distribution (mean ± sd, median, quartiles), for two sets:
one with the ProstT5-based Flexpert, one with the ProtT5-based Flexpert.

Neighbours are always searched over all chains. The summaries are given twice: over all chains
and over unique-sequence chains (sequence occurs once in the FASTA). Many PDB entries re-deposit
an identical sequence (lysozyme, SARS-CoV-2 Mpro, fragment-screen series); identical sequences
have identical embeddings in every space, so such chains share their neighbours trivially.

    python3 -m analysis.fig5_neighbours
    # -> analysis/figures/fig5b_knn_overlap_flexpert_{prostt5,prott5}.png, fig5b_knn_overlap.csv,
    #    knn40_<model>.npy (reused on later runs; --recompute to refit)
"""
import argparse
import os
from collections import Counter

import numpy as np

from analysis.fig5_tsne import MODELS, load_spaces

SETS = {
    'flexpert_prostt5': [('prott5', 'prostt5'), ('prott5', 'flexpert'), ('prostt5', 'flexpert')],
    'flexpert_prott5': [('prott5', 'prostt5'), ('prott5', 'flexpert_prott5'),
                        ('prostt5', 'flexpert_prott5')],
}
SUBSETS = {'all': 'All chains', 'unique': 'Unique-sequence chains'}
COLORS = {'all': '#2a78d6', 'unique': '#eb6834'}


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def knn_cosine(x, k, chunk=4096):
    """Indices ``(N, k)`` of each row's k most cosine-similar other rows, most similar first."""
    import torch

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.from_numpy(l2(x).astype(np.float32)).to(dev)
    out = np.empty((len(x), k), dtype=np.int64)
    for s in range(0, len(x), chunk):
        sim = x[s:s + chunk] @ x.T
        rows = torch.arange(sim.shape[0], device=dev)
        sim[rows, s + rows] = -float('inf')          # exclude self
        out[s:s + chunk] = sim.topk(k, dim=1).indices.cpu().numpy()
    return out


def shared_counts(a, b):
    """Per row, how many neighbour indices ``a[i]`` and ``b[i]`` have in common."""
    return np.array([len(np.intersect1d(r, q)) for r, q in zip(a, b)])


def neighbours(x, k):
    from sklearn.decomposition import PCA

    return knn_cosine(PCA(n_components=50, random_state=42).fit_transform(l2(x)), k)


def unique_sequence_mask(seqs):
    """True where a sequence occurs exactly once in ``seqs``."""
    counts = Counter(seqs)
    return np.array([counts[s] == 1 for s in seqs])


def summarise(c):
    q25, median, q75 = np.percentile(c, [25, 50, 75])
    return {'n_chains': len(c), 'mean': c.mean(), 'sd': c.std(), 'median': median, 'q25': q25, 'q75': q75}


def load_neighbours(fig_dir, model, feats, k, recompute, limit):
    """k-NN indices for one space; cached as ``knn<k>_<model>.npy`` (not for ``--limit`` runs)."""
    path = os.path.join(fig_dir, f'knn{k}_{model}.npy')
    if not recompute and not limit and os.path.exists(path):
        nbrs = np.load(path)
        if len(nbrs) == len(feats):
            return nbrs
    print(f'  {MODELS[model]}: {k}-NN over {len(feats):,} chains ...')
    nbrs = neighbours(feats, k)
    if not limit:
        np.save(path, nbrs)
    return nbrs


def plot_split_violins(counts, pairs, masks, k, title, out_path):
    """One split violin per pair: all chains on the left half, unique-sequence chains on the right."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for x, p in enumerate(pairs):
        for side, subset in ((-1, 'all'), (1, 'unique')):
            c = counts[p][masks[subset]]
            body = ax.violinplot([c], positions=[x], widths=0.8, showextrema=False)['bodies'][0]
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.clip(verts[:, 0], *((-np.inf, x) if side < 0 else (x, np.inf)))  # one half
            body.set(facecolor=COLORS[subset], edgecolor='none', alpha=0.75)
            s = summarise(c)
            xm = x + side * 0.12
            ax.errorbar(xm, s['mean'], yerr=s['sd'], fmt='o', ms=5, color='black', capsize=3, lw=1.2)
            ax.text(x + side * 0.22, k + 2.5, f"{s['mean']:.1f} ± {s['sd']:.1f}",
                    ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f'{MODELS[a]}\nvs {MODELS[b]}' for a, b in pairs], fontsize=9)
    ax.set_ylabel(f'Shared nearest neighbours (of {k})')
    ax.set_ylim(-1, k + 7)
    ax.set_yticks(range(0, k + 1, 10))
    ax.grid(axis='y', color='0.9', lw=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    n = {s: int(m.sum()) for s, m in masks.items()}
    ax.legend(handles=[Patch(color=COLORS[s], alpha=0.75, label=f'{SUBSETS[s]} (n = {n[s]:,})')
                       for s in SUBSETS],
              loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9)
    ax.set_title(f'{title}\nmean ± sd per half; points: mean, bars: ± sd', fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig_dir', default='analysis/figures')
    ap.add_argument('--fasta_path', default='data/pdb_sequences.fasta',
                    help='FASTA the pooled embeddings were made from (for the unique-sequence subset).')
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--limit', type=int, default=None, help='Only the first N chains (smoke test).')
    ap.add_argument('--recompute', action='store_true', help='Refit k-NN even if knn<k>_<model>.npy exists.')
    args = ap.parse_args()

    from run_inference import read_fasta

    names, feats = load_spaces(args.fig_dir, args.limit)
    seqs = read_fasta(args.fasta_path)
    masks = {'all': np.ones(len(names), dtype=bool),
             'unique': unique_sequence_mask([seqs[n] for n in names])}
    nbrs = {m: load_neighbours(args.fig_dir, m, feats[m], args.k, args.recompute, args.limit)
            for m in MODELS}

    suffix = f'_{args.limit}' if args.limit else ''
    csv = os.path.join(args.fig_dir, f'fig5b_knn_overlap{suffix}.csv')
    cols = ['n_chains', 'mean', 'sd', 'median', 'q25', 'q75']
    with open(csv, 'w') as f:
        f.write('set,pair,subset,' + ','.join(cols) + '\n')
        for set_name, pairs in SETS.items():
            counts = {p: shared_counts(nbrs[p[0]], nbrs[p[1]]) for p in pairs}
            for p in pairs:
                for subset, mask in masks.items():
                    s = summarise(counts[p][mask])
                    f.write(f'{set_name},{MODELS[p[0]]} vs {MODELS[p[1]]},{subset},{s["n_chains"]},'
                            + ','.join(f'{s[c]:.2f}' for c in cols[1:]) + '\n')
                    print(f'  {MODELS[p[0]]} vs {MODELS[p[1]]} [{subset}]: '
                          f'{s["mean"]:.1f} ± {s["sd"]:.1f} of {args.k} (n = {s["n_chains"]:,})')
            flexpert = MODELS[pairs[-1][1]]
            plot_split_violins(counts, pairs, masks, args.k,
                               f'{args.k}-NN overlap between embedding spaces, {flexpert}',
                               os.path.join(args.fig_dir, f'fig5b_knn_overlap_{set_name}{suffix}.png'))
    print(f'  wrote {csv}')


if __name__ == '__main__':
    main()
