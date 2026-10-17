"""Figure 5a / Supplementary Figure 4: t-SNE of mean-pooled embeddings, coloured by CATH architecture.

Inputs are the ``pooled_<model>.npz`` files from ``extract_pooled_embeddings`` (all chains, no
length filter) and the per-PDB CATH labels from ``extract_cath_labels``. Per model:
StandardScaler -> PCA(50, seed 42) -> openTSNE (perplexity 50, Euclidean, seed 42).

    python3 -m analysis.fig5_tsne
    # -> analysis/figures/fig5a_tsne_flexpert.png, supp4_tsne_{flexpert_prott5,prott5,prostt5}.png,
    #    tsne_coords.npz
"""
import argparse
import os

import numpy as np

from analysis.plot_embedding_universe_cath import build_arch_labels, load_labels
from analysis.plot_flexibility_universe import embed_tsne, plot_categorical

MODELS = {'flexpert': 'Flexpert (ProstT5)', 'flexpert_prott5': 'Flexpert (ProtT5)',
          'prott5': 'ProtT5', 'prostt5': 'ProstT5'}
OUT_NAMES = {'flexpert': 'fig5a_tsne_flexpert.png', 'flexpert_prott5': 'supp4_tsne_flexpert_prott5.png',
             'prott5': 'supp4_tsne_prott5.png', 'prostt5': 'supp4_tsne_prostt5.png'}


def load_spaces(fig_dir, limit=None):
    """``(names, {model: features})`` from ``<fig_dir>/pooled_<model>.npz``; rows must line up."""
    names, feats = None, {}
    for m in MODELS:
        z = np.load(os.path.join(fig_dir, f'pooled_{m}.npz'))
        n = [str(x) for x in z['names']]
        if names is None:
            names = n
        if n != names:
            raise ValueError(f'pooled_{m}.npz rows are not aligned with pooled_flexpert.npz')
        feats[m] = z['features'][:limit] if limit else z['features']
    return (names[:limit] if limit else names), feats


def reduce_tsne(x):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit_transform(x)
    xs = PCA(n_components=50, random_state=42).fit_transform(xs).astype(np.float32)
    return embed_tsne(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig_dir', default='analysis/figures', help='Holds pooled_*.npz; figures go here.')
    ap.add_argument('--labels', default='data/pdb_cath_labels.tsv')
    ap.add_argument('--top_n_arch', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None, help='Only the first N chains (smoke test).')
    ap.add_argument('--recompute', action='store_true', help='Refit t-SNE even if tsne_coords.npz exists.')
    args = ap.parse_args()

    names, feats = load_spaces(args.fig_dir, args.limit)
    cache = os.path.join(args.fig_dir, 'tsne_coords.npz' if not args.limit else f'tsne_coords_{args.limit}.npz')
    coords = {}
    if os.path.exists(cache) and not args.recompute:
        z = np.load(cache)
        coords = {m: z[m] for m in MODELS if m in z.files}
    for m in MODELS:
        if m not in coords:
            print(f'{MODELS[m]}: t-SNE on {len(names):,} chains ...')
            coords[m] = reduce_tsne(feats[m])
    np.savez(cache, names=np.array(names), **coords)

    arch, order, cmap = build_arch_labels(names, load_labels(args.labels), args.top_n_arch)
    for m, label in MODELS.items():
        plot_categorical(coords[m], arch, order, cmap, 'tsne', os.path.join(args.fig_dir, OUT_NAMES[m]),
                         fig_title=f'{label} embeddings — CATH Architecture')


if __name__ == '__main__':
    main()
