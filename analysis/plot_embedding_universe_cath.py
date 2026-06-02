"""Shade the ProtT5 embedding universe (t-SNE & UMAP) by CATH structural classification.

Reuses the cached 2D embeddings (``embedding_universe_cache.npz``) and the per-PDB CATH labels
from ``extract_cath_labels.py`` (``data/pdb_cath_labels.tsv``) to produce categorical scatter
plots colored by CATH **Class** (4 folds) and **Architecture** (top-N folds + Other), for both
reducers. Chains with no CATH entry are drawn light gray underneath.

    python3 extract_cath_labels.py            # produces data/pdb_cath_labels.tsv
    python3 plot_embedding_universe_cath.py    # writes 4 PNGs to plots/
"""
import argparse
import os
from collections import Counter

import numpy as np

from plot_flexibility_universe import plot_categorical

UNCLASSIFIED = "Unclassified"

# Fixed colors for the (small, stable) set of CATH classes.
CLASS_ORDER = ["Mainly Alpha", "Mainly Beta", "Alpha-Beta",
               "Few Secondary Structures", "Special", UNCLASSIFIED]
CLASS_COLORS = {
    "Mainly Alpha": "crimson",
    "Mainly Beta": "royalblue",
    "Alpha-Beta": "seagreen",
    "Few Secondary Structures": "darkorange",
    "Special": "purple",
    UNCLASSIFIED: "lightgray",
}


def load_labels(path):
    """Return {pdbid: (class_name, arch_name)} from the CATH labels TSV."""
    out = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        ci, ai = header.index("class_name"), header.index("arch_name")
        for line in f:
            p = line.rstrip("\n").split("\t")
            out[p[0]] = (p[ci], p[ai])
    return out


def build_class_labels(names, labels):
    cls = [labels.get(n, (UNCLASSIFIED, None))[0] for n in names]
    order = [k for k in CLASS_ORDER if k in set(cls)]
    return np.array(cls, dtype=object), order, dict(CLASS_COLORS)


def build_arch_labels(names, labels, top_n):
    raw = [labels.get(n, (None, UNCLASSIFIED))[1] or UNCLASSIFIED for n in names]
    counts = Counter(a for a in raw if a != UNCLASSIFIED)
    top = [a for a, _ in counts.most_common(top_n)]
    top_set = set(top)
    n_other = sum(1 for a in raw if a != UNCLASSIFIED and a not in top_set)
    n_lumped = len(counts) - len(top)
    print(f"  architecture: {len(counts)} categories; keeping top {len(top)}, "
          f"lumping {n_lumped} into 'Other' ({n_other:,} chains)")

    arch = np.array(
        [a if (a in top_set or a == UNCLASSIFIED) else "Other" for a in raw],
        dtype=object,
    )
    import matplotlib.pyplot as plt

    palette = plt.cm.tab20(np.linspace(0, 1, max(len(top), 1)))
    color_map = {a: palette[i] for i, a in enumerate(top)}
    order = list(top)
    if (arch == "Other").any():
        color_map["Other"] = (0.55, 0.55, 0.62)
        order.append("Other")
    color_map[UNCLASSIFIED] = "lightgray"
    order.append(UNCLASSIFIED)
    return arch, order, color_map


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="embedding_universe_cache.npz")
    ap.add_argument("--labels", default="data/pdb_cath_labels.tsv")
    ap.add_argument("--output_dir", default="plots")
    ap.add_argument("--methods", default="tsne,umap")
    ap.add_argument("--top_n_arch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="Only plot first N chains (smoke test).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    z = np.load(args.cache, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    embs = {m: z[f"emb_{m}"] for m in methods if f"emb_{m}" in z.files}
    missing = [m for m in methods if m not in embs]
    if missing:
        raise SystemExit(f"Cache {args.cache} has no embedding(s) for: {missing}")

    if args.limit:
        names = names[: args.limit]
        embs = {m: e[: args.limit] for m, e in embs.items()}

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels):,} PDB CATH labels; universe N={len(names):,}")

    schemes = {
        "class": build_class_labels(names, labels),
        "arch": build_arch_labels(names, labels, args.top_n_arch),
    }
    for scheme, (lab, order, cmap) in schemes.items():
        n_unc = int((lab == UNCLASSIFIED).sum())
        title = ("PDB embedding universe — CATH "
                 + ("Class" if scheme == "class" else "Architecture"))
        print(f"[{scheme}] {len(order)} categories, {n_unc:,} unclassified")
        for m in methods:
            out = os.path.join(args.output_dir, f"embedding_universe_{m}_cath_{scheme}.png")
            plot_categorical(embs[m], lab, order, cmap, m, out, fig_title=title)
    print("Done.")


if __name__ == "__main__":
    main()
