"""Embed the entire PDB "universe" of Flexpert flexibility predictions into 2D.

Flexpert-Seq was run over all of PDB, producing per-residue RMSF predictions for
~84k chains (``prediction_results/entire_pdb_atlas_prot5.txt``). Each chain has a
*variable-length* profile, so we describe every chain with a *fixed-length*,
length-invariant feature vector (a density histogram of its per-residue values
plus summary statistics) and embed those vectors with both UMAP and t-SNE.

Outputs one 2x2 figure per method (panels colored by mean flexibility, length,
flexibility std/range, and fraction-flexible) and a cache of the features +
embeddings so re-plotting is cheap.

Example
-------
    # quick smoke test on the first 2000 chains
    python3 plot_flexibility_universe.py --limit 2000

    # full run over all chains
    python3 plot_flexibility_universe.py
"""

import argparse
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------
def read_flexpert_predictions(path, limit=None):
    """Parse a Flexpert prediction file into ``{name: np.ndarray}``.

    Mirrors ``read_flexpert_predictions`` in ``get_correlation_analysis.py`` (inlined
    here to avoid that module's heavy top-level imports). Format is alternating lines:
    ``>name`` then comma-separated per-residue floats.
    """
    pdb_code_to_fluct = {}
    with open(path, "r") as f:
        lines = f.readlines()
    pairs = zip(lines[::2], lines[1::2])
    for i, (name_line, fluct_line) in enumerate(pairs):
        if limit is not None and i >= limit:
            break
        name = name_line.strip().lstrip(">")
        if "." in name:
            name = name.replace(".", "_")
        pdb_code_to_fluct[name] = np.array(
            fluct_line.strip().split(","), dtype=np.float32
        )
    return pdb_code_to_fluct


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
# Names of the scalar summary-stat features, in the order produced by
# ``_summary_stats``. Kept as a module constant so feature vectors stay aligned.
_STAT_NAMES = [
    "mean", "std", "min", "max", "range", "median",
    "p10", "p25", "p75", "p90", "iqr", "cv",
    "top10_mean", "bot10_mean", "skew", "kurtosis", "autocorr1",
]


def _summary_stats(x):
    """Length-invariant summary statistics of a 1D flexibility profile."""
    mean = float(x.mean())
    std = float(x.std())
    xmin, xmax = float(x.min()), float(x.max())
    p10, p25, median, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    # tails: mean of the most-flexible / most-rigid 10% of residues
    k = max(1, int(round(0.1 * len(x))))
    xs = np.sort(x)
    bot10_mean = float(xs[:k].mean())
    top10_mean = float(xs[-k:].mean())
    # higher moments (numpy only, no scipy dependency)
    if std > 1e-8:
        z = (x - mean) / std
        skew = float((z ** 3).mean())
        kurtosis = float((z ** 4).mean() - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0
    cv = std / mean if abs(mean) > 1e-8 else 0.0
    # lag-1 autocorrelation: how smooth the profile is along the chain
    if len(x) > 1 and std > 1e-8:
        a, b = x[:-1] - mean, x[1:] - mean
        autocorr1 = float((a * b).sum() / ((x - mean) ** 2).sum())
    else:
        autocorr1 = 0.0
    return [
        mean, std, xmin, xmax, xmax - xmin, float(median),
        float(p10), float(p25), float(p75), float(p90), float(p75 - p25), cv,
        top10_mean, bot10_mean, skew, kurtosis, autocorr1,
    ]


def featurize(preds, n_bins, bin_edges, flexible_threshold):
    """Turn ``{name: profile}`` into a feature matrix + per-protein color attrs.

    Returns
    -------
    names : list[str]
    features : (N, n_bins + len(_STAT_NAMES)) float32 array
    attrs : dict[str, np.ndarray] with keys mean/length/std/range/frac_flexible
    """
    names, feats = [], []
    a_mean, a_len, a_std, a_range, a_frac = [], [], [], [], []
    for name, x in preds.items():
        # density histogram over the shared global edges (sums to 1 -> length-invariant)
        hist, _ = np.histogram(x, bins=bin_edges, density=False)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        stats = _summary_stats(x)
        feats.append(np.concatenate([hist, stats]).astype(np.float32))
        names.append(name)
        a_mean.append(stats[0])              # mean
        a_std.append(stats[1])               # std
        a_range.append(stats[4])             # range
        a_len.append(len(x))
        a_frac.append(float((x > flexible_threshold).mean()))
    attrs = {
        "mean": np.asarray(a_mean, dtype=np.float32),
        "length": np.asarray(a_len, dtype=np.int32),
        "std": np.asarray(a_std, dtype=np.float32),
        "range": np.asarray(a_range, dtype=np.float32),
        "frac_flexible": np.asarray(a_frac, dtype=np.float32),
    }
    return names, np.asarray(feats, dtype=np.float32), attrs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_umap(X):
    import umap

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42, verbose=True
    )
    return reducer.fit_transform(X).astype(np.float32)


def embed_tsne(X):
    # openTSNE is FFT-accelerated + multicore, so the full ~84k set runs in minutes.
    from openTSNE import TSNE

    tsne = TSNE(
        perplexity=50, metric="euclidean", n_jobs=-1, random_state=42, verbose=True
    )
    return np.asarray(tsne.fit(X), dtype=np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_panels(emb, attrs, method, out_path, fig_title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns

    sns.set_style("whitegrid")

    panels = [
        ("mean", "Mean flexibility", attrs["mean"], False),
        ("length", "Protein length (residues)", attrs["length"].astype(float), True),
        ("std", "Flexibility std", attrs["std"], False),
        ("frac_flexible", "Fraction flexible", attrs["frac_flexible"], False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, (_key, title, c, log) in zip(axes.flat, panels):
        if log:
            c = np.clip(c, 1, None)
            norm = LogNorm(vmin=max(1.0, np.percentile(c, 2)), vmax=np.percentile(c, 98))
            sc = ax.scatter(
                emb[:, 0], emb[:, 1], c=c, s=2, alpha=0.3, cmap="viridis",
                norm=norm, rasterized=True, linewidths=0,
            )
        else:
            vmin, vmax = np.percentile(c, [2, 98])
            sc = ax.scatter(
                emb[:, 0], emb[:, 1], c=c, s=2, alpha=0.3, cmap="viridis",
                vmin=vmin, vmax=vmax, rasterized=True, linewidths=0,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f"{method.upper()}-1")
        ax.set_ylabel(f"{method.upper()}-2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if fig_title is None:
        fig_title = "PDB flexibility universe"
    fig.suptitle(
        f"{fig_title} ({method.upper()}, N={len(emb):,} chains)",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_categorical(emb, labels, order, color_map, method, out_path, fig_title=None,
                     unclassified="Unclassified"):
    """Single-panel scatter colored by a categorical label, with a legend.

    Parameters
    ----------
    emb : (N, 2) array of 2D coordinates.
    labels : sequence of length N; each entry is a category key present in ``order``.
    order : list of category keys, in legend/draw order. The ``unclassified`` category (if
        present) is always drawn first and underneath so colored folds sit on top.
    color_map : dict category-key -> matplotlib color.
    method : "tsne"/"umap" (axis labels). out_path : output PNG. fig_title : suptitle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    labels = np.asarray(labels, dtype=object)

    # draw unclassified first (underneath), then the rest in the given order
    draw_order = ([unclassified] if unclassified in order else []) + \
                 [k for k in order if k != unclassified]

    fig, ax = plt.subplots(figsize=(11, 9))
    for key in draw_order:
        mask = labels == key
        n = int(mask.sum())
        if n == 0:
            continue
        is_unc = key == unclassified
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[color_map[key]], s=2, alpha=0.12 if is_unc else 0.4,
            rasterized=True, linewidths=0,
            label=f"{key} ({n:,})",
        )
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_xticks([])
    ax.set_yticks([])
    if fig_title is None:
        fig_title = "PDB universe"
    ax.set_title(f"{fig_title} ({method.upper()}, N={len(emb):,} chains)", fontsize=14)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
                    markerscale=4, frameon=True)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        default="prediction_results/entire_pdb_atlas_prot5.txt",
        help="Flexpert prediction file (alternating >name / comma-sep floats).",
    )
    ap.add_argument("--output_dir", default="plots")
    ap.add_argument("--cache", default="flexibility_universe_cache.npz")
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument(
        "--flexible_threshold", type=float, default=None,
        help="Residue counts as 'flexible' above this RMSF. Default: global p75.",
    )
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--methods", default="umap,tsne")
    ap.add_argument("--limit", type=int, default=None, help="Only use first N chains (smoke test).")
    ap.add_argument("--recompute", action="store_true", help="Ignore cache and rebuild everything.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---- load from cache if possible (and not forced to recompute) ----------
    have_cache = os.path.exists(args.cache) and not args.recompute
    if have_cache:
        print(f"Loading cache: {args.cache}")
        z = np.load(args.cache, allow_pickle=True)
        names = list(z["names"])
        attrs = {k: z[f"attr_{k}"] for k in ["mean", "length", "std", "range", "frac_flexible"]}
        embeddings = {}
        for m in methods:
            if f"emb_{m}" in z.files:
                embeddings[m] = z[f"emb_{m}"]
        X = z["features"] if "features" in z.files else None
    else:
        names, attrs, embeddings, X = None, None, {}, None

    # ---- featurize (if not cached) ------------------------------------------
    if names is None:
        t0 = time.time()
        print(f"Parsing predictions: {args.predictions}")
        preds = read_flexpert_predictions(args.predictions, limit=args.limit)
        print(f"  parsed {len(preds):,} chains in {time.time() - t0:.1f}s")

        # drop chains that are too short or contain non-finite predictions
        clean = {
            n: x for n, x in preds.items()
            if len(x) >= args.min_len and np.all(np.isfinite(x))
        }
        dropped = len(preds) - len(clean)
        print(f"  kept {len(clean):,} chains ({dropped:,} dropped: short or non-finite)")

        # global histogram bin edges from the pooled value distribution
        all_vals = np.concatenate(list(clean.values()))
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bin_edges = np.linspace(lo, hi, args.n_bins + 1)
        thr = args.flexible_threshold
        if thr is None:
            thr = float(np.percentile(all_vals, 75))
        print(f"  histogram range [{lo:.4f}, {hi:.4f}] / {args.n_bins} bins; "
              f"flexible threshold = {thr:.4f}")

        names, X, attrs = featurize(clean, args.n_bins, bin_edges, thr)
        print(f"  feature matrix: {X.shape}")

    # ---- standardize ---------------------------------------------------------
    if X is not None:
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X).astype(np.float32)
    else:
        Xs = None  # only cached embeddings available

    # ---- embed (any methods not already cached) -----------------------------
    for m in methods:
        if m in embeddings:
            print(f"{m.upper()}: using cached embedding {embeddings[m].shape}")
            continue
        if Xs is None:
            raise RuntimeError(
                f"No cached embedding for '{m}' and no features to fit; rerun with --recompute."
            )
        t0 = time.time()
        print(f"{m.upper()}: fitting on {Xs.shape[0]:,} points ...")
        if m == "umap":
            embeddings[m] = embed_umap(Xs)
        elif m == "tsne":
            embeddings[m] = embed_tsne(Xs)
        else:
            raise ValueError(f"Unknown method: {m}")
        print(f"  {m.upper()} done in {time.time() - t0:.1f}s")

    # ---- cache ---------------------------------------------------------------
    cache_out = {
        "names": np.array(names, dtype=object),
        **{f"attr_{k}": v for k, v in attrs.items()},
        **{f"emb_{m}": e for m, e in embeddings.items()},
    }
    if X is not None:
        cache_out["features"] = X
    np.savez_compressed(args.cache, **cache_out)
    print(f"Cached -> {args.cache}")

    # ---- plot ----------------------------------------------------------------
    for m in methods:
        out_path = os.path.join(args.output_dir, f"flexibility_universe_{m}.png")
        print(f"Plotting {m.upper()} ...")
        plot_panels(embeddings[m], attrs, m, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
