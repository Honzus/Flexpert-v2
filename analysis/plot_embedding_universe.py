"""Embed the entire PDB "universe" of ProtT5 *embeddings* into 2D (t-SNE and/or UMAP).

Companion to ``plot_flexibility_universe.py``. Instead of describing each chain by its
flexibility profile, we describe it by the model's representation: the per-residue ProtT5
embeddings (last layer before the regression head, ``extract_embeddings.py`` ->
``prediction_results/entire_pdb_atlas_prot5_embeddings.h5``) are mean-pooled to one 1024-d
vector per chain, then reduced with t-SNE and/or UMAP.

Points are colored by the same per-protein attributes as the flexibility universe (mean
flexibility / length / std / fraction-flexible), reused from ``flexibility_universe_cache.npz``,
so the maps line up panel-for-panel and you can ask whether embedding-space structure tracks
flexibility. The pooled feature matrix and every embedding are cached, so adding a method
(e.g. UMAP after t-SNE) reuses the expensive 45 GB read and any already-computed embeddings.

Example
-------
    # quick smoke test on the first 2000 chains
    python3 plot_embedding_universe.py --limit 2000

    # full run, both methods (reuses cached features/embeddings)
    python3 plot_embedding_universe.py --methods tsne,umap
"""

import argparse
import os
import time

import numpy as np

# Reuse the flexibility-universe machinery (module is __main__-guarded, so importing is safe).
from plot_flexibility_universe import (
    embed_tsne,
    embed_umap,
    plot_panels,
    read_flexpert_predictions,
    _summary_stats,
)

_ATTR_KEYS = ["mean", "length", "std", "range", "frac_flexible"]
_KNOWN_METHODS = ["tsne", "umap"]


def _embed(method, Xs):
    if method == "tsne":
        return embed_tsne(Xs)
    if method == "umap":
        return embed_umap(Xs)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Mean-pool the per-residue embeddings
# ---------------------------------------------------------------------------
def pool_embeddings(h5_path, min_len, limit=None):
    """Read the HDF5 embeddings and mean-pool each protein to a 1024-d vector.

    Streamed protein-by-protein so memory stays bounded (~N*1024 float32 output from a
    multi-GB file). Returns ``(names, features (N, D) float32)``.
    """
    import h5py

    names, feats = [], []
    with h5py.File(h5_path, "r") as f:
        all_names = [n.decode() if isinstance(n, bytes) else n for n in f["names"][:]]
        if limit is not None:
            all_names = all_names[:limit]
        for i, name in enumerate(all_names):
            arr = np.asarray(f[name][:], dtype=np.float32)  # (L, 1024)
            if arr.shape[0] < min_len or not np.all(np.isfinite(arr)):
                continue
            feats.append(arr.mean(axis=0))
            names.append(name)
            if (i + 1) % 10000 == 0:
                print(f"  pooled {i + 1:,}/{len(all_names):,}")
    return names, np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-protein coloring attributes
# ---------------------------------------------------------------------------
def load_attrs(flex_cache, predictions, min_len):
    """Return ``{name: {attr: value}}`` for coloring.

    Prefers the precomputed ``flexibility_universe_cache.npz``; falls back to deriving the
    same five attrs from the prediction file (mirrors ``featurize`` in the flexibility script).
    """
    if flex_cache and os.path.exists(flex_cache):
        print(f"Loading coloring attrs from cache: {flex_cache}")
        z = np.load(flex_cache, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        cols = {k: z[f"attr_{k}"] for k in _ATTR_KEYS}
        return {n: {k: cols[k][i] for k in _ATTR_KEYS} for i, n in enumerate(names)}

    print(f"Flex cache absent; deriving attrs from predictions: {predictions}")
    preds = read_flexpert_predictions(predictions)
    clean = {n: x for n, x in preds.items() if len(x) >= min_len and np.all(np.isfinite(x))}
    thr = float(np.percentile(np.concatenate(list(clean.values())), 75))
    attrs = {}
    for n, x in clean.items():
        s = _summary_stats(x)  # [mean, std, min, max, range, ...]
        attrs[n] = {
            "mean": np.float32(s[0]),
            "length": np.int32(len(x)),
            "std": np.float32(s[1]),
            "range": np.float32(s[4]),
            "frac_flexible": np.float32((x > thr).mean()),
        }
    return attrs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings",
                    default="prediction_results/entire_pdb_atlas_prot5_embeddings.h5")
    ap.add_argument("--flex_cache", default="flexibility_universe_cache.npz",
                    help="Source of per-protein coloring attrs (mean/length/std/range/frac).")
    ap.add_argument("--predictions",
                    default="prediction_results/entire_pdb_atlas_prot5.txt",
                    help="Attr fallback if --flex_cache is missing.")
    ap.add_argument("--output_dir", default="plots")
    ap.add_argument("--cache", default="embedding_universe_cache.npz")
    ap.add_argument("--methods", default="tsne,umap",
                    help="Comma-separated reducers to produce: tsne, umap.")
    ap.add_argument("--pca", type=int, default=50,
                    help="PCA dims before reduction (standard high-dim pre-reduction; 0 disables).")
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only use first N chains from the HDF5 (smoke test).")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore cache and rebuild everything.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---- load from cache if possible ----------------------------------------
    have_cache = os.path.exists(args.cache) and not args.recompute
    if have_cache:
        print(f"Loading cache: {args.cache}")
        z = np.load(args.cache, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        X = z["features"]
        attrs = {k: z[f"attr_{k}"] for k in _ATTR_KEYS}
        # keep every already-computed embedding so re-runs never discard one
        embeddings = {m: z[f"emb_{m}"] for m in _KNOWN_METHODS if f"emb_{m}" in z.files}
    else:
        names, X, attrs, embeddings = None, None, None, {}

    # ---- pool embeddings (if not cached) ------------------------------------
    if names is None:
        t0 = time.time()
        print(f"Mean-pooling embeddings: {args.embeddings}")
        pooled_names, X = pool_embeddings(args.embeddings, args.min_len, limit=args.limit)
        print(f"  pooled {len(pooled_names):,} chains -> {X.shape} in {time.time() - t0:.1f}s")

        # ---- align with coloring attrs --------------------------------------
        attr_lookup = load_attrs(args.flex_cache, args.predictions, args.min_len)
        keep = [(i, n) for i, n in enumerate(pooled_names) if n in attr_lookup]
        idx = [i for i, _ in keep]
        names = [n for _, n in keep]
        X = X[idx]
        attrs = {
            k: np.asarray([attr_lookup[n][k] for n in names],
                          dtype=np.int32 if k == "length" else np.float32)
            for k in _ATTR_KEYS
        }
        dropped = len(pooled_names) - len(names)
        print(f"  aligned to {len(names):,} chains with coloring attrs ({dropped:,} dropped)")

    # ---- standardize + PCA (only if some requested method isn't cached) -----
    Xs = None
    if any(m not in embeddings for m in methods):
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X).astype(np.float32)
        if args.pca and 0 < args.pca < Xs.shape[1]:
            from sklearn.decomposition import PCA

            t0 = time.time()
            Xs = PCA(n_components=args.pca, random_state=42).fit_transform(Xs).astype(np.float32)
            print(f"PCA -> {Xs.shape} in {time.time() - t0:.1f}s")

    # ---- reduce (any requested methods not already cached) ------------------
    for m in methods:
        if m in embeddings:
            print(f"{m.upper()}: using cached embedding {embeddings[m].shape}")
            continue
        t0 = time.time()
        print(f"{m.upper()}: fitting on {Xs.shape[0]:,} points ...")
        embeddings[m] = _embed(m, Xs)
        print(f"  {m.upper()} done in {time.time() - t0:.1f}s")

    # ---- cache (preserve every embedding, not just the requested ones) ------
    cache_out = {
        "names": np.array(names, dtype=object),
        "features": X,
        **{f"attr_{k}": v for k, v in attrs.items()},
        **{f"emb_{m}": e for m, e in embeddings.items()},
    }
    np.savez_compressed(args.cache, **cache_out)
    print(f"Cached -> {args.cache}")

    # ---- plot ---------------------------------------------------------------
    for m in methods:
        out_path = os.path.join(args.output_dir, f"embedding_universe_{m}.png")
        print(f"Plotting {m.upper()} ...")
        plot_panels(embeddings[m], attrs, m, out_path, fig_title="PDB embedding universe (ProtT5)")
    print("Done.")


if __name__ == "__main__":
    main()
