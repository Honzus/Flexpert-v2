# Flexibility & embedding "universe" analysis

These scripts reproduce the large-scale 2D maps of protein flexibility used in the paper.
Flexpert-Seq is run over a large set of chains (~84k in the paper), and each chain is embedded
into 2D with t-SNE / UMAP — either from its **flexibility profile** or from the model's
**per-residue embeddings**.

> **Heads-up — large inputs are not shipped.** The full-PDB prediction file (~285 MB) and the
> per-residue embedding HDF5 (~45 GB) are far too large for the repository, and the CATH shading
> needs the mdCATH domain list. The scripts and the resulting figures are provided; regenerate the
> inputs as below (or run a quick `--limit 2000` smoke test on a smaller prediction file).

## Scripts

| Script | Purpose | Key inputs |
|--------|---------|------------|
| `extract_embeddings.py` | Dump per-residue **ProstT5** embeddings (encoder final hidden state) to HDF5 | Flexpert-Seq weights, an AA FASTA, a splits JSON |
| `plot_flexibility_universe.py` | 2D map of chains described by their flexibility profile | a predictions `.txt` (from `run_inference.py`) |
| `plot_embedding_universe.py` | 2D map of chains described by mean-pooled embeddings | the embeddings `.h5` + the flexibility cache |
| `plot_embedding_universe_cath.py` | Re-shade the embedding map by CATH class/architecture | `embedding_universe_cache.npz` + `data/pdb_cath_labels.tsv` |
| `extract_cath_labels.py` | Per-PDB CATH Class+Architecture labels | `cath-domain-list.txt` (from mdCATH) |

## Typical workflow

```bash
# 1. Predict flexibility for the chains you want to map (any AA fasta + splits file).
python3 run_inference.py --backbone prostt5 \
    --weights_path models/weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path <your_chains.fasta> --splits_path <your_splits.json> --split test \
    --output_path prediction_results/universe_predictions.txt

# 2. Flexibility universe (profile-based). --limit for a quick smoke test.
python3 analysis/plot_flexibility_universe.py \
    --predictions prediction_results/universe_predictions.txt --limit 2000

# 3. (Optional) Embedding universe. First dump embeddings, then plot.
python3 analysis/extract_embeddings.py \
    --weights_path models/weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path <your_chains.fasta> --splits_path <your_splits.json> --split test \
    --output_path prediction_results/universe_embeddings.h5
python3 analysis/plot_embedding_universe.py \
    --embeddings prediction_results/universe_embeddings.h5 --methods tsne,umap

# 4. (Optional) CATH-shaded variant — needs the mdCATH cath-domain-list.txt.
python3 analysis/extract_cath_labels.py        # -> data/pdb_cath_labels.tsv
python3 analysis/plot_embedding_universe_cath.py
```

Figures are written to `plots/` and intermediate features/embeddings are cached as `.npz`
(both are git-ignored). Run the scripts from the repository root so the relative paths resolve.

> **Note.** In this release the embeddings come from the ProstT5 Flexpert-Seq backbone (the paper's
> original universe used ProtT5). Some default filenames in the scripts still say `prot5`; pass
> `--embeddings` / `--predictions` explicitly to point at your own files.
