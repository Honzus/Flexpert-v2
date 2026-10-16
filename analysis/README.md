# Flexibility & embedding "universe" analysis

These scripts make the large 2D maps of protein flexibility shown in the paper. Flexpert-Seq is
run over a large set of chains (about 84k in the paper), and t-SNE / UMAP then places each chain
on a 2D map. A chain is described either by its **flexibility profile** or by the model's
**per-residue embeddings**.

> **Large inputs are not shipped.** The full-PDB prediction file (about 285 MB) and the
> per-residue embedding HDF5 (about 45 GB) are too large for the repository, and the CATH shading
> needs the CATH domain list. Regenerate the inputs as shown below. For a quick check, run the
> plots on a smaller prediction file with `--limit 2000`.

Run every command from the repository root, as a module (`python3 -m analysis.<script>`).

## Scripts

| Script | Purpose | Key inputs |
|--------|---------|------------|
| `extract_embeddings.py` | Save per-residue **ProstT5** embeddings (the encoder's final hidden state, i.e. the input to the regression head) to HDF5 | Flexpert-Seq weights, an AA FASTA, optionally a splits JSON |
| `plot_flexibility_universe.py` | 2D map of chains described by their flexibility profiles | a predictions `.txt` from `run_inference.py` |
| `plot_embedding_universe.py` | 2D map of chains described by their mean-pooled embeddings | the embeddings `.h5`, plus the flexibility cache for colouring |
| `extract_cath_labels.py` | One CATH Class + Architecture label per PDB entry | CATH `cath-domain-list.txt` |
| `plot_embedding_universe_cath.py` | Colour the embedding map by CATH class / architecture | `embedding_universe_cache.npz` + the CATH labels |

## Typical workflow

The example uses the bundled ATLAS sequences; use any AA FASTA to map other chains.
`run_inference.py` always needs a splits JSON. `extract_embeddings` processes every FASTA record
when you omit `--splits_path`.

```bash
# 0. Released Flexpert-Seq weights -> weights/
python3 download_flexpert_weights.py

# 1. Predict flexibility for the chains to map -> outputs/predictions/flexpert_seq_atlas_train.txt
python3 run_inference.py --backbone prostt5 \
    --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/atlas_sequences.fasta --splits_path data/atlas_splits.json --split train

# 2. Flexibility universe (profile-based). Drop --limit for the full run.
python3 -m analysis.plot_flexibility_universe \
    --predictions outputs/predictions/flexpert_seq_atlas_train.txt --limit 2000

# 3. (Optional) Embedding universe: save the embeddings, then plot.
python3 -m analysis.extract_embeddings \
    --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/atlas_sequences.fasta --splits_path data/atlas_splits.json --split train \
    --output_path outputs/predictions/flexpert_seq_atlas_train_embeddings.h5
python3 -m analysis.plot_embedding_universe \
    --embeddings outputs/predictions/flexpert_seq_atlas_train_embeddings.h5 \
    --predictions outputs/predictions/flexpert_seq_atlas_train.txt --methods tsne,umap

# 4. (Optional) CATH-coloured variant. Needs cath-domain-list.txt from a CATH release
#    (http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/).
python3 -m analysis.extract_cath_labels --cath_list <path/to/cath-domain-list.txt> \
    --cath_names outputs/cath-names.txt --output outputs/pdb_cath_labels.tsv
python3 -m analysis.plot_embedding_universe_cath --labels outputs/pdb_cath_labels.tsv
```

Outputs:

- Figures go to `plots/` (change with `--output_dir`).
- Features and 2D embeddings are cached in the current directory as
  `flexibility_universe_cache.npz` and `embedding_universe_cache.npz`; change with `--cache`.
- `plot_embedding_universe` reads its colours from `flexibility_universe_cache.npz`, so run
  step 2 first, or pass `--predictions` as above.
- `plots/`, `outputs/` and `*.npz` are all git-ignored.

Pass `--recompute` to ignore the caches.

The defaults, `outputs/predictions/entire_pdb_atlas_prot5.txt` and
`outputs/predictions/entire_pdb_atlas_prot5_embeddings.h5`, are the file names of the paper's
full-PDB run; pass `--predictions` / `--embeddings` to use your own files.

> **Note.** In this release the embeddings come from the ProstT5 Flexpert-Seq backbone. The
> paper's original universe used ProtT5, which is why some default file names and plot titles
> still say `prot5`.
