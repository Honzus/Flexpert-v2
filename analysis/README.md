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
`run_inference.py` and `extract_embeddings` both process every FASTA record when you omit
`--splits_path`.

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

## Figure 5 (proteome-scale annotation)

Five scripts reproduce the panels of Figure 5 and Supplementary Figure 4 with the released
Flexpert-Seq weights (ProstT5, and ProtT5 for the second set of 5b / Supp. 4). Every output (figures, CSVs, `pooled_*.npz`, caches) goes to
`analysis/figures/`, which is git-ignored.

| Script | Panel | Method |
|--------|-------|--------|
| `extract_pooled_embeddings.py` | inputs for a, b | Last-layer embeddings averaged over residues, one 1024-d vector per chain. Models: `flexpert` (Flexpert-Seq ProstT5), `flexpert_prott5` (Flexpert-Seq ProtT5), vanilla `prott5`, vanilla `prostt5`. Input is the AA sequence alone for all four (no `<AA2fold>` prefix). No length filter. |
| `fig5_tsne.py` | a, Supp. 4 | StandardScaler → PCA 50 (seed 42) → openTSNE (perplexity 50, Euclidean, seed 42). Coloured by CATH architecture. |
| `fig5_neighbours.py` | b | L2-normalise → PCA 50 (seed 42) → exact cosine 40-NN over all chains, self excluded. For every chain and model pair, the number of shared neighbours; split violins and mean ± sd / median / quartiles (CSV) over all chains and over unique-sequence chains. Two sets: with the ProstT5 and with the ProtT5 Flexpert. |
| `fig5_termini.py` | c | ATLAS test set. B-factor, MD RMSF and prediction each z-scored within the chain, residues pooled into 20 bins of relative position i/(L−1), mean ± SEM. Chains whose prediction is truncated (5vz0.A) are skipped. |
| `fig5_hydrophobicity.py` | d, e, f | Kyte-Doolittle vs mean prediction per amino acid; Spearman ρ over the 20 means; His runs ≥ 4 masked. `--set mdcath` splits residues at rSASA 0.25 (biotite Shrake-Rupley, `vdw_radii="Single"`, Tien 2013 maxima). |

The 84,179 PDB chains contain only 45,156 distinct sequences: many entries re-deposit the same
protein with another ligand or soak (lysozyme ×1042, SARS-CoV-2 Mpro ×779, fragment-screen series).
Identical sequences have identical embeddings in every model, so these chains share neighbours
trivially; 5b therefore also reports the 34,949 unique-sequence chains (sequence occurs once in the
FASTA).

Inputs not shipped with the repo (gitignored paths): `data/pdb_sequences.fasta` (the 84,179
monomeric PDB chains), `data/pdb_cath_labels.tsv` (`extract_cath_labels`), `data/atlas/analyses`
(`data/atlas/download_analyses.py`) and the mdCATH domain PDBs in `data/mdCATH/mdCATH/pdbs`.

```bash
W=weights/flexpert_seq_prostt5_aa_weights.bin
# GPU: predictions -> outputs/predictions/flexpert_seq_{pdb_sequences,mdCATH_sequences}.txt
python3 run_inference.py --backbone prostt5 --weights_path $W --fasta_path data/pdb_sequences.fasta --batch_size 16
python3 run_inference.py --backbone prostt5 --weights_path $W --fasta_path data/mdCATH_sequences.fasta --batch_size 16
python3 run_inference.py --backbone prostt5 --weights_path $W --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json --split test
# GPU: pooled embeddings -> analysis/figures/pooled_{flexpert,flexpert_prott5,prott5,prostt5}.npz
for m in flexpert flexpert_prott5 prott5 prostt5; do python3 -m analysis.extract_pooled_embeddings --model $m; done
# CPU: figures
python3 -m analysis.fig5_tsne                       # a + Supp. 4
python3 -m analysis.fig5_neighbours                 # b (k-NN cached as knn40_<model>.npy)
python3 -m analysis.fig5_termini                    # c
python3 -m analysis.fig5_hydrophobicity --set pdb     # d
python3 -m analysis.fig5_hydrophobicity --set mdcath  # e, f (rSASA cached in mdcath_rsasa.npz)
```

> **Note.** In this release the embeddings come from the ProstT5 Flexpert-Seq backbone. The
> paper's original universe used ProtT5, which is why some default file names and plot titles
> still say `prot5`.
