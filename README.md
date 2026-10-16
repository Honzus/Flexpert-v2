![Flexpert](Flexpert_logo.png)

# Flexpert-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Flexpert-v2 predicts **per-residue protein flexibility** (RMSF) by LoRA fine-tuning protein
language models. It is an improved version of the original
[Flexpert](https://github.com/KoubaPetr/Flexpert)
([*Learning to engineer protein flexibility*](https://arxiv.org/abs/2412.18275)), with new
backbones and training on more MD data.

This repository provides two models, Flexpert-Seq (with two backbones) and Flexpert-3D:

| Model | Backbone | Input | Train script |
|-------|----------|-------|--------------|
| **Flexpert-Seq** | [ProstT5](https://huggingface.co/Rostlab/ProstT5) (AA-only) | amino-acid sequence | `train.py` |
| **Flexpert-Seq (ProtT5)** | [ProtT5-XL-U50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | amino-acid sequence | `train.py --backbone prott5` |
| **Flexpert-3D**  | [SaProt-650M](https://huggingface.co/westlake-repl/SaProt_650M_PDB) | structure-aware AA + 3Di tokens | `train_3d.py` |

Flexpert-Seq (ProtT5) is the original Flexpert-Seq backbone, retrained with this codebase's
recipe; it is included for comparison. All models predict one flexibility value per residue, and
all use the same inference script, `run_inference.py`.

---

## Contents

- [Installation](#installation)
- [Pretrained weights](#pretrained-weights)
- [Inference](#inference)
- [Data](#data)
- [Training](#training)
- [Evaluation and reproducing the released numbers](#evaluation-and-reproducing-the-released-numbers)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [How the models work](#how-the-models-work)
- [Tests](#tests)
- [Flexibility and embedding universes](#flexibility-and-embedding-universes)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Installation

Tested with **Python 3.11** on Linux.

```bash
git clone https://github.com/Honzus/Flexpert-v2.git
cd Flexpert-v2

# Preferred: uv (https://docs.astral.sh/uv/)
uv sync
source .venv/bin/activate

# Alternative: pip
pip install -r requirements.txt
```

**Run every command in this README from the repository root.**

Optional settings live in `configs/env_config.yaml`:

- `huggingface.HF_HOME`: where the backbones are cached (ProstT5 ≈ 5.5 GB, ProtT5 ≈ 11 GB,
  SaProt-650M ≈ 2.5 GB).
  The first run downloads them from the HuggingFace Hub. If you leave the `/path/to/...`
  placeholder, or `HF_HOME` is already set in your shell, the normal HuggingFace cache is used.
- `gpus.cuda_visible_device`: which GPU to use. It only applies when `CUDA_VISIBLE_DEVICES` is
  not already set, so it never overrides SLURM.

A CUDA GPU is strongly recommended. The code also runs on CPU, but that is only practical for a
handful of short sequences.

## Pretrained weights

The trained weights are on the HuggingFace Hub at
[`Honzus24/Flexpert-v2`](https://huggingface.co/Honzus24/Flexpert-v2). Download them into
`weights/`:

```bash
python3 download_flexpert_weights.py                              # ATLAS-trained Seq + 3D (default)
python3 download_flexpert_weights.py --dataset all                # the six ProstT5 / SaProt checkpoints
python3 download_flexpert_weights.py --models all --dataset all   # all nine, incl. ProtT5
```

`--dataset` accepts `atlas`, `mdcath`, `combined` or `all`. `--models` accepts `v2` (ProstT5 Seq
and SaProt 3D, the default), `prott5` or `all`. Each model has one checkpoint per training dataset:

| | ATLAS (default) | mdCATH | combined |
|---|---|---|---|
| Flexpert-Seq (ProstT5) | `flexpert_seq_prostt5_aa_weights.bin` | `flexpert_seq_prostt5_aa_mdcath_weights.bin` | `flexpert_seq_prostt5_aa_combined_weights.bin` |
| Flexpert-3D (SaProt) | `flexpert_3d_saprot_weights.bin` | `flexpert_3d_saprot_mdcath_weights.bin` | `flexpert_3d_saprot_combined_weights.bin` |
| Flexpert-Seq (ProtT5) | `flexpert_seq_prott5_weights.bin` | `flexpert_seq_prott5_mdcath_weights.bin` | `flexpert_seq_prott5_combined_weights.bin` |

## Inference

`run_inference.py` reads a FASTA file and writes a text file with a `>NAME` line followed by one
line of comma-separated per-residue predictions for each chain. With `--splits_path` it predicts
only the chains of one split; without it, every sequence in the FASTA.

```bash
# Flexpert-Seq (ProstT5, AA FASTA) -> outputs/predictions/flexpert_seq_atlas_test.txt
python3 run_inference.py --backbone prostt5 \
    --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json --split test

# Flexpert-3D (SaProt, SA-pair FASTA) -> outputs/predictions/flexpert_3d_atlas_test.txt
python3 run_inference.py --backbone saprot \
    --weights_path weights/flexpert_3d_saprot_weights.bin \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json --split test
```

Options:

- `--backbone`: `prostt5` or `prott5` (Flexpert-Seq, AA FASTA) or `saprot` (Flexpert-3D,
  SA-pair FASTA). It must match the weights.
- `--splits_path` / `--split`: optional splits JSON and split (`train`, `validation` or `test`,
  default `test`).
- `--output_path`: where to write the predictions. The default is
  `outputs/predictions/<model>_<dataset>_<split>.txt` with a splits file, or
  `outputs/predictions/<model>_<fasta name>.txt` without one. `<model>` is `flexpert_seq`,
  `flexpert_seq_prott5` or `flexpert_3d`, plus `_mdcath` / `_combined` for those weights;
  `<dataset>` is the splits file name without `_splits.json`.
- `--weights_path`: a `.bin` or `.safetensors` file, a sharded checkpoint (`*.index.json`), or a
  checkpoint directory such as `outputs/checkpoints/<run>_<timestamp>/final_model`.
- `--lora_r` / `--lora_alpha`: must match the values the weights were trained with. The released
  weights use the defaults (4 and 1).
- `--batch_size` (default 4).

The models were trained on at most 1023 (Flexpert-Seq) or 1022 (Flexpert-3D) residues per
chain. Flexpert-Seq truncates longer chains, so its predictions cover only their first 1023
residues; Flexpert-3D predicts the whole chain. Either way the script prints one warning listing
the long chains.

### Predicting your own proteins

Any of the 9 weight files above works; pass the matching `--backbone` (`prostt5` for
`flexpert_seq_prostt5_aa*`, `prott5` for `flexpert_seq_prott5*`, `saprot` for `flexpert_3d_saprot*`).
A GPU is recommended, but CPU works for a handful of proteins (slowly).

**Flexpert-Seq** needs only sequences. Use any FASTA file: the name of each prediction is the
FASTA header up to the first space (`>sp|P69905|HBA_HUMAN Hemoglobin` becomes
`sp|P69905|HBA_HUMAN`), so names must be unique. Sequences may span several lines; lower-case
letters, `*`, `-` and spaces are accepted, and non-standard amino acids become `X`.

```bash
# ProstT5 -> outputs/predictions/flexpert_seq_fasta_dataset.txt
python3 run_inference.py --backbone prostt5 \
    --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/custom_dataset/fasta_dataset.fasta

# ProtT5 -> outputs/predictions/flexpert_seq_prott5_fasta_dataset.txt
python3 run_inference.py --backbone prott5 \
    --weights_path weights/flexpert_seq_prott5_weights.bin \
    --fasta_path data/custom_dataset/fasta_dataset.fasta
```

**Flexpert-3D** needs a structure for each protein (experimental, or predicted by
AlphaFold / ESMFold). `data/scripts/build_sa_fasta.py` turns `.pdb` / `.cif` files into an
SA-pair FASTA (an uppercase amino acid followed by a lowercase
[Foldseek](https://github.com/steineggerlab/foldseek) 3Di letter for every residue):

```bash
python3 -m data.scripts.build_sa_fasta --pdb_dir data/PDBs --out outputs/custom_sa.fasta

python3 run_inference.py --backbone saprot \
    --weights_path weights/flexpert_3d_saprot_weights.bin \
    --fasta_path outputs/custom_sa.fasta
# -> outputs/predictions/flexpert_3d_custom_sa.txt
```

`build_sa_fasta` needs the Foldseek binary. Install it with
`conda install -c conda-forge -c bioconda foldseek`, or download the static binary from
<https://mmseqs.com/foldseek/> (for example `foldseek-linux-avx2.tar.gz`, which contains
`foldseek/bin/foldseek`). The script looks for it in `--foldseek`, then `$FOLDSEEK`, then
`foldseek_bin` in `configs/data_config.yaml`, then `PATH`.

It names each sequence after its file (`1a62_A.pdb` becomes `>1a62_A`) and uses the first chain
in each file. Other options:

- `--chain X`: use chain `X` from every file instead (for multi-chain files).
- `--plddt_mask`: for AlphaFold / ESMFold models, mask the 3Di letter of residues with pLDDT below
  `--plddt_threshold` (default 70) with `#`.
- `--names FILE`: restrict and order the output to the names in `FILE` (one per line). This also
  sets the FASTA headers, for example to match a splits JSON.
- `--glob`: only use files matching this pattern (default: all `.pdb`, `.ent`, `.cif`, `.mmcif`
  files, optionally gzipped).

It never overwrites an existing file unless you pass `--force`, and it refuses to write over the
bundled `data/*_sa_sequences.fasta`.

## Data

Three datasets ship in `data/`. Each has the same four files, so you only need to change the
`--data_path` / `--fasta_path` / `--splits_path` arguments to train or evaluate on another one:

| Dataset | AA FASTA (Seq) | SA-pair FASTA (3D) | RMSF labels | Splits |
|---------|----------------|--------------------|-------------|--------|
| ATLAS    | `atlas_sequences.fasta` | `atlas_sa_sequences.fasta` | `rmsf_atlas_data_prottransready.txt` | `atlas_splits.json` |
| mdCATH   | `mdCATH_sequences.fasta` | `mdCATH_sa_sequences.fasta` | `rmsf_mdCATH_data.txt` | `mdCATH_splits.json` |
| combined | `combined_sequences.fasta` | `combined_sa_sequences.fasta` | `rmsf_combined_data.txt` | `combined_splits.json` |

| Dataset | Chains in FASTA / labels | train / validation / test | Name format |
|---------|--------------------------|---------------------------|-------------|
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) | 1390 | 1112 / 139 / 139 | `1k5n.A` (PDB id + chain) |
| [mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH) | 5398 | 4524 / 191 / 529 | `1bprA00` (CATH domain) |
| combined (ATLAS + mdCATH) | 6788 | 5636 / 330 / 668 | both |

The mdCATH and combined FASTA and label files contain 154 domains that are in none of the
splits.

File formats:

- **FASTA**: AA FASTAs hold plain sequences. SA-pair FASTAs interleave an uppercase amino acid
  and a lowercase 3Di letter for each residue, so a chain of N residues is a 2N-character string.
- **RMSF labels**: one line per chain, `NAME:\tv1, v2, ...` (name, colon, tab, then
  comma-separated values). Values above 900 mark missing or disordered residues and are
  ignored in training and evaluation.
- **Splits**: `{"train": [...], "validation": [...], "test": [...]}`. Training reads all three
  keys; inference reads only the split you ask for.

Other bundled files:

| Path | Contents |
|------|----------|
| `data/PDBs/` | 10 example ATLAS structures (used in the Flexpert-3D example above) |
| `data/custom_dataset/` | the same 10 chains as a FASTA (`fasta_dataset.fasta`), plus the `chain_set.jsonl` that `prepare_dataset.py` makes from `data/PDBs/` |
| `data/atlas/precomputed_flexibility_profiles/` | ANM/GNM and ESM-pLDDT baselines, and the Flexpert v1 predictions (see [below](#flexpert-v1-baselines)) |
| `data/atlas/2022_06_13_ATLAS_info.tsv`, `data/atlas/2022_06_13_ATLAS_pdb.txt` | ATLAS metadata and chain list |
| `data/mdCATH/mdCATH/pdbs_list.txt` | the mdCATH domain list |

The raw MD data is **not** shipped: the ATLAS per-protein analyses and the mdCATH HDF5 files
(about 1.5 TB). You only need them to rebuild the labels or to run the Table 1 / mdCATH analyses
below.

### Preparing your own dataset

To train on your own chains, make the four files above in the same formats:

- an AA FASTA (Flexpert-Seq) and/or an SA-pair FASTA (Flexpert-3D; build it with
  `python3 -m data.scripts.build_sa_fasta`, see [above](#predicting-your-own-proteins));
- an RMSF label file;
- a splits JSON with `train`, `validation` and `test` lists.

Names must match exactly across the FASTA, the labels and the splits.

To rebuild the ATLAS labels from the raw ATLAS analyses (the bundled
`data/rmsf_atlas_data_prottransready.txt` was made this way), run the two commands below. The
first downloads the per-protein analyses via the ATLAS API into `data/atlas/analyses/`, which is
large and slow.

```bash
python3 data/atlas/download_analyses.py
python3 -m data.scripts.extract_rmsf_labels --out outputs/rmsf_atlas_labels.txt
```

`extract_rmsf_labels` refuses to overwrite an existing file unless you pass `--force`.
If you use ATLAS, please cite [Vander Meersche et al.](https://academic.oup.com/nar/article/52/D1/D384/7438909).

`python3 -m data.scripts.prepare_dataset` parses the PDBs in `data/PDBs/` into
`data/custom_dataset/chain_set.jsonl` (sequences plus backbone coordinates). Flexpert-Seq and
Flexpert-3D do not need this file. The input and output directories are set in
`configs/data_config.yaml`.

## Training

Defaults are in `configs/train_config.yaml`: the bundled ATLAS data, batch size 8, 100 epochs,
learning rate 3e-4, evaluation every epoch, and the best checkpoint chosen by validation
Spearman. Command-line flags override them.

```bash
# Flexpert-Seq (ProstT5, AA FASTA)
python3 train.py --run_name atlasFlexpertProstT5 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json

# Flexpert-Seq on ProtT5 (recipe of the released ProtT5 weights; mdCATH used --save_steps 512)
python3 train.py --backbone prott5 --run_name atlasFlexpertProtT5 --save_steps 256 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json

# Flexpert-3D (SaProt): note the SA-pair FASTA
python3 train_3d.py --run_name atlasFlexpertSaprot \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json
```

The final weights are written to
`outputs/checkpoints/<run_name>_<timestamp>/final_model/pytorch_model.bin`. That is the file
(or the `final_model` directory) you pass to `run_inference.py --weights_path`.

Useful flags (same for both scripts):

- `--backbone` (`train.py` only): `prostt5` (default) or `prott5`.
- `--batch_size`, `--epochs`, `--gradient_accumulation_steps` and `--mixed_precision` (fp16).
- `--save_steps N`: validate and checkpoint every N steps instead of every epoch.
- `--seed` (default 42).
- `--lora_r` / `--lora_alpha` (default 4 / 1, as for the released weights).
- `--max_steps N`: a quick smoke test. It trains N steps with no evaluation and still writes
  `final_model`, for example `python3 train.py --run_name smoke_seq --max_steps 4`.

The training loss uses at most the first 1023 residues of a chain (ProstT5 / ProtT5) or 1022
(SaProt).

Logging goes to [Weights & Biases](https://wandb.ai):

- `WANDB_MODE=offline` keeps the logs local, under `outputs/logs/`.
- `WANDB_MODE=disabled` turns W&B off.

`slurm/train_prostt5_seq.sh`, `slurm/train_prott5_seq.sh` and `slurm/train_saprot_3d.sh` are
example batch scripts. Submit them from the repository root, and change the partition and
environment lines for your cluster.

## Evaluation and reproducing the released numbers

**Per-chain correlations.** `evaluation/evaluate_ablations.py` computes the Spearman and Pearson
correlation between predictions and labels for every chain in a split, and prints the mean ± std
over chains for each prediction file:

```bash
python3 -m evaluation.evaluate_ablations \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --splits_path data/atlas_splits.json \
    --pred_files \
        "Flexpert-Seq:outputs/predictions/flexpert_seq_atlas_test.txt" \
        "Flexpert-3D:outputs/predictions/flexpert_3d_atlas_test.txt"
```

Add `--per_chain_out <file.csv>` to also save the per-chain values. `slurm/eval_atlas.sh` runs
inference for both models on the ATLAS test split and then this evaluation.

**Reference numbers.** Run the released weights with `run_inference.py` on each test split,
then evaluate with `evaluate_ablations`. The table shows mean Spearman ρ / Pearson r over test
chains; you should get these to within about ±0.005. For mdCATH, use `data/mdCATH_sequences.fasta`
(or `_sa_`), `data/rmsf_mdCATH_data.txt` and `data/mdCATH_splits.json`; for combined, use the
`combined_*` files.

| Weights (trained on) | ATLAS test (N = 139) | mdCATH test (N = 529) | combined test (N = 668) |
|---|---|---|---|
| Flexpert-Seq (ATLAS) | 0.7142 / 0.7794 | 0.6678 / 0.7197 | 0.6775 / 0.7321 |
| Flexpert-3D (ATLAS) | 0.8240 / 0.8748 | 0.8004 / 0.8476 | 0.8053 / 0.8533 |
| Flexpert-Seq (mdCATH) | 0.7254 / 0.7825 | 0.7759 / 0.8086 | 0.7654 / 0.8032 |
| Flexpert-3D (mdCATH) | 0.8110 / 0.8438 | 0.8545 / 0.8907 | 0.8455 / 0.8810 |
| Flexpert-Seq (combined) | 0.7553 / 0.8046 | 0.7735 / 0.8125 | 0.7697 / 0.8109 |
| Flexpert-3D (combined) | 0.8445 / 0.8800 | 0.8522 / 0.8885 | 0.8506 / 0.8868 |
| Flexpert-Seq ProtT5 (ATLAS) | 0.6907 / 0.7607 | 0.6772 / 0.7255 | 0.6800 / 0.7328 |
| Flexpert-Seq ProtT5 (mdCATH) | 0.6815 / 0.7541 | 0.7457 / 0.7839 | 0.7323 / 0.7777 |
| Flexpert-Seq ProtT5 (combined) | 0.7116 / 0.7798 | 0.7489 / 0.7803 | 0.7411 / 0.7802 |

For example, the mdCATH column for the ATLAS-trained Flexpert-Seq weights:

```bash
python3 run_inference.py --backbone prostt5 \
    --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/mdCATH_sequences.fasta --splits_path data/mdCATH_splits.json
python3 -m evaluation.evaluate_ablations \
    --data_path data/rmsf_mdCATH_data.txt --splits_path data/mdCATH_splits.json \
    --pred_files "Flexpert-Seq:outputs/predictions/flexpert_seq_mdcath_test.txt"
```

To check all 27 numbers at once, run the 9 released weights on the 3 test splits with the
default output names, then:

```bash
python3 -m evaluation.check_reproduction     # PASS/FAIL per row against the table above (±0.005)
```

`sbatch slurm/verify_core.sh` does all of this on one GPU (it downloads the weights if needed,
smoke-trains each model for a few steps, and runs every weight file on the custom-protein
examples above); set the `--partition` line for your cluster first.

### Flexpert v1 baselines

The original Flexpert's predictions (v1, ProtT5-based) for the ATLAS test split are
shipped in `data/atlas/precomputed_flexibility_profiles/` as
`flexpert_v1_prott5_seq_atlas_test.txt` and `flexpert_v1_prott5_3d_atlas_test.txt`. They use the
same format as `run_inference.py` output, so you can pass them to `evaluate_ablations` directly.
On the ATLAS test split they give 0.7090 / 0.7773 (v1 Seq) and 0.7627 / 0.8258 (v1 3D).

### Correlation with other flexibility proxies

`evaluation/get_correlation_analysis.py` compares MD RMSF with B-factors, AF2/ESMFold pLDDT and
ANM/GNM fluctuations. With `--evaluate_flexpert`, it uses only the ATLAS test split and adds rows
for Flexpert v1 (from the shipped files) and Flexpert v2. The v2 predictions are read from
`outputs/predictions/flexpert_{seq,3d}_atlas_test.txt`, the default inference outputs above, and
the v2 rows are skipped with a message if those files are missing. The script needs the
per-protein ATLAS analyses, so download them first:

```bash
python3 data/atlas/download_analyses.py                       # -> data/atlas/analyses/
python3 -m evaluation.get_correlation_analysis                # whole ATLAS set
python3 -m evaluation.get_correlation_analysis --evaluate_flexpert
```

### mdCATH reliability ceiling

`evaluation/get_correlation_analysis_mdCATH.py` reports the inter-replica reliability ceiling,
which is the highest correlation any predictor can reach given the noise between the five MD
replicas. With `--evaluate_flexpert` it also reports the Pearson correlation of Flexpert-Seq
predictions (`--predictions`, default `outputs/predictions/flexpert_seq_mdcath_test.txt`).

The script reads the per-domain mdCATH HDF5 files from `data/mdCATH/mdCATH/data/`. The full set
is about 1.5 TB, so you can download only the test domains:

```bash
python3 -c "import json; print('\n'.join(json.load(open('data/mdCATH_splits.json'))['test']))" \
    > outputs/mdcath_test_codes.txt
python3 data/mdCATH/download_mdcath.py --codes_file outputs/mdcath_test_codes.txt
python3 -m evaluation.get_correlation_analysis_mdCATH --evaluate_flexpert
```

## Outputs

Everything the code generates goes to a fixed, git-ignored place:

| Path | Written by | Contents |
|------|------------|----------|
| `weights/` | `download_flexpert_weights.py` | released weights |
| `outputs/checkpoints/<run>_<timestamp>/` | `train.py`, `train_3d.py` | Trainer checkpoints; `final_model/pytorch_model.bin` + `config.json` |
| `outputs/predictions/` | `run_inference.py` | `<model>_<dataset>_<split>.txt` or `<model>_<fasta name>.txt` predictions |
| `outputs/logs/` | training, SLURM scripts | W&B runs and `slurm-<jobname>-<jobid>.out` logs |
| `plots/` | `analysis/` scripts | figures |

## Repository layout

```
.
├── train.py                    # train Flexpert-Seq (ProstT5 or ProtT5, AA-only)
├── train_3d.py                 # train Flexpert-3D (SaProt)
├── run_inference.py            # inference for all models
├── download_flexpert_weights.py  # fetch released weights into weights/
├── models/                     # ProstT5 / ProtT5 / SaProt token-regression models, linear head
├── utils/                      # datasets, collators, trainer, metrics (utils.py),
│                               # LoRA scaling (lora_utils.py), env setup (env.py)
├── configs/                    # train_config.yaml, data_config.yaml, env_config.yaml
├── data/                       # bundled datasets (see Data)
│   └── scripts/                # build_sa_fasta, extract_rmsf_labels, prepare_dataset
├── evaluation/                 # evaluate_ablations, get_correlation_analysis{,_mdCATH}
├── analysis/                   # flexibility / embedding universe plots (analysis/README.md)
├── slurm/                      # example SLURM scripts
├── tests/                      # CPU tests for the core code
└── weights/  outputs/          # downloaded weights and generated files (git-ignored)
```

## How the models work

```
Flexpert-Seq:  AA sequence  ──→ ProstT5 (or ProtT5) encoder + LoRA ──→ per-residue embedding ──→ linear head ──→ RMSF
Flexpert-3D:   AA+3Di pairs ──→ SaProt                      + LoRA ──→ per-residue embedding ──→ linear head ──→ RMSF
```

- **LoRA + MultiplicativeScaling.** LoRA adapters (rank 4, alpha 1) are added to the attention
  projections, and each adapted layer is wrapped in `MultiplicativeScaling`
  (`utils/lora_utils.py`). The backbone stays frozen; only the LoRA weights, the scalings, the
  LayerNorms and the head are trained.
- **Input format.** ProstT5 and ProtT5 get space-separated amino acids with no prefix token.
  Uncommon amino acids (O, B, U, Z) are replaced by X.
- **Masked MSE loss.** `ENMAdaptedTrainer` ignores padding and residues whose label is above 900
  (missing or disordered).
- **Metrics.** Spearman and Pearson correlation and MSE; the best checkpoint is chosen by
  Spearman.

## Tests

The tests run on CPU in under a minute. They use tiny random models and need no GPU, weights or
network access:

```bash
python3 -m pytest -q
```

## Flexibility and embedding universes

`analysis/` makes the large 2D maps of protein flexibility (t-SNE / UMAP over
predicted flexibility profiles or per-residue embeddings). See
[`analysis/README.md`](analysis/README.md) for the workflow and a quick `--limit` test.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- The LoRA fine-tuning of protein language models is based on
  [Schmirler et al.](https://www.nature.com/articles/s41467-024-51844-2)
  ([ProtTrans fine-tuning repo](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning)).
- Flexibility labels come from the [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) MD dataset
  ([Vander Meersche et al.](https://academic.oup.com/nar/article/52/D1/D384/7438909)) and from
  [mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH) (Mirarchi et al.).
- Backbones: [ProstT5](https://huggingface.co/Rostlab/ProstT5),
  [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) and
  [SaProt](https://huggingface.co/westlake-repl/SaProt_650M_PDB).
