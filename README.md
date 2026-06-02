![Flexpert](Flexpert_logo.png)

# Flexpert: Learning to engineer protein flexibility

[![ICLR badge](https://img.shields.io/badge/ICLR-2025-brown.svg)](https://openreview.net/forum?id=L238BAx0wP)
[![arXiv badge](https://img.shields.io/badge/arXiv-2412.18275-b31b1b.svg?color=blue)](https://arxiv.org/abs/2412.18275)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Flexpert predicts **per-residue protein flexibility** (RMSF) by LoRA fine-tuning protein
language models. It accompanies the ICLR 2025 paper
[*Learning to engineer protein flexibility*](https://arxiv.org/abs/2412.18275) by Petr Kouba,
Joan Planas-Iglesias, Jiri Damborsky, Jiri Sedlar, Stanislav Mazurenko and Josef Sivic.

This repository provides two models:

| Model | Backbone | Input | Script |
|-------|----------|-------|--------|
| **Flexpert-Seq** | [ProstT5](https://huggingface.co/Rostlab/ProstT5) (AA-only) | amino-acid sequence | `train.py` |
| **Flexpert-3D**  | [SAProt-650M](https://huggingface.co/westlake-repl/SaProt_650M_PDB) | structure-aware 3Di+AA tokens | `train_3d.py` |

Both predict a scalar flexibility value per residue and run through a single unified inference
script, `run_inference.py`.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Pretrained weights](#pretrained-weights)
- [Quickstart: inference](#quickstart-inference)
- [How the models work](#how-the-models-work)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Flexibility & embedding universes](#flexibility--embedding-universes)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Repository layout

```
.
├── train.py                  # Train Flexpert-Seq (ProstT5, AA-only)
├── train_3d.py               # Train Flexpert-3D (SAProt)
├── run_inference.py          # Unified inference for both models
├── evaluate_ablations.py     # Per-protein Spearman/Pearson table from prediction files
├── get_correlation_analysis.py       # ATLAS baseline correlation table (Table 1)
├── get_correlation_analysis_mdCATH.py  # mdCATH reliability ceiling + Flexpert correlation
├── download_flexpert_weights.py  # Fetch pretrained weights from HuggingFace
├── models/
│   ├── T5_encoder_per_token.py  # ProstT5 + SAProt token-regression models (LoRA)
│   ├── enm_adaptor_heads.py     # Linear regression head
│   └── weights/                 # Pretrained weights live here (git-ignored)
├── utils/
│   ├── utils.py              # Datasets, collators, trainer, metrics
│   └── lora_utils.py         # MultiplicativeScaling LoRA wrapper
├── configs/                  # train / env / data / lora YAML configs
├── data/                     # Bundled ATLAS + mdCATH + combined data; data-prep scripts
├── analysis/                 # Flexibility / embedding "universe" plotting (see analysis/README.md)
└── slurm/                    # Example SLURM job scripts
```

## Installation

Tested with **Python 3.11**.

```bash
# Preferred: uv (https://docs.astral.sh/uv/)
uv sync
source .venv/bin/activate

# Alternative: pip
pip install -r requirements.txt
```

Then edit `configs/env_config.yaml`:

- `huggingface.HF_HOME` — cache dir for the downloaded backbones
  (ProstT5 ≈ 5.5 GB, SaProt-650M ≈ 2.5 GB). The first run downloads them from the HuggingFace Hub.
- `gpus.cuda_visible_device` — which GPU to use.

A CUDA GPU is strongly recommended (the backbones are large). The code falls back to CPU, which
is only practical for very small inputs.

## Pretrained weights

The trained weights are hosted on the HuggingFace Hub at
[`Honzus24/Flexpert-v2`](https://huggingface.co/Honzus24/Flexpert-v2):

```bash
python3 download_flexpert_weights.py                 # ATLAS-trained (default) → models/weights/
python3 download_flexpert_weights.py --dataset all   # all six checkpoints
```

Each backbone has three checkpoints, one per training dataset:

| | ATLAS (default) | mdCATH | combined |
|---|---|---|---|
| Flexpert-Seq (ProstT5) | `flexpert_seq_prostt5_aa_weights.bin` | `flexpert_seq_prostt5_aa_mdcath_weights.bin` | `flexpert_seq_prostt5_aa_combined_weights.bin` |
| Flexpert-3D (SAProt) | `flexpert_3d_saprot_weights.bin` | `flexpert_3d_saprot_mdcath_weights.bin` | `flexpert_3d_saprot_combined_weights.bin` |

`--dataset {atlas,mdcath,combined,all}` selects which to fetch. Alternatively, pass `--weights_path`
directly to any local checkpoint (`results/results_<run_name>_<timestamp>/final_model/pytorch_model.bin`).

## Quickstart: inference

`run_inference.py` reads a FASTA, keeps the chosen split of a `*_splits.json` file, and writes a
text file with one `>NAME` header line followed by comma-separated per-residue predictions.

```bash
# Flexpert-Seq (ProstT5, AA fasta)
python3 run_inference.py --backbone prostt5 \
    --weights_path models/weights/flexpert_seq_prostt5_aa_weights.bin \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path prediction_results/seq_atlas_test.txt

# Flexpert-3D (SAProt, SA-pair fasta)
python3 run_inference.py --backbone saprot \
    --weights_path models/weights/flexpert_3d_saprot_weights.bin \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path prediction_results/3d_atlas_test.txt
```

`--split` accepts `train` / `validation` / `test` (default `test`). To predict for every sequence in
a FASTA, point `--splits_path` at a JSON whose chosen split lists all the sequence names.

## How the models work

```
Flexpert-Seq:  AA sequence ──→ ProstT5  + LoRA ──→ per-residue embedding ──→ linear head ──→ RMSF
Flexpert-3D:   3Di+AA pairs ──→ SAProt   + LoRA ──→ per-residue embedding ──→ linear head ──→ RMSF
```

- **LoRA + MultiplicativeScaling.** LoRA adapters are injected into the attention projections; each
  is wrapped by `MultiplicativeScaling` (`utils/lora_utils.py`). The base PLM is frozen — only LoRA,
  the scalings, and LayerNorm parameters are trained.
- **Masked MSE loss.** `ENMAdaptedTrainer` masks padding and residues whose RMSF label is the
  missing/disordered sentinel (> 900), which is mapped to `-100`.
- **Metrics.** Spearman/Pearson correlation and MSE; the best checkpoint is selected by Spearman.

## Data

A small, ready-to-run subset of [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) ships in `data/`:

| File | Description |
|------|-------------|
| `atlas_sequences.fasta` | AA sequences (Flexpert-Seq input) |
| `atlas_sa_sequences.fasta` | SA-pair sequences — interleaved uppercase AA + lowercase 3Di (Flexpert-3D input) |
| `rmsf_atlas_data_prottransready.txt` | per-residue RMSF labels (`NAME:\tval1, val2, ...`) |
| `atlas_splits.json` | `{"train": [...], "validation": [...], "test": [...]}` of `PDBID_CHAIN` names |
| `PDBs/` | 10 example PDB structures |
| `custom_dataset/` | tiny end-to-end example (sequences + `chain_set.jsonl`) |
| `atlas/precomputed_flexibility_profiles/` | ANM/GNM/ESM-pLDDT baselines for the correlation table |

### Bundled datasets

Three datasets ship under `data/`, each with the same four files — swap the `--data_path` /
`--fasta_path` / `--splits_path` arguments to train or evaluate on any of them:

| Dataset | AA fasta (Seq) | SA-pair fasta (3D) | RMSF labels | Splits |
|---------|----------------|--------------------|-------------|--------|
| ATLAS    | `atlas_sequences.fasta` | `atlas_sa_sequences.fasta` | `rmsf_atlas_data_prottransready.txt` | `atlas_splits.json` |
| mdCATH   | `mdCATH_sequences.fasta` | `mdCATH_sa_sequences.fasta` | `rmsf_mdCATH_data.txt` | `mdCATH_splits.json` |
| combined | `combined_sequences.fasta` | `combined_sa_sequences.fasta` | `rmsf_combined_data.txt` | `combined_splits.json` |

`combined` = ATLAS + mdCATH. The bundled files are the sequences, labels and splits needed for
training/inference; the raw mdCATH MD trajectories (HDF5, ~1.5 TB) are **not** shipped — fetch them
from the public dataset [`compsciencelab/mdCATH`](https://huggingface.co/datasets/compsciencelab/mdCATH)
with `python3 data/mdCATH/download_mdcath.py` only if you need the mdCATH correlation analysis.

### Preparing your own dataset

Paths for the steps below are set in `configs/data_config.yaml`. Run all commands from the repo root.

1. **PDB → `chain_set.jsonl`** (sequences + backbone coordinates):
   ```bash
   python3 data/scripts/prepare_dataset.py
   ```
2. **SA-pair FASTA for Flexpert-3D** (3Di tokens from structure). SAProt needs structural-alphabet
   sequences, produced with [Foldseek](https://github.com/steineggerlab/foldseek):
   ```bash
   python3 data/scripts/sa_prot_seqs.py     # requires a foldseek binary on PATH
   ```
   (The bundled `data/atlas_sa_sequences.fasta` lets you run the Flexpert-3D example without Foldseek.)
3. **RMSF labels from ATLAS** (optional — the preprocessed labels are already bundled). Downloads MD
   analyses via the ATLAS API (large, slow) and extracts per-residue RMSF:
   ```bash
   python3 data/atlas/download_analyses.py
   python3 data/scripts/extract_rmsf_labels.py
   ```
   If you use ATLAS, please cite [Meersche et al.](https://academic.oup.com/nar/article/52/D1/D384/7438909).

## Training

Defaults in `configs/train_config.yaml` point at the bundled ATLAS data; CLI flags override them.
Set `WANDB_MODE=offline` if you do not want to log to Weights & Biases.

```bash
# Flexpert-Seq (ProstT5, AA-only)
python3 train.py --run_name atlasFlexpertProstT5 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json

# Flexpert-3D (SAProt) — note the SA-pair fasta
python3 train_3d.py --run_name atlasFlexpertSaprot \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json
```

Checkpoints are written to `results/results_<run_name>_<timestamp>/final_model/pytorch_model.bin`,
which is exactly the path you pass to `run_inference.py --weights_path`. See `slurm/` for example
batch scripts (adapt the partition/account/environment lines to your cluster).

## Evaluation

**Per-protein correlations** between predictions and ground-truth RMSF:

```bash
python3 evaluate_ablations.py \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --splits_path data/atlas_splits.json \
    --pred_files \
        "Flexpert-Seq:prediction_results/seq_atlas_test.txt" \
        "Flexpert-3D:prediction_results/3d_atlas_test.txt"
```

**Baseline correlation table (paper Table 1).** Compares Flexpert against ANM/GNM, ESM-pLDDT, AF2
pLDDT and crystallographic B-factors. This needs the per-protein ATLAS analysis files, so run
`python3 data/atlas/download_analyses.py` first.

```bash
python3 get_correlation_analysis.py                 # over the whole ATLAS set
python3 get_correlation_analysis.py --evaluate_flexpert   # over the test split, incl. Flexpert columns
```

`--evaluate_flexpert` reads the prediction files named in `configs/data_config.yaml`
(`prediction_results/seq_atlas_test.txt` and `prediction_results/3d_atlas_test.txt` by default).

**mdCATH correlation analysis.** `get_correlation_analysis_mdCATH.py` reports the inter-replica
reliability ceiling (the maximum correlation achievable given MD noise across the 5 replicas) and
the Flexpert-Seq Pearson correlation on mdCATH. It reads the per-domain HDF5 files under
`data/mdCATH/mdCATH/data/`, which are not bundled — download them first
(`python3 data/mdCATH/download_mdcath.py`, optionally `--codes_file data/mdCATH/mdCATH/pdbs_list.txt`).

```bash
python3 get_correlation_analysis_mdCATH.py                  # reliability ceiling only
python3 get_correlation_analysis_mdCATH.py --evaluate_flexpert   # + Flexpert-Seq Pearson
```

For `--evaluate_flexpert`, set `flexpert_seq_predictions_path` in `configs/data_config.yaml` to your
mdCATH predictions (e.g. run `run_inference.py --backbone prostt5 --fasta_path data/mdCATH_sequences.fasta
--splits_path data/mdCATH_splits.json` first).

## Flexibility & embedding universes

`analysis/` reproduces the large-scale 2D maps of protein flexibility (t-SNE / UMAP over predicted
flexibility profiles or per-residue embeddings). These run over very large prediction sets; see
[`analysis/README.md`](analysis/README.md) for the full workflow and a `--limit` smoke test.

## Citation

```bibtex
@inproceedings{kouba2025learning,
  title     = {Learning to engineer protein flexibility},
  author    = {Petr Kouba and Joan Planas-Iglesias and Jiri Damborsky and Jiri Sedlar and Stanislav Mazurenko and Josef Sivic},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=L238BAx0wP}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- The LoRA fine-tuning of protein language models is derived from
  [Schmirler et al.](https://www.nature.com/articles/s41467-024-51844-2)
  ([ProtTrans fine-tuning repo](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning)).
- Flexibility labels come from the [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) MD dataset
  ([Meersche et al.](https://academic.oup.com/nar/article/52/D1/D384/7438909)).
- Backbones: [ProstT5](https://huggingface.co/Rostlab/ProstT5) and
  [SaProt](https://huggingface.co/westlake-repl/SaProt_650M_PDB).
