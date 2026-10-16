#!/usr/bin/env python3
"""Download pretrained Flexpert weights from the HuggingFace Hub into weights/.

    python3 download_flexpert_weights.py                 # v2 models, atlas-trained (default)
    python3 download_flexpert_weights.py --dataset mdcath
    python3 download_flexpert_weights.py --dataset all   # all six v2 checkpoints
    python3 download_flexpert_weights.py --models all --dataset all   # all nine checkpoints

Weights live at https://huggingface.co/Honzus24/Flexpert-v2 :
  v2      Flexpert-Seq (ProstT5, AA-only): flexpert_seq_prostt5_aa[_<dataset>]_weights.bin
          Flexpert-3D  (SAProt):           flexpert_3d_saprot[_<dataset>]_weights.bin
  prott5  Flexpert-Seq (ProtT5, AA-only):  flexpert_seq_prott5[_<dataset>]_weights.bin
where <dataset> is `mdcath` or `combined`; the atlas-trained checkpoints are unsuffixed.
Pass the downloaded path (e.g. weights/flexpert_seq_prostt5_aa_weights.bin) to run_inference.py
via --weights_path.
"""
import os
import argparse

from huggingface_hub import hf_hub_download

REPO_ID = "Honzus24/Flexpert-v2"
DEST = "weights"
DATASETS = ["atlas", "mdcath", "combined"]

WEIGHTS = {  # models -> dataset -> files
    "v2": {
        "atlas":    ["flexpert_seq_prostt5_aa_weights.bin",          "flexpert_3d_saprot_weights.bin"],
        "mdcath":   ["flexpert_seq_prostt5_aa_mdcath_weights.bin",   "flexpert_3d_saprot_mdcath_weights.bin"],
        "combined": ["flexpert_seq_prostt5_aa_combined_weights.bin", "flexpert_3d_saprot_combined_weights.bin"],
    },
    "prott5": {
        "atlas":    ["flexpert_seq_prott5_weights.bin"],
        "mdcath":   ["flexpert_seq_prott5_mdcath_weights.bin"],
        "combined": ["flexpert_seq_prott5_combined_weights.bin"],
    },
}


def weight_files(models="v2", dataset="atlas"):
    """File names for --models / --dataset (either may be 'all')."""
    model_sets = list(WEIGHTS) if models == "all" else [models]
    datasets = DATASETS if dataset == "all" else [dataset]
    return [f for ds in datasets for m in model_sets for f in WEIGHTS[m][ds]]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="atlas", choices=DATASETS + ["all"],
                    help="Which training dataset's weights to fetch (default: atlas).")
    ap.add_argument("--models", default="v2", choices=list(WEIGHTS) + ["all"],
                    help="v2 = ProstT5 Seq + SaProt 3D (default); prott5 = ProtT5 Seq; all = both.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(DEST, exist_ok=True)
    for filename in weight_files(args.models, args.dataset):
        print(f"Downloading {filename} from {REPO_ID} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=DEST)
        print(f"  -> {path}")
    print(f"Done. Weights are in {DEST}/.")


if __name__ == "__main__":
    main()
