#!/usr/bin/env python3
"""Download pretrained Flexpert weights from the HuggingFace Hub into models/weights/.

    python3 download_flexpert_weights.py                 # atlas-trained (default)
    python3 download_flexpert_weights.py --dataset mdcath
    python3 download_flexpert_weights.py --dataset all   # all six checkpoints

Weights live at https://huggingface.co/Honzus24/Flexpert-v2 :
  Flexpert-Seq (ProstT5, AA-only): flexpert_seq_prostt5_aa[_<dataset>]_weights.bin
  Flexpert-3D  (SAProt):           flexpert_3d_saprot[_<dataset>]_weights.bin
where <dataset> is `mdcath` or `combined`; the atlas-trained checkpoints are unsuffixed.
Pass the downloaded path to run_inference.py via --weights_path.
"""
import os
import argparse

from huggingface_hub import hf_hub_download

REPO_ID = "Honzus24/Flexpert-v2"
DEST = "models/weights"

WEIGHTS = {
    "atlas":    ["flexpert_seq_prostt5_aa_weights.bin",          "flexpert_3d_saprot_weights.bin"],
    "mdcath":   ["flexpert_seq_prostt5_aa_mdcath_weights.bin",   "flexpert_3d_saprot_mdcath_weights.bin"],
    "combined": ["flexpert_seq_prostt5_aa_combined_weights.bin", "flexpert_3d_saprot_combined_weights.bin"],
}


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="atlas",
                    choices=["atlas", "mdcath", "combined", "all"],
                    help="Which training dataset's weights to fetch (default: atlas).")
    return ap.parse_args()


def main():
    args = parse_args()
    datasets = ["atlas", "mdcath", "combined"] if args.dataset == "all" else [args.dataset]
    os.makedirs(DEST, exist_ok=True)
    for ds in datasets:
        for filename in WEIGHTS[ds]:
            print(f"Downloading {filename} from {REPO_ID} ...")
            path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=DEST)
            print(f"  -> {path}")
    print(f"Done. Weights are in {DEST}/.")


if __name__ == "__main__":
    main()
