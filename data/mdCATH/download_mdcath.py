#!/usr/bin/env python3
"""Download the public mdCATH dataset (HDF5 per-domain MD data) from the HuggingFace Hub.

mdCATH (Mirarchi et al., Sci Data 2024; CC-BY-4.0) is hosted at `compsciencelab/mdCATH`.
The correlation analysis (`get_correlation_analysis_mdCATH.py`) reads per-domain RMSF replicas
from `data/mdCATH/mdCATH/data/mdcath_dataset_<code>.h5`.

WARNING: the full dataset is very large (~1.5 TB of HDF5). Use `--codes_file` to fetch only the
domains you need (e.g. the test split), or `--limit` for a quick subset.

Examples
--------
    # everything (huge)
    python3 data/mdCATH/download_mdcath.py

    # only the codes listed (one per line, e.g. data/mdCATH/mdCATH/pdbs_list.txt)
    python3 data/mdCATH/download_mdcath.py --codes_file data/mdCATH/mdCATH/pdbs_list.txt --workers 16
"""
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "compsciencelab/mdCATH"
# Download into data/mdCATH/mdCATH/ relative to this script (matches the analysis script's path).
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdCATH")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes_file", default=None,
                    help="Optional file of mdCATH codes (one per line) to fetch only those HDF5 files.")
    ap.add_argument("--limit", type=int, default=None, help="Only download the first N matching files.")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    files = list_repo_files(REPO_ID, repo_type="dataset")

    if args.codes_file:
        with open(args.codes_file) as f:
            codes = {line.strip() for line in f if line.strip()}
        # pdbs_list.txt entries look like "<code>.pdb"; strip extension if present.
        codes = {c[:-4] if c.endswith(".pdb") else c for c in codes}
        files = [f for f in files if any(c in f for c in codes)]

    if args.limit:
        files = files[:args.limit]

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Downloading {len(files)} files from {REPO_ID} into {LOCAL_DIR}")

    def _get(filename):
        hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset",
                        local_dir=LOCAL_DIR)
        return filename

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_get, f): f for f in files}
        for fut in as_completed(futures):
            try:
                print("downloaded", fut.result())
            except Exception as e:
                print(f"FAILED {futures[fut]}: {e}")


if __name__ == "__main__":
    main()
