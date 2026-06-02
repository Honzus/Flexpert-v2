#!/usr/bin/env python3
"""Correlation analysis for the mdCATH dataset.

For each mdCATH domain it reads the five MD RMSF replicas (320 K) from the per-domain HDF5 file
and reports:
  1. an inter-replica reliability ceiling — the maximum correlation any predictor could reach given
     the variance between MD replicas; and
  2. (with --evaluate_flexpert) the Pearson correlation between the replica-averaged RMSF and the
     Flexpert-Seq predictions named in configs/data_config.yaml.

Requires the mdCATH per-domain HDF5 files under data/mdCATH/mdCATH/data/ — these are NOT shipped;
download them with data/mdCATH/download_mdcath.py (public dataset compsciencelab/mdCATH).
"""
import os
import argparse

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from data.scripts.extract_rmsf_labels import extract_rmsf_labels_mdCATH, get_all_rmsf_labels_mdCATH

MDCATH_RMSF_PATH = "data/mdCATH/mdCATH/data/mdcath_dataset_{}.h5"


def read_flexpert_predictions(path):
    preds = {}
    with open(path) as f:
        lines = f.readlines()
    for name_line, fluct_line in zip(lines[::2], lines[1::2]):
        name = name_line.strip().lstrip('>').replace('.', '_')
        preds[name] = np.array(fluct_line.strip().split(','), dtype=np.float32)
    return preds


def reliability_ceiling(y_ijk):
    """Upper bound on achievable correlation given 5-replica MD variance (ICC-style).

    y_ijk[i] is a (5, L_i) array of per-residue RMSF for the 5 replicas of protein i.
    """
    N = len(y_ijk)
    y_ik = [[sum(col) / 5 for col in zip(*block)] for block in y_ijk]   # per-residue replica mean
    G = sum(len(v) for v in y_ik)
    grand_mean = sum(sum(v) for v in y_ik) / (N * G)
    sum_between = sum((y_ik[i][k] - grand_mean) ** 2
                      for i in range(N) for k in range(len(y_ik[i])))
    sum_within = sum((y_ijk[i][j][k] - y_ik[i][k]) ** 2
                     for i in range(N) for j in range(5) for k in range(len(y_ijk[i][j])))
    phi_x = (1 / 5) * (((1 / (N - 1)) * (5 * sum_between)) - ((1 / (4 * N)) * sum_within))
    phi_eps = (1 / (4 * N)) * sum_within
    return np.sqrt(phi_x / (phi_x + phi_eps / 5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate_flexpert', action='store_true', default=False)
    args = parser.parse_args()

    config = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)

    flexpert_seq_predictions = {}
    if args.evaluate_flexpert:
        seq_path = config['flexpert_seq_predictions_path']
        assert os.path.exists(seq_path), f"Flexpert-Seq predictions file does not exist: {seq_path}"
        flexpert_seq_predictions = read_flexpert_predictions(seq_path)

    with open(config['mdCATH_pdb_codes_path']) as f:
        mdcath_list = [line.strip()[:-4] for line in f if line.strip()]   # strip ".pdb"

    if args.evaluate_flexpert:
        mdcath_list = [a for a in mdcath_list if a in flexpert_seq_predictions]

    fluctuations = {}
    y_ijk = []
    for key in tqdm(mdcath_list):
        rmsf = extract_rmsf_labels_mdCATH(MDCATH_RMSF_PATH.format(key))[1]
        y_ijk.append(get_all_rmsf_labels_mdCATH(MDCATH_RMSF_PATH.format(key))[1])
        if args.evaluate_flexpert and key in flexpert_seq_predictions:
            pred = flexpert_seq_predictions[key]
            n = min(len(rmsf), len(pred))
            fluctuations[key] = pd.DataFrame({'rmsf': rmsf[:n], 'flexpert_seq': pred[:n]})
        else:
            fluctuations[key] = pd.DataFrame({'rmsf': rmsf})

    print(f"Inter-replica reliability ceiling (max achievable correlation): "
          f"{reliability_ceiling(y_ijk):.4f}")

    if args.evaluate_flexpert:
        cols = ['rmsf', 'flexpert_seq']
        pcs = []
        for df in fluctuations.values():
            if not all(c in df.columns for c in cols):
                continue
            pc = df[cols].corr(method='pearson')
            if not np.any(np.isnan(pc.values)):
                pcs.append(pc)
        print("\nPearson correlations (mdCATH):")
        print(pd.DataFrame(np.round(np.mean(pcs, axis=0), 2), index=cols, columns=cols))
