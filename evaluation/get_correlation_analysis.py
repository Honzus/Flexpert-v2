"""ATLAS correlation analysis (Table 1 of the paper).

Averages per-protein Pearson correlation matrices between the MD RMSF and other flexibility
proxies (B-factor, AF2/ESMFold pLDDT, ProDy ANM/GNM). With ``--evaluate_flexpert`` it restricts
to the ATLAS test split and adds the Flexpert predictions:

  * Flexpert v1 Seq/3D (ProtT5) — the original Flexpert paper's predictions, shipped in
    data/atlas/precomputed_flexibility_profiles/flexpert_v1_prott5_{seq,3d}_atlas_test.txt;
  * Flexpert-Seq / Flexpert-3D (v2) — read from outputs/predictions/flexpert_{seq,3d}_atlas_test.txt
    (produce them with run_inference.py); skipped with a message if absent.

Needs the per-protein ATLAS analyses (RMSF/B-factor/pLDDT TSVs) under data/atlas/analyses —
download them with data/atlas/download_analyses.py. Paths come from configs/data_config.yaml.

Run from the repository root:
    python3 -m evaluation.get_correlation_analysis --evaluate_flexpert
"""
import pickle
import numpy as np
import json
import pandas as pd
from data.scripts.extract_rmsf_labels import extract_rmsf_labels, extract_bfactor_labels, extract_plddt_labels
import yaml
from tqdm import tqdm
import os
def get_flucts_from_pickle(f):
    return pickle.load(f)

def get_flucts_from_jsonl(f):
    _flucts = f.readlines()
    pdb_code_to_fluct_dict = {}
    for line in _flucts:
        json_obj = json.loads(line.strip())
        pdb_code_to_fluct_dict[json_obj['pdb_name']] = np.array(json_obj['fluctuations'])
    return pdb_code_to_fluct_dict

def read_flexpert_predictions(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        pdb_code_to_fluct_dict = {}

        for name_line, fluct_line in zip(lines[::2], lines[1::2]):
            _name = name_line.strip().strip('>')
            if '.' in _name:
                _name = _name.replace('.', '_')
            pdb_code_to_fluct_dict[_name] = np.array(fluct_line.strip().split(','), dtype=np.float32)
    return pdb_code_to_fluct_dict

# (column, config key, display label) for the Flexpert prediction columns.
FLEXPERT_V1 = [
    ('flexpert_v1_seq', 'flexpert_v1_seq_predictions_path', 'Flexpert v1 Seq (ProtT5)'),
    ('flexpert_v1_3d', 'flexpert_v1_3d_predictions_path', 'Flexpert v1 3D (ProtT5)'),
]
FLEXPERT_V2 = [
    ('flexpert_seq', 'flexpert_seq_predictions_path', 'Flexpert-Seq'),
    ('flexpert_3d', 'flexpert_3d_predictions_path', 'Flexpert-3D'),
]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evaluate_flexpert', action='store_true', default=False)
    args = parser.parse_args()

    config = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)
    DATA_DIR = config['precomputed_flexibility_profiles_dir']

    flexpert_predictions = {}   # column -> {pdb_code: np.ndarray}
    labels = {c: c for c in ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']}
    if args.evaluate_flexpert:
        for col, key, label in FLEXPERT_V1:
            path = config[key]
            assert os.path.exists(path), f"{label} predictions file does not exist: {path}"
            flexpert_predictions[col] = read_flexpert_predictions(path)
            labels[col] = label
        for col, key, label in FLEXPERT_V2:
            path = config[key]
            if not os.path.exists(path):
                print(f"Skipping {label}: predictions file not found ({path}); "
                      f"produce it with run_inference.py (see README).")
                continue
            flexpert_predictions[col] = read_flexpert_predictions(path)
            labels[col] = label

    with open(f'{DATA_DIR}/anm_square_fluctuations.pickle','rb') as f:
        anm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/gnm_square_fluctuations.pickle','rb') as f:
        gnm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/atlas_esm_plddt.jsonl','rb') as f:
        esm_plddt = get_flucts_from_jsonl(f)

    atlas_list_path = config['pdb_codes_path']
    atlas_analyses_dir = config['atlas_out_dir']

    atlas_bfactor_path = atlas_analyses_dir + "/{}/{}_Bfactor.tsv"
    atlas_plddt_path = atlas_analyses_dir + "/{}/{}_pLDDT.tsv"
    atlas_rmsf_path = atlas_analyses_dir + "/{}/{}_RMSF.tsv"

    with open(atlas_list_path,'r') as f:
        atlas_list = f.readlines()
        atlas_list = [a.strip() for a in atlas_list]

    fluctuations = {}

    if args.evaluate_flexpert:
        print("Filtering to the ATLAS test set to evaluate: "
              + ", ".join(labels[c] for c in flexpert_predictions))
        atlas_list = [a for a in atlas_list
                      if all(a in preds for preds in flexpert_predictions.values())]

    for key in tqdm(atlas_list):
        fluctuations[key] = pd.DataFrame()
        fluctuations[key]['prody_ANM'] = np.sqrt(anm_sqFlucts.get(key, np.nan))
        fluctuations[key]['prody_GNM'] = np.sqrt(gnm_sqFlucts.get(key, np.nan))
        fluctuations[key]['esm_plddt'] = 1 - esm_plddt.get(key, np.nan)
        fluctuations[key]['rmsf'] = extract_rmsf_labels(atlas_rmsf_path.format(key, key))[1]
        fluctuations[key]['bfactor'] = extract_bfactor_labels(atlas_bfactor_path.format(key, key))[1]
        fluctuations[key]['af2_plddt'] = 1 - extract_plddt_labels(atlas_plddt_path.format(key, key))[1]
        for col, preds in flexpert_predictions.items():
            # Predictions may be shorter than the chain (inference truncates at 1024 tokens);
            # pad with NaN so the correlation uses the residues that were predicted.
            pred = np.asarray(preds.get(key), dtype=np.float64)[:len(fluctuations[key])]
            padded = np.full(len(fluctuations[key]), np.nan)
            padded[:len(pred)] = pred
            fluctuations[key][col] = padded

    columns = ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']
    columns += list(flexpert_predictions)

    pearson_correlations = []

    for pdb_code,df in fluctuations.items():
        pc = df[columns].corr(method='pearson')
        if  np.any(np.isnan(pc)):
            print(f'{pdb_code} has NaN values in Pearson correlation')
            continue
        pearson_correlations.append(pc)

    #compute average across all pdb codes
    names = [labels[c] for c in columns]
    print(f"Pearson correlations (mean over {len(pearson_correlations)} proteins):")
    pearson_mean = np.mean(pearson_correlations, axis=0)
    pearson_mean_rounded = np.round(pearson_mean, 2)
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        print(pd.DataFrame(pearson_mean_rounded, index=names, columns=names))
    print("\n")
