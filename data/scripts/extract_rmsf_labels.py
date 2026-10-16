r"""Readers for ATLAS / mdCATH flexibility labels, and the ATLAS RMSF label-file builder.

As a script it rebuilds the ATLAS label file from the per-protein ATLAS analyses
(data/atlas/analyses/<pdb>_<chain>/<pdb>_<chain>_RMSF.tsv, fetched by data/atlas/download_analyses.py)
in the format every reader in this repo expects — one line per chain, ``NAME:\tv1, v2, ...``
with ``NAME`` like ``1k5n.A``:

    python3 -m data.scripts.extract_rmsf_labels --out data/my_atlas_labels.txt
"""
import os
import argparse

import yaml
import pandas as pd
import h5py
import numpy as np


def _atlas_protein_id(file_path):
    """data/atlas/analyses/1k5n_A/1k5n_A_RMSF.tsv -> '1k5n.A' (the name used in FASTAs/splits)."""
    folder = os.path.basename(os.path.dirname(file_path))
    return '.'.join(folder.rsplit('_', 1))


def format_label_line(protein_id, values):
    r"""One label-file line: ``NAME:\tv1, v2, ...`` (readers split on ':\t' and ', ')."""
    return f"{protein_id}:\t{', '.join(map(str, values))}\n"


def extract_rmsf_labels(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        protein_id = _atlas_protein_id(file_path)
        rmsf_values = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            rmsf_r1 = float(parts[1])
            rmsf_r2 = float(parts[2])
            rmsf_r3 = float(parts[3])
            avg_rmsf = (rmsf_r1 + rmsf_r2 + rmsf_r3) / 3
            rmsf_values.append(avg_rmsf)
    return protein_id, rmsf_values

def extract_rmsf_labels_mdCATH(file_path):
    f = h5py.File(file_path, 'r')
    keys = list(f.keys())
    protein = f[keys[0]]
    
    rmsfs = np.array([protein["320"][f"{i}"]["rmsf"] for i in range(5)])
    rmsf = np.mean(rmsfs, axis = 0)

    protein_name = protein.name[1:]
    
    return protein_name, rmsf

def get_all_rmsf_labels_mdCATH(file_path):
    f = h5py.File(file_path, 'r')
    keys = list(f.keys())
    protein = f[keys[0]]
    
    rmsfs = np.array([protein["320"][f"{i}"]["rmsf"] for i in range(5)])

    protein_name = protein.name[1:]
    
    return protein_name, rmsfs

def extract_bfactor_labels(file_path):
    bfactor = pd.read_csv(file_path, delimiter='\t')['Bfactor']
    protein_id = _atlas_protein_id(file_path)
    return protein_id, bfactor

def extract_plddt_labels(file_path):
    plddt = pd.read_csv(file_path, delimiter='\t')['pLDDT']
    protein_id = _atlas_protein_id(file_path)
    return protein_id, plddt

if __name__ == "__main__":
    config = yaml.load(open('configs/data_config.yaml'), Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_dir', default=config['atlas_out_dir'],
                        help='ATLAS analyses directory (default: atlas_out_dir in configs/data_config.yaml).')
    parser.add_argument('--out', default=config['atlas_labels_path'],
                        help='Output label file (default: atlas_labels_path in configs/data_config.yaml).')
    parser.add_argument('--force', action='store_true', help='Overwrite --out if it exists.')
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.force:
        parser.error(f'{args.out} exists; pass --force to overwrite it or choose another --out.')

    rmsf_data = {}
    for folder in sorted(os.listdir(args.in_dir)):
        folder_path = os.path.join(args.in_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_RMSF.tsv"):
                    file_path = os.path.join(folder_path, file)
                    protein_id, rmsf_labels = extract_rmsf_labels(file_path)
                    rmsf_data[protein_id] = rmsf_labels

    with open(args.out, 'w') as out_file:
        for protein_id, values in rmsf_data.items():
            out_file.write(format_label_line(protein_id, values))
    print(f"Wrote {len(rmsf_data)} label lines to {args.out}")
