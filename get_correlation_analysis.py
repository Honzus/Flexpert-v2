import pickle
import numpy as np
import json
import pandas as pd
from data.scripts.extract_rmsf_labels import extract_rmsf_labels, extract_bfactor_labels, extract_plddt_labels
import yaml
from tqdm import tqdm

def get_flucts_from_pickle(f):
    return pickle.load(f)

def get_flucts_from_jsonl(f):
    _flucts = f.readlines()
    pdb_code_to_fluct_dict = {}
    for line in _flucts:
        json_obj = json.loads(line.strip())
        pdb_code_to_fluct_dict[json_obj['pdb_name']] = np.array(json_obj['fluctuations'])
    return pdb_code_to_fluct_dict

if __name__ == "__main__":
    
    DATA_DIR = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['precomputed_flexibility_profiles_dir']

    with open(f'{DATA_DIR}/anm_square_fluctuations.pickle','rb') as f:
        anm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/gnm_square_fluctuations.pickle','rb') as f:
        gnm_sqFlucts = get_flucts_from_pickle(f)

    with open(f'{DATA_DIR}/atlas_esm_plddt.jsonl','rb') as f:
        esm_plddt = get_flucts_from_jsonl(f)

    atlas_list_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['pdb_codes_path']
    atlas_analyses_dir = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['atlas_out_dir']

    atlas_bfactor_path = atlas_analyses_dir + "/{}_analysis/{}_Bfactor.tsv"
    atlas_plddt_path = atlas_analyses_dir + "/{}_analysis/{}_pLDDT.tsv"
    atlas_rmsf_path = atlas_analyses_dir + "/{}_analysis/{}_RMSF.tsv"

    with open(atlas_list_path,'r') as f:
        atlas_list = f.readlines()
        atlas_list = [a.strip() for a in atlas_list]

    fluctuations = {}
    for key in tqdm(atlas_list):
        fluctuations[key] = pd.DataFrame({
            'prody_ANM': np.sqrt(anm_sqFlucts.get(key, np.nan)),
            'prody_GNM': np.sqrt(gnm_sqFlucts.get(key, np.nan)),
            'esm_plddt': 1 - esm_plddt.get(key, np.nan),
            'rmsf': extract_rmsf_labels(atlas_rmsf_path.format(key, key))[1],
            'bfactor': extract_bfactor_labels(atlas_bfactor_path.format(key, key))[1],
            'af2_plddt': 1 - extract_plddt_labels(atlas_plddt_path.format(key, key))[1]
        })

    pearson_correlations = []
    spearman_correlations = []

    for pdb_code,df in fluctuations.items():
        cols = ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']

        pc = df[cols].corr(method='pearson')
        sc = df[cols].corr(method='spearman')
        if  np.any(np.isnan(pc)):
            print(f'{pdb_code} has NaN values in Pearson correlation')
            continue
        pearson_correlations.append(pc)
        spearman_correlations.append(sc)
        
    #compute average across all pdb codes
    columns = ['rmsf', 'bfactor', 'af2_plddt', 'esm_plddt', 'prody_GNM', 'prody_ANM']
    print("Pearson correlations:")
    pearson_mean = np.mean(pearson_correlations, axis=0)
    pearson_mean_rounded = np.round(pearson_mean, 2)
    print(pd.DataFrame(pearson_mean_rounded, index=columns, columns=columns))
    print("\n")
    print("Spearman correlations:")
    spearman_mean = np.mean(spearman_correlations, axis=0)
    spearman_mean_rounded = np.round(spearman_mean, 2)
    print(pd.DataFrame(spearman_mean_rounded, index=columns, columns=columns))
    print("\n")
    #TODO: load the Flexpert-3D and Flexpert-Seq predictions for testset and evaluate the correlations
