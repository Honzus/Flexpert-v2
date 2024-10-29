from data_utils import parse_PDB, align_pdb_dict_formats
import os
import re
import json


# pdb_dict = parse_PDB('data/fold_1/1ceeB2k42A_1cee_B.pdb',name='test', input_chain_list=['B'])[0] #CAREFUL THE DSSP WAS DEACTIVATED DUE TO ERRORS
# new_pdb_dict = align_pdb_dict_formats(pdb_dict,'B')
# print(new_pdb_dict.keys())

#fold_dirs = ['data/atlas/distant-frame-pairs_NO_SUPERPOSITION/frames_1','data/atlas/distant-frame-pairs_NO_SUPERPOSITION/frames_2']
fold_dirs = ['atlas_eval_proteinmpnn/atlas_full/minimized_PDBs','atlas_eval_proteinmpnn/atlas_full/refolded_PDBs']
DISTANT_FRAMES = True

for fold_dir in fold_dirs:
    fold_list = []
    fold_files = os.listdir(fold_dir)
    fold_files = [filename for filename in fold_files if re.match(".*\.pdb$", filename)]

    for file in fold_files:
        if DISTANT_FRAMES:
            _name= file.split('_')[0]
            _chain = file.split('_')[1]
        else:
            _name= file.split('_')[1]
            _chain = file.split('_')[2].split('.')[0]
        _path = f'{fold_dir}/{file}'
        old_pdb = parse_PDB(_path,name=_name, input_chain_list=[_chain])[0]
        new_pdb = align_pdb_dict_formats(old_pdb,_chain)
        fold_list.append(new_pdb)
    with open(f'{fold_dir}/chain_set.jsonl','w') as f:
        for dict in fold_list:
            json.dump(dict,f)
            f.write('\n')

