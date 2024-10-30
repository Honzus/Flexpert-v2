from data_utils import parse_PDB, align_pdb_dict_formats
import os
import re
import json
import yaml

fold_dirs = yaml.load(open('configs/data_config.yaml'))['pdb_dir']

fold_list = []
fold_files = os.listdir(fold_dir)
fold_files = [filename for filename in fold_files if re.match(".*\.pdb$", filename)]

for file in fold_files:
    _name= file.split('_')[0]
    _chain = file.split('_')[1]
    _path = f'{fold_dir}/{file}'
    old_pdb = parse_PDB(_path,name=_name, input_chain_list=[_chain])[0]
    new_pdb = align_pdb_dict_formats(old_pdb,_chain)
    fold_list.append(new_pdb)

with open(f'{fold_dir}/chain_set.jsonl','w') as f:
    for dict in fold_list:
        json.dump(dict,f)
        f.write('\n')

