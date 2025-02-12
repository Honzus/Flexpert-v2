from data.scripts.data_utils import parse_PDB
from utils.utils import ClassConfig, DataCollatorForTokenRegression, process_in_batches_and_combine
from models.T5_encoder_per_token import PT5_classification_model
from data.scripts.get_enm_fluctuations_for_dataset import get_fluctuation_for_json_dict
import argparse
import os
import yaml
import torch
from Bio import SeqIO
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--modality", type=str, required=True, help="Indicate 'Seq' or '3D' to use Flexpert-Seq or Flexpert-3D?")
    parser.add_argument("--splits_file", type=str, required=False, help="Path to the file defining the splits, in case that input_file is a dataset which should be subsampled.")
    parser.add_argument("--split", type=str, required=False, help="Specify test/train/val to subselect the respective split. If specified, the splits file needs to be provided as well.")
    parser.add_argument("--output_enm", action='store_true', help="If true, the ENM values will be outputted in separate file(s).")
    args = parser.parse_args()

    args.modality = args.modality.upper()
    filename, suffix = os.path.splitext(args.input_file)
    
    if args.modality not in ["SEQ", "3D"]:
        raise ValueError("Modality must be either Seq or 3D")
    if args.splits_file is not None and args.split is None:
        raise ValueError("If splits_file is provided, split must be specified.")
    if args.split is not None and args.splits_file is None:
        raise ValueError("If split is specified, splits_file must be provided.")
    if args.split is not None and args.split not in ["test", "train", "val", "validation"]:
        raise ValueError("Split must be either 'test', 'train', 'val' or 'validation'")
    if args.output_enm and (args.modality not in ["3D"]):
        raise ValueError("Output ENM is only supported for 3D modality")

    if args.splits_file is not None:
        with open(args.splits_file, 'r') as f:
            splits = json.load(f)
        if 'val' in splits.keys() and args.split == 'validation':
            args.split = 'val'
        elif 'validation' in splits.keys() and args.split == 'val':
            args.split = 'validation'
        
        datapoint_for_eval = splits[args.split]
    else:
        datapoint_for_eval = 'all'

    if suffix == ".fasta":
        if args.modality == "3D":
            raise ValueError("Flexpert-3D needs the structure, fasta is not enough")

        sequences = []
        names = []
        backbones = []
        # Load FASTA file using Biopython
        for record in SeqIO.parse(args.input_file, "fasta"):
            if '_' in record.name:
                dot_separated_name = '.'.join(record.name.split('_'))
            elif '.' in record.name:
                dot_separated_name = record.name
            else:
                raise ValueError("Sequence name must contain either an underscore or a dot to separate the PDB code and the chain code.")
            if datapoint_for_eval == 'all' or dot_separated_name in datapoint_for_eval:
                names.append(dot_separated_name)
                sequences.append(str(record.seq))
                backbones.append(None)

    elif suffix == ".pdb":
        parsed_name = filename.split('/')[-1].split('_')
        if len(parsed_name[0]) != 4 or len(parsed_name[1]) != 1 or not parsed_name[1].isalpha():
            raise ValueError("PDB file name is expected to be in the format of 'name_chain.pdb', e.g.: 1BUI_C.pdb")
        _name= parsed_name[0]
        _chain = parsed_name[1]
        parsed_pdb = parse_PDB(args.input_file,name=_name, input_chain_list=[_chain])[0]
        backbone, sequence = parsed_pdb['coords_chain_{}'.format(_chain)], parsed_pdb['seq_chain_{}'.format(_chain)]
        backbones = [backbone]
        sequences = [sequence]
        names = [_name+"."+_chain]#[_name+"_"+_chain]
    elif suffix == ".jsonl":
        sequences = []
        names = []
        backbones = []
        for line in open(args.input_file, 'r'):
            _dict = json.loads(line)

            if '_' in _dict['name']:
                dot_separated_name = '.'.join(_dict['name'].split('_'))
            elif '.' in record.name:
                dot_separated_name = _dict['name']
            else:
                raise ValueError("Sequence name must contain either an underscore or a dot to separate the PDB code and the chain code.")

            if datapoint_for_eval == 'all' or dot_separated_name in datapoint_for_eval:
                backbones.append(_dict['coords'])
                sequences.append(_dict['seq'])
                names.append(dot_separated_name)
    else:
        raise ValueError("Input file must be a fasta, pdb or jsonl file")

    ### Set environment variables
    env_config = yaml.load(open('configs/env_config.yaml', 'r'), Loader=yaml.FullLoader)
    # Set folder for huggingface cache
    os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]= env_config['gpus']['cuda_visible_device']

    config = yaml.load(open('configs/train_config.yaml', 'r'), Loader=yaml.FullLoader)
    class_config=ClassConfig(config)
    class_config.adaptor_architecture = 'no-adaptor' if args.modality == 'SEQ' else 'conv'
    model, tokenizer = PT5_classification_model(half_precision=config['mixed_precision'], class_config=class_config)

    model.to(config['inference_args']['device'])
    if args.modality == 'SEQ':
        state_dict = torch.load(config['inference_args']['seq_model_path'], map_location=config['inference_args']['device'])
        model.load_state_dict(state_dict, strict=False)
    elif args.modality == '3D':
        state_dict = torch.load(config['inference_args']['3d_model_path'], map_location=config['inference_args']['device'])
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    data_to_collate = []
    for backbone, sequence in zip(backbones, sequences):
        if args.modality == '3D':
            _dict = {'coords': backbone, 'seq': sequence}
            flucts, _ = get_fluctuation_for_json_dict(_dict, enm_type = config['inference_args']['enm_type'])
            flucts = flucts.tolist()
            flucts.append(0.0) #To match the special token for the sequence
            flucts = torch.tensor(flucts).to(config['inference_args']['device'])
        
        tokenizer_out = tokenizer(' '.join(sequence), add_special_tokens=True, return_tensors='pt')
        tokenized_seq, attention_mask = tokenizer_out['input_ids'].to(config['inference_args']['device']), tokenizer_out['attention_mask'].to(config['inference_args']['device'])
        
        if args.modality == '3D':
            data_to_collate.append({'input_ids': tokenized_seq[0,:], 'attention_mask': attention_mask[0,:], 'enm_vals': flucts})
        elif args.modality == 'SEQ':
            data_to_collate.append({'input_ids': tokenized_seq[0,:], 'attention_mask': attention_mask[0,:]})

    # Use the data collator to process the input
    data_collator = DataCollatorForTokenRegression(tokenizer)
    batch = data_collator(data_to_collate)  # Wrap in list since collator expects batch
    batch.to(model.device)
    
    # Predict
    with torch.no_grad():
        output_logits = process_in_batches_and_combine(model, batch, config['inference_args']['batch_size'])
        predictions = output_logits[:,:,0] #includes the prediction for the added token
        # subselect the predictions using the attention mask
    
    output_filename = config['inference_args']['prediction_output_dir'].format(filename.split('/')[-1], args.modality, 'all' if not args.split else args.split)

    with open(output_filename, 'w') as f:
        print("Saving predictions to {}.".format(output_filename))
        for prediction, mask, name, sequence in zip(predictions, batch['attention_mask'], names, sequences):
            prediction = prediction[mask.bool()]
            assert len(prediction) == len(sequence)+1
            f.write('>' + name + '\n')
            f.write(', '.join([str(p) for p in prediction.tolist()[:-1]]) + '\n')
    
    if suffix == ".pdb":
        pdb_output_filename = output_filename.replace('.txt', '.pdb')
        with open(pdb_output_filename, 'w') as f:
            print("Saving prediction to {}.".format(pdb_output_filename))
            from data.scripts.data_utils import modify_bfactor_biotite
            chain_id = parsed_name[1]
            modify_bfactor_biotite(args.input_file, chain_id, pdb_output_filename, predictions[:,:-1]) #writing the prediction without the last token

    if args.output_enm:
        enm_txt_output_filename = output_filename.replace('.txt', '_enm.txt')
        with open(enm_txt_output_filename, 'w') as f:
            print("Saving ENM predictions to {}.".format(enm_txt_output_filename))
            for enm_prediction, name, sequence in zip(batch['enm_vals'], names, sequences):
                f.write('>' + name + '\n')
                f.write(', '.join([str(p) for p in enm_prediction.tolist()[:-1]]) + '\n')
    
        if suffix == ".pdb":
            enm_pdb_output_filename = enm_txt_output_filename.replace('.txt', '.pdb')
            with open(enm_pdb_output_filename, 'w') as f:
                print("Saving ENM prediction to {}.".format(enm_pdb_output_filename))
                from data.scripts.data_utils import modify_bfactor_biotite
                chain_id = parsed_name[1]
                modify_bfactor_biotite(args.input_file, chain_id, enm_pdb_output_filename, batch['enm_vals'][:,:-1]) #writing the prediction without the last token