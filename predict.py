from data.scripts.data_utils import parse_PDB

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--modality", type=str, required=True, help="Indicate 'Seq' or '3D' to use Flexpert-Seq or Flexpert-3D?")
    args = parser.parse_args()

    args.modality = args.modality.upper()
    filename, suffix = os.path.splitext(args.input_file)
        
    if args.modality not in ["SEQ", "3D"]:
        raise ValueError("Modality must be either Seq or 3D")

    if suffix == ".fasta":
        if args.modality == "3D":
            raise ValueError("Flexpert-3D needs the structure, fasta is not enough")
        #TODO parse fasta file
        pass
    elif suffix == ".pdb":
        parsed_name = filename.split('/')[-1].split('_')
        if len(parsed_name[0]) != 4 or len(parsed_name[1]) != 1 or not parsed_name[1].isalpha():
            raise ValueError("PDB file name is expected to be in the format of 'name_chain.pdb', e.g.: 1BUI_C.pdb")
        _name= parsed_name[0]
        _chain = parsed_name[1]
        parsed_pdb = parse_PDB(args.input_file,name=_name, input_chain_list=[_chain])[0]
        backbone, sequence = parsed_pdb['coords_chain_{}'.format(_chain)], parsed_pdb['seq_chain_{}'.format(_chain)]
    elif suffix == ".jsonl":
        #TODO parse jsonl file
        pass
    else:
        raise ValueError("Input file must be a fasta, pdb or jsonl file")

    #load model Seq / 3D and the tokenizer
