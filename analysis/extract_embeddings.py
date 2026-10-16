#!/usr/bin/env python3
"""Extract per-residue ProstT5 embeddings (final encoder hidden state, before the head).

Mirrors run_inference.py's ProstT5 (Flexpert-Seq) path, but instead of writing the
regression-head logits it captures the encoder's final hidden state — the tensor fed into
the regression head, shape (L, 1024) per protein — and stores it to HDF5.

Output is an .h5 file with one float16 dataset per protein, keyed by PDBID, so the
embeddings line up 1:1 with the per-residue flexibilities in a matching predictions file
produced by `run_inference.py --backbone prostt5`. These feed the embedding-universe plots
(see analysis/README.md).

Run from the repository root:
    python3 -m analysis.extract_embeddings --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
        --fasta_path data/atlas_sequences.fasta --splits_path data/atlas_splits.json --split test \
        --output_path outputs/predictions/flexpert_seq_atlas_test_embeddings.h5
"""
import os
import json
import argparse

# Must run before torch/transformers are imported so HF_HOME etc. take effect.
from utils.env import setup_env
setup_env()

import h5py
import torch
import numpy as np
from Bio import SeqIO
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.T5_encoder_per_token import ProstT5_classification_model
from utils.utils import ClassConfig, DataCollatorForTokenRegression, create_dataset
from run_inference import load_weights, clean_aa, _INFERENCE_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Extract per-residue ProstT5 embeddings to HDF5')
    parser.add_argument('--weights_path', required=True,
                        help='Path to Flexpert-Seq pytorch_model.bin (sharded models supported)')
    parser.add_argument('--fasta_path', required=True, help='AA-only FASTA')
    parser.add_argument('--splits_path', default=None,
                        help='Optional splits JSON. If omitted, every FASTA record is extracted.')
    parser.add_argument('--split', default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--output_path', required=True,
                        help='Output .h5 file (one dataset per protein, keyed by PDBID)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    parser.add_argument('--compression', default='gzip', choices=['gzip', 'lzf', 'none'])
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class_config = ClassConfig(dict(_INFERENCE_CONFIG))
    model, tokenizer = ProstT5_classification_model(False, class_config, lora_r=4)
    load_weights(model, args.weights_path, device)
    model = model.to(device)
    model.eval()

    # Capture the input to the regression head (encoder final hidden state), shape (B, L, 1024).
    captured = {}
    model.classifier.register_forward_pre_hook(lambda m, a: captured.__setitem__('emb', a[0]))

    if args.splits_path is not None:
        with open(args.splits_path) as f:
            splits = json.load(f)
        wanted = set(splits[args.split])
    else:
        wanted = None

    all_seqs = {}
    for record in SeqIO.parse(args.fasta_path, 'fasta'):
        if wanted is None or record.name in wanted:
            all_seqs[record.name] = str(record.seq)
    names = list(all_seqs.keys())
    print(f'Extracting embeddings for {len(names)} proteins (backbone=prostt5)')

    aa_inputs = [' '.join(clean_aa(all_seqs[n])) for n in names]
    dummy = [[0.0] for _ in aa_inputs]
    # max_length=None: no truncation. T5's relative attention handles sequences longer
    # than the 1024-token training cap, and downstream embedding consumers want the
    # full per-residue tensor.
    dataset = create_dataset(tokenizer, aa_inputs, dummy, max_length=None)
    collator = DataCollatorForTokenRegression(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    np_dtype = np.float16 if args.dtype == 'float16' else np.float32
    compression = None if args.compression == 'none' else args.compression

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    protein_idx = 0
    written = 0
    with h5py.File(args.output_path, 'a') as h5, torch.no_grad():
        for batch in tqdm(loader):
            bsz = batch['input_ids'].shape[0]
            inputs = {k: v.to(device) for k, v in batch.items()
                      if isinstance(v, torch.Tensor) and k != 'labels'}
            model(**inputs)                      # populates `captured['emb']` via the hook
            emb = captured['emb']                # (B, L, 1024)

            for i in range(bsz):
                name = names[protein_idx]
                protein_idx += 1
                # ProstT5 layout [r1, ..., rN, </s>] (no prefix): residue rows are 0 .. n_real-2.
                n_real = int(batch['attention_mask'][i].sum().item())
                emb_i = emb[i, :n_real - 1, :].to(torch.float32).cpu().numpy().astype(np_dtype)
                if name in h5:                   # allow resume
                    continue
                h5.create_dataset(name, data=emb_i, compression=compression)
                written += 1

        if 'names' in h5:
            del h5['names']
        h5.create_dataset('names', data=np.array(names, dtype=h5py.string_dtype()))

    print(f'Saved embeddings for {written} proteins ({len(names)} requested) to {args.output_path}')


if __name__ == '__main__':
    main()
