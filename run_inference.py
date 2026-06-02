#!/usr/bin/env python3
"""Unified Flexpert inference — predict per-residue flexibility (RMSF).

Two backbones:

* ``--backbone prostt5``  → Flexpert-Seq (AA-only FASTA, e.g. data/atlas_sequences.fasta)
* ``--backbone saprot``   → Flexpert-3D (SA-pair FASTA, e.g. data/atlas_sa_sequences.fasta)

The dataset is filtered to a chosen split (default ``test``) of a ``*_splits.json`` file.
Output is a text file: one ``>NAME`` header line followed by comma-separated per-residue
predictions.

Examples
--------
    # Flexpert-Seq (ProstT5, AA-only)
    python3 run_inference.py --backbone prostt5 \
        --weights_path models/weights/flexpert_seq_prostt5_aa_weights.bin \
        --fasta_path data/atlas_sequences.fasta --splits_path data/atlas_splits.json \
        --split test --output_path prediction_results/seq_atlas_test.txt

    # Flexpert-3D (SAProt, SA-pair)
    python3 run_inference.py --backbone saprot \
        --weights_path models/weights/flexpert_3d_saprot_weights.bin \
        --fasta_path data/atlas_sa_sequences.fasta --splits_path data/atlas_splits.json \
        --split test --output_path prediction_results/3d_atlas_test.txt
"""
import os
import re
import json
import argparse

import yaml
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.T5_encoder_per_token import ProstT5_classification_model, SAProt_classification_model
from utils.utils import (
    ClassConfig, DataCollatorForTokenRegression, DataCollatorForTokenRegression3D,
    create_dataset, create_dataset_3d, split_sa_pair,
)

_INFERENCE_CONFIG = {
    'num_labels': 1,
    'dropout_rate': 0.2,
}
_UNCOMMON = re.compile('[OUBZ-]')


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Flexpert inference')
    parser.add_argument('--backbone', required=True, choices=['prostt5', 'saprot'],
                        help='prostt5 → Flexpert-Seq (AA fasta); saprot → Flexpert-3D (SA-pair fasta).')
    parser.add_argument('--weights_path', required=True,
                        help='Path to pytorch_model.bin (sharded checkpoints supported).')
    parser.add_argument('--fasta_path', required=True)
    parser.add_argument('--splits_path', required=True)
    parser.add_argument('--split', default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


def load_weights(model, weights_path, device):
    index_path = weights_path + '.index.json'
    if os.path.exists(index_path):
        print(f'Loading sharded checkpoint from {os.path.dirname(weights_path)}')
        with open(index_path) as f:
            index = json.load(f)
        state_dict = {}
        model_dir = os.path.dirname(weights_path)
        for shard_file in set(index['weight_map'].values()):
            state_dict.update(torch.load(os.path.join(model_dir, shard_file), map_location=device))
    else:
        state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded weights — {len(missing)} missing, {len(unexpected)} unexpected keys')


def clean_aa(seq):
    return _UNCOMMON.sub('X', seq.upper())


def clean_sa_pair(seq):
    """Clean uncommon AAs at AA positions (even indices); leave 3Di (odd) untouched."""
    chars = list(seq)
    for i in range(0, len(chars), 2):
        if chars[i] in 'OUBZ-':
            chars[i] = 'X'
    return ''.join(chars)


def main():
    args = parse_args()

    env_config = yaml.load(open('configs/env_config.yaml'), Loader=yaml.FullLoader)
    os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    os.environ['CUDA_VISIBLE_DEVICES'] = env_config['gpus']['cuda_visible_device']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_config = ClassConfig(dict(_INFERENCE_CONFIG))

    if args.backbone == 'prostt5':
        model, tokenizer = ProstT5_classification_model(False, class_config, lora_r=4)
    else:
        model, tokenizer = SAProt_classification_model(False, class_config)

    load_weights(model, args.weights_path, device)
    model = model.to(device)
    model.eval()

    # Filter FASTA to the requested split.
    with open(args.splits_path) as f:
        splits = json.load(f)
    wanted = set(splits[args.split])

    all_seqs = {}
    for record in SeqIO.parse(args.fasta_path, 'fasta'):
        if record.name in wanted:
            all_seqs[record.name] = str(record.seq)
    names = list(all_seqs.keys())
    print(f'Running inference on {len(names)} proteins (backbone={args.backbone})')

    if args.backbone == 'prostt5':
        # AA-only, space-separated, no directional prefix. If an SA-pair fasta was passed
        # by mistake, pull the AA stream out first.
        aa_inputs = []
        for n in names:
            s = all_seqs[n]
            if any(c.islower() for c in s) and any(c.isupper() for c in s):
                s = split_sa_pair(s)[0]
            aa_inputs.append(' '.join(clean_aa(s)))
        dummy = [[0.0] for _ in aa_inputs]
        dataset = create_dataset(tokenizer, aa_inputs, dummy)
        collator = DataCollatorForTokenRegression(tokenizer)
    else:
        sa_inputs = [clean_sa_pair(all_seqs[n]) for n in names]
        dummy = [[0.0] for _ in sa_inputs]
        dataset = create_dataset_3d(tokenizer, sa_inputs, dummy)
        collator = DataCollatorForTokenRegression3D(tokenizer)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    all_preds = {}
    protein_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            bsz = batch['input_ids'].shape[0]
            inputs = {k: v.to(device) for k, v in batch.items()
                      if isinstance(v, torch.Tensor) and k != 'labels'}
            logits = model(**inputs).logits  # (B, seq_len, 1)
            for i in range(bsz):
                name = names[protein_idx]
                protein_idx += 1
                n_real = int(batch['attention_mask'][i].sum().item())
                if args.backbone == 'prostt5':
                    # ProstT5 (no prefix): [r1, ..., rN, </s>] -> residues at 0 .. n_real-2.
                    pred = logits[i, :n_real - 1, 0].cpu().numpy()
                else:
                    # SAProt: [<cls>, aa0, ..., aaN-1, <eos>] -> residues at 1 .. n_real-2.
                    pred = logits[i, 1:n_real - 1, 0].cpu().numpy()
                all_preds[name] = pred

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, 'w') as f:
        for name in names:
            f.write(f'>{name}\n')
            f.write(', '.join(str(v) for v in all_preds[name]) + '\n')

    print(f'Saved {len(all_preds)} predictions to {args.output_path}')


if __name__ == '__main__':
    main()
