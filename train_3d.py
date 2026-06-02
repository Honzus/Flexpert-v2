#!/usr/bin/env python3
"""Train Flexpert-3D: a SAProt per-residue flexibility (RMSF) regressor.

SAProt consumes structure-aware "SA-pair" sequences (interleaved uppercase AA +
lowercase 3Di, one token per residue). The encoder is LoRA fine-tuned. This release
is ENM-free: structural information comes entirely from the 3Di tokens.

Example
-------
    python3 train_3d.py --run_name atlasFlexpertSaprot \
        --data_path data/rmsf_atlas_data_prottransready.txt \
        --fasta_path data/atlas_sa_sequences.fasta \
        --splits_path data/atlas_splits.json
"""
import os
import argparse
from datetime import datetime

import yaml
import pandas as pd
from Bio import SeqIO
import wandb
from transformers import TrainingArguments

from utils.utils import (
    ClassConfig, ENMAdaptedTrainer, set_seeds, create_dataset_3d,
    DataCollatorForTokenRegression3D, do_topology_split, update_config,
    compute_metrics, save_finetuned_model,
)
from models.T5_encoder_per_token import SAProt_classification_model

_UNCOMMON = set('OUBZ-')


def clean_sa_pair(seq):
    """Replace uncommon AAs (O/U/B/Z/-) with X at AA positions (even indices) only,
    leaving the lowercase 3Di tokens (odd indices) untouched."""
    chars = list(seq)
    for i in range(0, len(chars), 2):
        if chars[i] in _UNCOMMON:
            chars[i] = 'X'
    return ''.join(chars)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flexpert-3D (SAProt).')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run.')
    parser.add_argument('--adaptor_architecture', type=str, default='no-adaptor',
                        choices=['no-adaptor'],
                        help='Only the structure-only (no-adaptor) head is supported in this release.')
    parser.add_argument('--data_path', type=str, help='RMSF label file (NAME:\\tval1, val2, ...).')
    parser.add_argument('--fasta_path', type=str, help='SA-pair FASTA (interleaved AA + 3Di).')
    parser.add_argument('--splits_path', type=str, help='JSON file with train/validation/test splits.')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--mixed_precision', action='store_true', help='Enable fp16 training.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open('configs/train_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = update_config(config, args)

    config['training_args']['run_name'] = config['run_name']
    config['training_args']['output_dir'] = config['training_args']['output_dir'].format(
        run_name=config['run_name'],
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    config['training_args']['fp16'] = config['mixed_precision']
    config['training_args']['gradient_accumulation_steps'] = config['gradient_accumulation_steps']
    config['training_args']['num_train_epochs'] = config['epochs']
    config['training_args']['per_device_train_batch_size'] = config['batch_size']
    config['training_args']['per_device_eval_batch_size'] = config['batch_size']
    config['training_args']['eval_steps'] = config['training_args']['save_steps']

    print("Training with the following config:\n", config)

    env_config = yaml.load(open('configs/env_config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    os.environ['CUDA_VISIBLE_DEVICES'] = env_config['gpus']['cuda_visible_device']

    wandb.init(project=env_config['wandb']['project'], name=config['run_name'], config=config)

    # --- Load SA-pair sequences + labels into a dataframe ---
    sequences = []
    with open(config['fasta_path'], 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append([record.name, str(record.seq)])
    df = pd.DataFrame(sequences, columns=['name', 'sequence'])

    names, labels = [], []
    with open(config['data_path'], 'r') as f:
        for line in f:
            _split_line = line.split(":\t")
            names.append(_split_line[0])
            labels.append([float(v) for v in _split_line[1].split(", ")])
    label_map = dict(zip(names, labels))
    df['label'] = df['name'].map(label_map)
    df = df[df['label'].notna()].reset_index(drop=True)
    df['sequence'] = df['sequence'].map(clean_sa_pair)

    set_seeds(config['seed'])

    class_config = ClassConfig(config)
    model, tokenizer = SAProt_classification_model(
        half_precision=config['mixed_precision'], class_config=class_config)

    train, valid, test = do_topology_split(df, config['splits_path'])
    # Mask >900 RMSF sentinels (missing/disordered residues) with -100.
    for split_df in (train, valid):
        split_df['label'] = split_df.apply(
            lambda row: [-100 if x > 900 else x for x in row['label']], axis=1)

    train_set = create_dataset_3d(tokenizer, list(train['sequence']), list(train['label']))
    valid_set = create_dataset_3d(tokenizer, list(valid['sequence']), list(valid['label']))

    training_args = TrainingArguments(**config['training_args'])
    data_collator = DataCollatorForTokenRegression3D(tokenizer)

    trainer = ENMAdaptedTrainer(
        model,
        training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_finetuned_model(trainer.model, config['training_args']['output_dir'])
