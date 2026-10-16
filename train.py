#!/usr/bin/env python3
"""Train Flexpert-Seq: a sequence-only per-residue flexibility (RMSF) regressor.

``--backbone prostt5`` (default) fine-tunes ProstT5; ``--backbone prott5`` the original Flexpert
backbone ProtT5 (Rostlab/prot_t5_xl_uniref50). The encoder is LoRA fine-tuned; AA sequences are
fed space-separated with no directional prefix. The model is sequence-only (ENM-free).

Example
-------
    python3 train.py --run_name atlasFlexpertProstT5 \
        --data_path data/rmsf_atlas_data_prottransready.txt \
        --fasta_path data/atlas_sequences.fasta \
        --splits_path data/atlas_splits.json

The released ProtT5 weights used the same recipe (configs/train_config.yaml: batch 8, lr 3e-4,
100 epochs, seed 42, LoRA r4/alpha1) except that validation/checkpointing ran every 256 steps
(512 for mdCATH) instead of every epoch:
    python3 train.py --backbone prott5 --run_name atlasFlexpertProtT5 --save_steps 256

The final weights are written to outputs/checkpoints/<run_name>_<timestamp>/final_model/
pytorch_model.bin (load them with run_inference.py --weights_path). Smoke test:
``--max_steps 4`` trains a few steps without evaluation/checkpointing, then saves final_model.
"""
import os
import json
import argparse
from datetime import datetime

# Must run before torch/transformers are imported so HF_HOME etc. take effect.
from utils.env import setup_env
ENV = setup_env()

import yaml
import pandas as pd
from Bio import SeqIO
import wandb
from transformers import TrainingArguments

from utils.utils import (
    ClassConfig, ENMAdaptedTrainer, set_seeds, create_dataset,
    DataCollatorForTokenRegression, do_topology_split, update_config, compute_metrics,
    save_finetuned_model,
)
from models.T5_encoder_per_token import ProstT5_classification_model, ProtT5_classification_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flexpert-Seq (ProstT5 or ProtT5, AA-only).')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run.')
    parser.add_argument('--backbone', type=str, default='prostt5', choices=['prostt5', 'prott5'],
                        help='prostt5 (default) or prott5 (Rostlab/prot_t5_xl_uniref50).')
    parser.add_argument('--adaptor_architecture', type=str, default='no-adaptor',
                        choices=['no-adaptor'],
                        help='Only the sequence-only (no-adaptor) head is supported in this release.')
    parser.add_argument('--data_path', type=str, help='RMSF label file (NAME:\\tval1, val2, ...).')
    parser.add_argument('--fasta_path', type=str, help='FASTA file with AA sequences.')
    parser.add_argument('--splits_path', type=str, help='JSON file with train/validation/test splits.')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_steps', type=int,
                        help='Validate and checkpoint every N steps instead of every epoch.')
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--mixed_precision', action='store_true', help='Enable fp16 training.')
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank (default: 4).')
    parser.add_argument('--lora_alpha', type=int, default=1, help='LoRA alpha (default: 1).')
    parser.add_argument('--seed', type=int, help='Random seed (default: configs/train_config.yaml).')
    parser.add_argument('--max_steps', type=int,
                        help='Smoke mode: train this many steps with no evaluation/checkpointing; '
                             'final_model is still saved.')
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
    if config.get('save_steps'):
        config['training_args'].update(eval_strategy='steps', save_strategy='steps',
                                       save_steps=config['save_steps'])
    config['training_args']['eval_steps'] = config['training_args']['save_steps']
    config['training_args']['seed'] = config['seed']
    if os.environ.get('WANDB_MODE') == 'disabled':
        config['training_args']['report_to'] = 'none'
    else:
        config['training_args'].setdefault('report_to', 'wandb')
    if config.get('max_steps'):
        # Smoke mode: a few optimiser steps, no eval/checkpoints (final_model is still saved).
        config['training_args'].update(
            max_steps=config['max_steps'], eval_strategy='no', save_strategy='no',
            load_best_model_at_end=False, logging_strategy='steps', logging_steps=1)

    print("Training with the following config:\n", config)

    if 'wandb' in str(config['training_args']['report_to']):
        wandb.init(project=ENV['wandb']['project'], name=config['run_name'], config=config)

    # --- Load sequences + labels into a dataframe ---
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

    set_seeds(config['seed'])

    class_config = ClassConfig(config)
    factory = ProtT5_classification_model if config['backbone'] == 'prott5' else ProstT5_classification_model
    model, tokenizer = factory(
        half_precision=config['mixed_precision'], class_config=class_config,
        lora_r=config['lora_r'], lora_alpha=config['lora_alpha'])

    train, valid, test = do_topology_split(df, config['splits_path'])
    # Mask >900 RMSF sentinels (missing/disordered residues) with -100; clean uncommon AAs;
    # and space-separate residues for the T5 tokenizer (no directional prefix).
    for split_df in (train, valid):
        split_df['label'] = split_df.apply(
            lambda row: [-100 if x > 900 else x for x in row['label']], axis=1)
        split_df['sequence'] = split_df['sequence'].str.replace(
            '|'.join(['O', 'B', 'U', 'Z', '-']), 'X', regex=True)
        split_df['sequence'] = split_df['sequence'].apply(lambda s: ' '.join(s))

    train_set = create_dataset(tokenizer, list(train['sequence']), list(train['label']))
    valid_set = create_dataset(tokenizer, list(valid['sequence']), list(valid['label']))

    training_args = TrainingArguments(**config['training_args'])
    data_collator = DataCollatorForTokenRegression(tokenizer)

    trainer = ENMAdaptedTrainer(
        model,
        training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_finetuned_model(trainer.model, config['training_args']['output_dir'])
