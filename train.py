#import dependencies
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import random
import warnings
import json
import tempfile
import matplotlib.pyplot as plt
from io import StringIO
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from Bio import SeqIO
from tqdm import tqdm

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import transformers
from transformers import T5EncoderModel, T5Tokenizer, TrainingArguments, Trainer, set_seed
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from evaluate import load
from datasets import Dataset

import wandb
import argparse

from utils.lora_utils import LoRAConfig, modify_with_lora
from utils.utils import ClassConfig, ENMAdaptedTrainer, set_seeds, create_dataset, save_finetuned_model, DataCollatorForTokenRegression, do_topology_split, update_config
from models.T5_encoder_per_token import PT5_classification_model, T5EncoderForTokenClassification
from models.enm_adaptor_heads import ENMAdaptedAttentionClassifier, ENMAdaptedDirectClassifier, ENMAdaptedConvClassifier, ENMNoAdaptorClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the CATH dataset')
    # Required arguments
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run.')
    parser.add_argument('--adaptor_architecture', type=str, required=True, help='What model to use to adapt the ENM values into the sequence latent space.')

    # Optional arguments
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, help='Size of the batch for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training.')
    parser.add_argument('--save_steps', type=int, help='After how many training steps to save the checkpoint.')
    parser.add_argument('--add_pearson_loss', action='store_true', help='If provided, Pearson correlation term will be added to the loss function.')
    parser.add_argument('--add_sse_loss', action='store_true', help='If provided, term forcing the model to predict same values along sse blocks will be added to the loss function.')
    parser.add_argument('--fasta_path', type=str, help='Path to the FASTA file with the AA sequences for the dataset.')
    parser.add_argument('--enm_path', type=str, help='Path to the enm file with precomputed flexibilities (ENM).')
    parser.add_argument('--splits_path', type=str, help='Path to the file with the data splits.')
    
    #Optional ENM adaptor arguments
    parser.add_argument('--enm_embed_dim', type=int, help='Dimension of the ENM embedding / number of conv filters.')
    parser.add_argument('--enm_att_heads', type=int, help='Number of attention heads for the ENM embedding.')
    parser.add_argument('--num_layers', type=int, help='Number of conv layers in the ENM adaptor.')
    parser.add_argument('--kernel_size', type=int, help='Size of the convolutional kernels in the ENM adaptor.')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of steps to accumulate gradients before performing a backward/update pass.')
    return parser.parse_args()
    
# Main training function
def train_per_residue(
        run_name,         #name of the run
        train_df,         #training data
        valid_df,         #validation data      
        num_labels= 1,    #number of classes
    
        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8 
        batch= 4,         #for training
        accum= 2,         #gradient accumulation
    
        val_batch = 16,   #batch size for evaluation
        epochs= 10,       #training epochs
        lr= 3e-4,         #recommended learning rate
        seed= 42,         #random seed
        mixed= False,     #enable mixed precision training  
        gpu= 1,
        save_steps=528,
        add_pearson_loss = False,
        add_sse_loss = False,
        adaptor_architecture = None,
        enm_embed_dim = None,
        enm_att_heads = None,
        num_layers = None,
        kernel_size = None):         #gpu selection (1 for first gpu)

    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu-1)
    
    # Set all random seeds
    set_seeds(seed)
    
    # load model
    class_config=ClassConfig(num_labels=num_labels, add_pearson_loss=add_pearson_loss, add_sse_loss=add_sse_loss, adaptor_architecture = adaptor_architecture, enm_embed_dim = enm_embed_dim, enm_att_heads = enm_att_heads, num_layers = num_layers, kernel_size = kernel_size)
    model, tokenizer = PT5_classification_model(half_precision=mixed, class_config=class_config)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["sequence"]=train_df["sequence"].str.replace('|'.join(["O","B","U","Z","-"]),"X",regex=True)
    valid_df["sequence"]=valid_df["sequence"].str.replace('|'.join(["O","B","U","Z","-"]),"X",regex=True)
    # Add spaces between each amino acid for PT5 to correctly use them
    train_df['sequence']=train_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    valid_df['sequence']=valid_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)


    # Create Datasets
    train_set=create_dataset(tokenizer,list(train_df['sequence']),list(train_df['label']),list(train_df['enm_vals']))
    valid_set=create_dataset(tokenizer,list(valid_df['sequence']),list(valid_df['label']),list(valid_df['enm_vals']))
    
    # Huggingface Trainer arguments
    args = TrainingArguments(
        output_dir = f"./results/results_{run_name}",
        run_name = run_name,
        evaluation_strategy = "steps",
        eval_steps = save_steps,
        # evaluation_strategy = "epoch",
        save_strategy = "steps",
        load_best_model_at_end=True,         # Load the best model (based on metric) at the end of training
        metric_for_best_model='spearmanr',    # The metric to use to compare models
        greater_is_better=True,           # Defines whether higher values of the above metric are better
        save_total_limit=3,     
        logging_strategy = "epoch",
        save_steps = save_steps,
        #save_strategy = "no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        #per_device_eval_batch_size=val_batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed,
        fp16 = mixed,
        save_safetensors = False
    ) 

    # Metric definition for validation data
    def compute_metrics(eval_pred):

        predictions, labels = eval_pred
        predictions=predictions.flatten()
        labels=labels.flatten()

        valid_labels=labels[np.where((labels != -100 ) & (labels < 900 ))]
        valid_predictions=predictions[np.where((labels != -100 ) & (labels < 900 ))]
        #assuming the ENM vals are subtracted from the labels for correct evaluation
        spearman = load("spearmanr")
        pearson = load("pearsonr")
        mse = load("mse")
        return {"spearmanr": spearman.compute(predictions=valid_predictions, references=valid_labels)['spearmanr'],
                "pearsonr": pearson.compute(predictions=valid_predictions, references=valid_labels)['pearsonr'],
                "mse": mse.compute(predictions=valid_predictions, references=valid_labels)['mse']}

        #return metric.compute(predictions=valid_predictions, references=valid_labels)

    # For token classification we need a data collator here to pad correctly
    data_collator = DataCollatorForTokenRegression(tokenizer) 

    # Trainer          
    trainer = ENMAdaptedTrainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    save_finetuned_model(trainer.model,"./results_"+run_name)
    
    return tokenizer, model, trainer.state.log_history

if __name__=='__main__':
    args = parse_args()
    config = yaml.load(open('configs/train_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = update_config(config, args)
    print("Training with the following config: \n", config)

    env_config = yaml.load(open('configs/env_config.yaml', 'r'), Loader=yaml.FullLoader)

    # Set HF_HOME
    os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    # Initialize wandb
    wandb.init(project=env_config['wandb']['project'], name=config['run_name'], config = config)

    # Load data
    DATA_PATH = config['data_path']
    FASTA_PATH = config['fasta_path']
    ENM_PATH = config['enm_path']
    SPLITS_PATH = config['splits_path']

    sequences, names, labels, enm_vals = [], [], [], []

    with open(FASTA_PATH, "r") as fasta_file:
        # Load FASTA file using Biopython
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append([record.name, str(record.seq)])
        # Create dataframe
        df = pd.DataFrame(sequences, columns=["name", "sequence"])

    with open(ENM_PATH, "r") as f:
        enm_lines = f.readlines()
        enm_vals_dict={}
        for l in enm_lines:
            _d = json.loads(l)
            _key = ".".join(_d['pdb_name'].split("_"))
            enm_vals_dict[_key] = _d['fluctuations']
            enm_vals_dict[_key].append(0.0)

    with open(DATA_PATH, "r") as f:
        lines = f.readlines()
        # Split each line into name and label
        for l in lines:
            _split_line = l.split(":\t")
            names.append(_split_line[0])
            labels.append([float(label) for label in _split_line[1].split(", ")])
            enm_vals.append(enm_vals_dict[_split_line[0]])

    # Add label column
    df["label"] = labels
    # warnings.warn("The labels are now the RMSF values directly!")
    df["enm_vals"] = enm_vals

    train,valid,test = do_topology_split(df, SPLITS_PATH)
    train = train[["sequence", "label", "enm_vals"]]
    valid = valid[["sequence", "label", "enm_vals"]]
    test = test[["sequence", "label", "enm_vals"]]
    
    train.reset_index(drop=True,inplace=True)
    valid.reset_index(drop=True,inplace=True)
    test.reset_index(drop=True,inplace=True)

    # Replace invalid labels (>900) with -100 (will be ignored by pytorch loss)
    #TODO: necessary to mask some ENM values?
    train['label'] = train.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)
    valid['label'] = valid.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)
    test['label'] = test.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)
    
    tokenizer, model, history = train_per_residue(args.run_name, train, valid, num_labels=1, batch=args.batch_size, accum=args.gradient_accumulation_steps, 
                                                  epochs=args.epochs, seed=42, gpu=1, mixed = args.mixed_precision, save_steps=args.save_steps, 
                                                  add_pearson_loss=args.add_pearson_loss, add_sse_loss=args.add_sse_loss,
                                                  adaptor_architecture = args.adaptor_architecture, enm_embed_dim = args.enm_embed_dim,
                                                  enm_att_heads = args.enm_att_heads, num_layers = args.num_layers, kernel_size = args.kernel_size)


