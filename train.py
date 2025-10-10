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
from transformers import T5EncoderModel, T5Tokenizer, TrainingArguments, Trainer, set_seed,  TrainerState, TrainerControl, TrainerCallback
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from datasets import Dataset

import wandb
import argparse
from datetime import datetime

# from utils.lora_utils import LoRAConfig, modify_with_lora
from utils.utils import (
    ClassConfig, ENMAdaptedTrainer, set_seeds, create_dataset, save_finetuned_model, 
    DataCollatorForTokenRegression, do_topology_split, update_config, compute_metrics
)
from models.T5_encoder_per_token import PT5_classification_model, T5EncoderForTokenClassification, ESM2_classification_model, ESM2EncoderForTokenClassification
from models.enm_adaptor_heads import (
    ENMAdaptedAttentionClassifier, ENMAdaptedDirectClassifier, 
    ENMAdaptedConvClassifier, ENMNoAdaptorClassifier
)

def reset_and_get_peak(step):
    """Resets peak tracker and prints the peak memory allocated in the prior step."""
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        
        # Get the maximum recorded allocation since the last reset
        peak_allocated_gb = torch.cuda.max_memory_allocated(device_id) / 1024**3
        
        # Reset the peak tracker so the next call measures the next step only
        torch.cuda.reset_peak_memory_stats(device_id)
        
        print(f"--- GPU {device_id} Peak Status @ End of Step {step} ---")
        print(f"Peak Allocated: {peak_allocated_gb:.2f} GB (The hidden max)")
        print("-" * 45)

def print_cuda_memory_status(step):
    """Prints the currently allocated and reserved GPU memory in GB."""
    if torch.cuda.is_available():
        # Using device 0 as an assumption, adjust if you're using a different GPU
        device_id = torch.cuda.current_device()
        
        # Memory currently occupied by Tensors (what you're actually using)
        allocated_gb = torch.cuda.memory_allocated(device_id) / 1024**3
        
        # Memory reserved by PyTorch's caching allocator (the total chunk size)
        reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
        
        # Total VRAM on the device (for context)
        total_vram_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        
        print(f"\n--- GPU {device_id} Status @ Step {step} ---")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")
        print(f"Allocated:  {allocated_gb:.2f} GB (Actual Tensor Usage)")
        print(f"Reserved:   {reserved_gb:.2f} GB (PyTorch Pool Size)")
        print("-" * 40)

class PeakMonitorCallback(TrainerCallback):
    # This runs at the start of the *first* step only, setting a clean slate
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.is_world_process_zero:
            # Clear historical stats before training starts
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    # This runs *after* the entire forward/backward pass and optimization for the step
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and state.global_step > 0:
            reset_and_get_peak(state.global_step - 1) # Log peak from the step that just finished
            # Note: The `reset_peak_memory_stats` call is essential here!

class HighFrequencyGPUMonitorCallback(TrainerCallback):
    """
    Calls the monitoring function at the start of every training step.
    This provides the maximum possible monitoring frequency.
    """
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only check on the main process (world_process_zero) to prevent flooding 
        # the logs if you are using multi-GPU training.
        if state.is_world_process_zero:
            print_cuda_memory_status(state.global_step)
            
    # Optionally, also monitor after the crash cleanup if the system is still alive
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            print("\nMonitoring at Training End (Post-Cleanup Check):")
            print_cuda_memory_status(state.global_step)

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
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Enable mixed precision training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of steps to accumulate gradients before performing a backward/update pass.')
    return parser.parse_args()

def preprocess_data(tokenizer, train, valid, test):

    train = train[["sequence", "label", "enm_vals"]]
    valid = valid[["sequence", "label", "enm_vals"]]
    test = test[["sequence", "label", "enm_vals"]]
    
    train.reset_index(drop=True,inplace=True)
    valid.reset_index(drop=True,inplace=True)
    test.reset_index(drop=True,inplace=True)

    # Replace invalid labels (>900) with -100 (will be ignored by pytorch loss)
    train['label'] = train.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)
    valid['label'] = valid.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)
    test['label'] = test.apply(lambda row:  [-100 if x > 900 else x for x in row['label']], axis=1)

    # Preprocess inputs for the model
    # Replace uncommon AAs with "X"
    train["sequence"]=train["sequence"].str.replace('|'.join(["O","B","U","Z","-"]),"X",regex=True)
    valid["sequence"]=valid["sequence"].str.replace('|'.join(["O","B","U","Z","-"]),"X",regex=True)
    # Add spaces between each amino acid for PT5 to correctly use them
    train['sequence']=train.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    valid['sequence']=valid.apply(lambda row : " ".join(row["sequence"]), axis = 1)


    # Create Datasets
    train_set=create_dataset(tokenizer,list(train['sequence']),list(train['label']),list(train['enm_vals']))
    valid_set=create_dataset(tokenizer,list(valid['sequence']),list(valid['label']),list(valid['enm_vals']))

    return train_set, valid_set, test

if __name__=='__main__':
    ### Read and update config
    args = parse_args()
    config = yaml.load(open('configs/train_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = update_config(config, args)
    # Update training arguments
    config['training_args']['run_name'] = config['run_name']
    config['training_args']['output_dir'] = config['training_args']['output_dir'].format(
        run_name=config['run_name'],
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    config['training_args']['fp16'] = config['mixed_precision']
    config['training_args']['gradient_accumulation_steps'] = config['gradient_accumulation_steps']
    config['training_args']['num_train_epochs'] = config['epochs']
    config['training_args']['per_device_train_batch_size'] = config['batch_size']
    config['training_args']['per_device_eval_batch_size'] = config['batch_size']
    config['training_args']['eval_steps'] = config['training_args']['save_steps']

    print("Training with the following config: \n", config)

    env_config = yaml.load(open('configs/env_config.yaml', 'r'), Loader=yaml.FullLoader)
    ### Set environment variables
    # Set folder for huggingface cache
    os.environ['HF_HOME'] = env_config['huggingface']['HF_HOME']
    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]= env_config['gpus']['cuda_visible_device']
    
    ### Initialize wandb
    wandb.init(project=env_config['wandb']['project'], name=config['run_name'], config = config)

    ### Load data - into dataframe
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

    # Add label and enm_vals columns
    df["label"] = labels
    df["enm_vals"] = enm_vals
    
    ### Set all random seeds
    set_seeds(config['seed'])
        
    ### Load model
    class_config=ClassConfig(config)
    # model, tokenizer = PT5_classification_model(half_precision=config['mixed_precision'], class_config=class_config)
    print(config['mixed_precision'])
    model, tokenizer = ESM2_classification_model(half_precision=config['mixed_precision'], class_config=class_config)
    ### Split data into train, valid, test and preprocess
    train,valid,test = do_topology_split(df, SPLITS_PATH)
    train_set, valid_set, test = preprocess_data(tokenizer, train, valid, test)
    ### Set training arguments
    training_args = TrainingArguments(**config['training_args'])
    ### For token classification (regression) we need a data collator here to pad correctly
    data_collator = DataCollatorForTokenRegression(tokenizer)
    ### Trainer

    trainer = ENMAdaptedTrainer(
            model,
            training_args,
            train_dataset=train_set,
            eval_dataset=valid_set,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            )

    # 0. Set the model to training mode (important for components like Dropout)
    # model.train()
    
    # # 1. Get a single batch of data from the training dataset
    # train_dataloader = trainer.get_train_dataloader()
    # batch = next(iter(train_dataloader))
    # print("Done 7: Retrieved first data batch")
    
    # # 2. Move the batch to the device
    # device = model.device 
    # batch_on_device = {}
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         batch_on_device[k] = v.to(device)
    #     # CRITICAL: Ensure your 'labels' tensor (derived from 'enm_vals') is on the device
    #     else:
    #         batch_on_device[k] = v

    # # The custom Trainer expects 'labels' and 'attention_mask' on the device
    # # We must also ensure 'labels' is correctly present in the batch, as per your Trainer logic.
    # if 'enm_vals' in batch_on_device and 'labels' not in batch_on_device:
    #     # If your data collator is using 'enm_vals' for the targets, rename it to 'labels' 
    #     # as expected by your custom compute_loss (which explicitly checks for "labels").
    #     batch_on_device['labels'] = batch_on_device.pop('enm_vals')

    # print(f"Done 8: Batch moved to device: {device}. Keys: {batch_on_device.keys()}")

    # # 3. Manually create a simple Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # # Clear any existing gradients
    # optimizer.zero_grad()
    # print("Done 9: Gradients zeroed and Optimizer ready")

    # # 4. Execute a single FORWARD PASS
    # print("Executing single FORWARD pass...")
    # # Call the model with the batch data
    # outputs = model(**batch_on_device)
    # print("Done 10a: Forward pass complete (logits computed).")
    
    # # 5. CRITICAL FIX: MANUALLY COMPUTE THE LOSS (replicating Trainer's compute_loss)
    # # The standard HF loss (outputs.loss) is None because your model is a custom classifier.
    # # We must replicate the logic from ENMAdaptedTrainer.compute_loss here:
    
    # labels = batch_on_device.get("labels")
    # logits = outputs.get('logits')
    # mask = batch_on_device.get('attention_mask')
    # loss_fct = MSELoss()

    # # Apply mask and filter for valid labels/logits, exactly as in your custom method
    # active_loss = mask.view(-1) == 1
    # active_logits = logits.view(-1)
    
    # # The torch.where is used to correctly handle padded tokens (set to -100)
    # # Ensure the -100 is of the correct type (float for regression labels)
    # neg_100 = torch.tensor(-100.0).type_as(labels) 
    # active_labels = torch.where(active_loss, labels.view(-1), neg_100)
    
    # # Filter out padded tokens
    # valid_logits = active_logits[active_labels != -100.0]
    # valid_labels = active_labels[active_labels != -100.0]

    # # Calculate final loss
    # loss = loss_fct(valid_labels, valid_logits)
    
    # print(f"Done 10b: Loss manually computed. Calculated loss: {loss.item():.4f}")

    # # 6. Execute a single BACKWARD PASS
    # # This step calculates and stores gradients.
    # print("Executing single BACKWARD pass...")
    # loss.backward()
    # print("Done 11: Backward pass complete (gradients calculated)")

    #torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    ### Train model and save
    device_id = torch.cuda.current_device()
    
    # Memory currently occupied by Tensors (what you're actually using)
    allocated_gb = torch.cuda.memory_allocated(device_id) / 1024**3
    
    # Memory reserved by PyTorch's caching allocator (the total chunk size)
    reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
    
    # Total VRAM on the device (for context)
    total_vram_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    
    print(f"\n--- GPU {device_id} Status @ Start")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Allocated:  {allocated_gb:.2f} GB (Actual Tensor Usage)")
    print(f"Reserved:   {reserved_gb:.2f} GB (PyTorch Pool Size)")
    print("-" * 40)

    trainer.train()

# This creates a trace that can be viewed in TensorBoard.
    save_finetuned_model(trainer.model,config['training_args']['output_dir'])