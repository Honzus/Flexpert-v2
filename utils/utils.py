"""Data collators, dataset builders, trainer and metric helpers for Flexpert.

Two input pipelines are supported:

* **ProstT5 (Flexpert-Seq)** — space-separated AA sequences, no directional prefix
  (``create_dataset`` + ``DataCollatorForTokenRegression``).
* **SAProt (Flexpert-3D)** — interleaved 3Di+AA "SA-pair" sequences
  (``create_dataset_3d`` + ``DataCollatorForTokenRegression3D``).
"""
import os
import json
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import MSELoss
from transformers import Trainer, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


@dataclass
class DataCollatorForTokenRegression(DataCollatorMixin):
    """Dynamically pad inputs and per-residue float labels (ProstT5 / T5-style tokenizers)."""

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch


@dataclass
class DataCollatorForTokenRegression3D(DataCollatorMixin):
    """Like the above but prepends a -100 label for the leading <cls> token (ESM/SAProt)."""

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                [self.label_pad_token_id] + to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label) - 1) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch


class ClassConfig:
    """Lightweight config object built from the merged YAML/CLI config dict."""

    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


class ENMAdaptedTrainer(Trainer):
    """HF Trainer with a masked-MSE loss over valid per-residue labels."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        mask = inputs.get('attention_mask')
        loss_fct = MSELoss()

        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1)
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(-100).type_as(labels))
        valid_logits = active_logits[active_labels != -100]
        valid_labels = active_labels[active_labels != -100]

        loss = loss_fct(valid_labels, valid_logits)
        return (loss, outputs) if return_outputs else loss


def set_seeds(s):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# ---------------------------------------------------------------------------
# SAProt (Flexpert-3D) dataset
# ---------------------------------------------------------------------------
def create_dataset_3d(tokenizer, seqs, labels, names=None):
    """Tokenize interleaved 3Di+AA "SA-pair" sequences for the SAProt backbone.

    SAProt expects one token per (AA, 3Di) pair. We first try the fast batched
    tokenizer; if its token count doesn't match N residues we fall back to encoding
    each pair individually (robust to tokenizer edge cases).
    """
    all_input_ids, all_attention_masks = [], []
    for seq in seqs:
        tokenized = tokenizer(seq, max_length=1024, padding=False, truncation=True)
        if len(tokenized['input_ids']) - 2 != len(seq) // 2:
            alt_input_ids = [0]
            for aa in range(0, len(seq) - 1, 2):
                pair_output = tokenizer(seq[aa] + seq[aa + 1])
                alt_input_ids.append(pair_output['input_ids'][1])
            alt_input_ids = alt_input_ids + [2]
            tokenized = {
                'input_ids': alt_input_ids,
                'attention_mask': [1] * len(alt_input_ids),
            }
        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])

    dataset = Dataset.from_dict({'input_ids': all_input_ids, 'attention_mask': all_attention_masks})
    # Truncate labels to the model's max residue count (1022 + <cls>/<eos>).
    labels = [l[:1022] for l in labels]
    dataset = dataset.add_column("labels", labels)
    if names:
        dataset = dataset.add_column("name", names)
    return dataset


# ---------------------------------------------------------------------------
# ProstT5 (Flexpert-Seq) dataset
# ---------------------------------------------------------------------------
def split_sa_pair(sa_seq):
    """Split an interleaved AA+3Di string of length 2N into (aa_seq, di_seq).

    SA-pair convention: even indices = uppercase AA, odd indices = lowercase 3Di.
    """
    n = len(sa_seq) // 2
    aa = ''.join(sa_seq[2 * i] for i in range(n))
    di = ''.join(sa_seq[2 * i + 1] for i in range(n))
    return aa, di


def create_dataset(tokenizer, seqs, labels, names=None):
    """Tokenize space-separated AA sequences for the ProstT5 backbone (Flexpert-Seq).

    No directional prefix is added — this matches how the released ProstT5 weights were
    fine-tuned. Layout per sample: [r_1, ..., r_N, </s>]. Labels are cut to 1023 so the
    collator pads them to match input_ids (1023 residues + trailing </s>).
    """
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    labels = [l[:1023] for l in labels]
    dataset = dataset.add_column("labels", labels)
    if names:
        dataset = dataset.add_column("name", names)
    return dataset


# ---------------------------------------------------------------------------
# Splitting / saving / metrics
# ---------------------------------------------------------------------------
def do_topology_split(df, split_path):
    with open(split_path, 'r') as f:
        splits = json.load(f)
    train_df = df[df['name'].isin(splits['train'])]
    valid_df = df[df['name'].isin(splits['validation'])]
    test_df = df[df['name'].isin(splits['test'])]
    return train_df, valid_df, test_df


def save_finetuned_model(model, target_folder):
    """Save all parameters that were changed during fine-tuning."""
    filepath = os.path.join(target_folder, "final_model")
    model.save_pretrained(filepath, safe_serialization=False)
    print(f"Final model saved to {filepath}")


def update_config(config, args):
    """Overwrite YAML config values with any non-None command-line arguments."""
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    return config


def compute_metrics(eval_pred):
    """Spearman/Pearson correlation + MSE over valid (non-padding, in-range) residues."""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()

    valid_labels = labels[np.where((labels != -100) & (labels < 900))]
    valid_predictions = predictions[np.where((labels != -100) & (labels < 900))]

    if valid_labels.size == 0:
        return {"spearmanr": 0.0, "pearsonr": 0.0, "mse": 0.0}

    spearman_rho, _ = spearmanr(valid_labels, valid_predictions)
    pearson_r, _ = pearsonr(valid_labels, valid_predictions)
    mse_value = mean_squared_error(valid_labels, valid_predictions)
    return {"spearmanr": spearman_rho, "pearsonr": pearson_r, "mse": mse_value}
