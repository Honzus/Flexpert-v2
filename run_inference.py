#!/usr/bin/env python3
"""Unified Flexpert inference — predict per-residue flexibility (RMSF).

Three backbones:

* ``--backbone prostt5``  → Flexpert-Seq (ProstT5; AA FASTA, e.g. data/atlas_sequences.fasta)
* ``--backbone prott5``   → Flexpert-Seq on the original ProtT5 backbone (AA FASTA)
* ``--backbone saprot``   → Flexpert-3D (SA-pair FASTA, e.g. data/atlas_sa_sequences.fasta)

With ``--splits_path`` only the chains of one split (default ``test``) are predicted; without it,
every record in the FASTA is. Output is a text file: one ``>NAME`` header line (the FASTA id, i.e.
the header up to the first whitespace) followed by comma-separated per-residue predictions.

The fine-tuned weights may be given as a ``pytorch_model.bin`` / ``*.safetensors`` file, a
sharded checkpoint (``*.index.json``), or a checkpoint directory such as
``outputs/checkpoints/<run>_<ts>/final_model``. If ``--output_path`` is omitted, predictions go to
``outputs/predictions/<model>_<dataset>_<split>.txt`` (with ``--splits_path``; dataset = the splits
file name without ``_splits.json``, lower-cased) or ``outputs/predictions/<model>_<fasta stem>.txt``
(without it). ``<model>`` is flexpert_seq / flexpert_seq_prott5 / flexpert_3d, plus ``_mdcath`` /
``_combined`` for those released weights, e.g. ``flexpert_seq_atlas_test.txt``.

Examples
--------
    python3 download_flexpert_weights.py          # fetch the released weights into weights/

    # Flexpert-Seq (ProstT5, AA-only)  -> outputs/predictions/flexpert_seq_atlas_test.txt
    python3 run_inference.py --backbone prostt5 \
        --weights_path weights/flexpert_seq_prostt5_aa_weights.bin \
        --fasta_path data/atlas_sequences.fasta --splits_path data/atlas_splits.json --split test

    # Flexpert-3D (SAProt, SA-pair)    -> outputs/predictions/flexpert_3d_atlas_test.txt
    python3 run_inference.py --backbone saprot \
        --weights_path weights/flexpert_3d_saprot_weights.bin \
        --fasta_path data/atlas_sa_sequences.fasta --splits_path data/atlas_splits.json --split test

    # Your own proteins (every FASTA record) -> outputs/predictions/flexpert_seq_my_proteins.txt
    python3 run_inference.py --backbone prostt5 \
        --weights_path weights/flexpert_seq_prostt5_aa_weights.bin --fasta_path my_proteins.fasta
"""
import os
import re
import json
import argparse

# Must run before torch/transformers are imported so HF_HOME etc. take effect.
from utils.env import setup_env
setup_env()

import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.T5_encoder_per_token import (
    ProstT5_classification_model, ProtT5_classification_model, SAProt_classification_model,
)
from utils.utils import (
    ClassConfig, DataCollatorForTokenRegression, DataCollatorForTokenRegression3D,
    create_dataset, create_dataset_3d, split_sa_pair,
)

_INFERENCE_CONFIG = {
    'num_labels': 1,
    'dropout_rate': 0.2,
}
_FACTORIES = {'prostt5': ProstT5_classification_model, 'prott5': ProtT5_classification_model,
              'saprot': SAProt_classification_model}
_MODEL_NAMES = {'prostt5': 'flexpert_seq', 'prott5': 'flexpert_seq_prott5', 'saprot': 'flexpert_3d'}
# Residues per chain supervised in training (1024 tokens). ProstT5/ProtT5 inputs are truncated to
# this; SaProt (rotary positions) predicts the whole chain, as the released reference numbers did.
MAX_RESIDUES = {'prostt5': 1023, 'prott5': 1023, 'saprot': 1022}
_STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
_DROP = re.compile(r'[\s*\-.]')
_SA_PAIR = re.compile(r'(?:[A-Z#][a-z#])+')
_SA_HINT = ('Flexpert-3D needs an SA-pair FASTA (uppercase amino acid + lowercase 3Di letter per '
            'residue, e.g. "MdKvLp"). Build one from structures with: '
            'python3 -m data.scripts.build_sa_fasta --pdb_dir <dir> --out <fasta>')


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Flexpert inference')
    parser.add_argument('--backbone', required=True, choices=['prostt5', 'prott5', 'saprot'],
                        help='prostt5 / prott5 → Flexpert-Seq (AA fasta); '
                             'saprot → Flexpert-3D (SA-pair fasta).')
    parser.add_argument('--weights_path', required=True,
                        help='Fine-tuned weights: a .bin/.safetensors file, a sharded checkpoint '
                             '(*.index.json), or a checkpoint directory (e.g. .../final_model).')
    parser.add_argument('--fasta_path', required=True)
    parser.add_argument('--splits_path', default=None,
                        help='Optional splits JSON; predict only --split. Default: every FASTA record.')
    parser.add_argument('--split', default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--output_path', default=None,
                        help='Output .txt (default: outputs/predictions/<model>_<dataset>_<split>.txt, '
                             'or <model>_<fasta stem>.txt without --splits_path).')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=4,
                        help='LoRA rank the weights were trained with (released weights: 4).')
    parser.add_argument('--lora_alpha', type=int, default=1,
                        help='LoRA alpha the weights were trained with (released weights: 1).')
    return parser.parse_args()


def default_output_path(backbone, splits_path, split, weights_path=None, fasta_path=None):
    """outputs/predictions/<model>_<dataset>_<split>.txt, e.g. flexpert_seq_atlas_test.txt.

    Without a splits file: outputs/predictions/<model>_<fasta stem>.txt. The released mdCATH /
    combined weights add their training set to <model> (flexpert_seq_mdcath, flexpert_3d_combined,
    flexpert_seq_prott5_mdcath, ...) so runs of different weights do not collide.
    """
    model = _MODEL_NAMES[backbone]
    weights_name = os.path.basename(weights_path or '')
    for train_set in ('mdcath', 'combined'):
        if weights_name.endswith(f'_{train_set}_weights.bin'):
            model = f'{model}_{train_set}'
    if splits_path is None:
        stem = os.path.basename(fasta_path)
        for ext in ('.gz', '.fasta', '.fas', '.faa', '.fa'):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
        return os.path.join('outputs', 'predictions', f'{model}_{stem}.txt')
    dataset = os.path.basename(splits_path)
    for suffix in ('.json', '_splits'):
        if dataset.endswith(suffix):
            dataset = dataset[:-len(suffix)]
    return os.path.join('outputs', 'predictions', f'{model}_{dataset.lower()}_{split}.txt')


def _load_file(path, device):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device)


def _resolve_weights(weights_path):
    """Map a checkpoint directory to the weights file inside it; leave file paths unchanged."""
    if not os.path.isdir(weights_path):
        return weights_path
    for fname in ('pytorch_model.bin', 'model.safetensors',
                  'pytorch_model.bin.index.json', 'model.safetensors.index.json'):
        candidate = os.path.join(weights_path, fname)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'No pytorch_model.bin / model.safetensors (or sharded index) in {weights_path}')


def load_weights(model, weights_path, device):
    """Load fine-tuned weights into ``model`` (non-strict; prints missing/unexpected counts).

    Accepts a ``.bin`` / ``.safetensors`` file, a sharded checkpoint (the ``*.index.json`` itself,
    or a ``.bin``/``.safetensors`` path whose ``<path>.index.json`` exists), or a directory
    containing one of those (e.g. ``outputs/checkpoints/<run>_<ts>/final_model``).
    """
    weights_path = _resolve_weights(weights_path)
    if weights_path.endswith('.index.json'):
        index_path = weights_path
    elif os.path.exists(weights_path + '.index.json'):
        index_path = weights_path + '.index.json'
    else:
        index_path = None

    if index_path is not None:
        model_dir = os.path.dirname(index_path)
        print(f'Loading sharded checkpoint from {model_dir or "."}')
        with open(index_path) as f:
            index = json.load(f)
        state_dict = {}
        for shard_file in sorted(set(index['weight_map'].values())):
            state_dict.update(_load_file(os.path.join(model_dir, shard_file), device))
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights not found: {weights_path} '
                                    '(run `python3 download_flexpert_weights.py` for the released weights)')
        state_dict = _load_file(weights_path, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded weights — {len(missing)} missing, {len(unexpected)} unexpected keys')
    return missing, unexpected


def read_fasta(path):
    """Ordered ``{id: sequence}``; id = header up to the first whitespace. Duplicate ids raise."""
    seqs, dups = {}, []
    for record in SeqIO.parse(path, 'fasta'):
        if record.id in seqs:
            dups.append(record.id)
        seqs[record.id] = str(record.seq)
    if dups:
        raise ValueError(f'{path}: duplicate FASTA ids {sorted(set(dups))[:10]} '
                         '(ids are the header up to the first whitespace and must be unique)')
    if not seqs:
        raise ValueError(f'{path}: no FASTA records found')
    return seqs


def is_sa_pair(seq):
    return bool(_SA_PAIR.fullmatch(seq))


def clean_aa(seq):
    """Upper-case; drop whitespace, '*', '-', '.'; map non-standard letters (O/U/B/Z/J...) to X."""
    seq = _DROP.sub('', seq).upper()
    bad = sorted({c for c in seq if not c.isalpha()})
    if bad:
        raise ValueError(f'invalid characters {bad}')
    return ''.join(c if c in _STANDARD_AA else 'X' for c in seq)


def clean_sa_pair(seq):
    """Validate an SA-pair string; clean uncommon AAs at AA positions (even indices) to X."""
    chars = list(re.sub(r'\s', '', seq))
    for i in range(0, len(chars), 2):
        if chars[i] in 'OUBZ-':
            chars[i] = 'X'
    seq = ''.join(chars)
    if not is_sa_pair(seq):
        raise ValueError('not an SA-pair sequence')
    return seq


def prepare_sequences(backbone, seqs):
    """Clean/validate ``{id: raw sequence}`` for ``backbone``; returns ``{id: model-ready string}``.

    Seq backbones get the AA string (the AA stream is taken from an SA-pair input); SaProt gets
    the validated SA-pair string. Bad records raise one ValueError naming them.
    """
    out, errors = {}, []
    for name, s in seqs.items():
        try:
            if backbone == 'saprot':
                out[name] = clean_sa_pair(s)
            else:
                s = re.sub(r'\s', '', s)
                out[name] = clean_aa(split_sa_pair(s)[0] if is_sa_pair(s) else s)
            if not out[name]:
                raise ValueError('empty sequence')
        except ValueError as e:
            errors.append(f'{name}: {e}')
    if errors:
        more = f'\n  ... and {len(errors) - 10} more' if len(errors) > 10 else ''
        hint = '\n' + _SA_HINT if backbone == 'saprot' else ''
        raise ValueError(f'{len(errors)} invalid sequence(s):\n  ' + '\n  '.join(errors[:10])
                         + more + hint)
    return out


def warn_long(backbone, seqs):
    """One warning listing chains longer than the training length (``MAX_RESIDUES``)."""
    limit = MAX_RESIDUES[backbone]
    step = 2 if backbone == 'saprot' else 1
    long = [f'{n} ({len(s) // step})' for n, s in seqs.items() if len(s) // step > limit]
    if not long:
        return long
    what = (f'Flexpert-3D still predicts every residue, but training never supervised residues '
            f'beyond {limit}' if backbone == 'saprot'
            else f'predictions cover only their first {limit} residues')
    print(f'Warning: {len(long)} chain(s) longer than {limit} residues; {what}: '
          + ', '.join(long[:20]) + (' ...' if len(long) > 20 else ''))
    return long


def main():
    args = parse_args()
    if args.output_path is None:
        args.output_path = default_output_path(args.backbone, args.splits_path, args.split,
                                               args.weights_path, args.fasta_path)

    # Read and check the input before loading the (large) model.
    all_seqs = read_fasta(args.fasta_path)
    if args.splits_path is not None:
        with open(args.splits_path) as f:
            wanted = set(json.load(f)[args.split])
        all_seqs = {n: s for n, s in all_seqs.items() if n in wanted}
    seqs = prepare_sequences(args.backbone, all_seqs)
    names = list(seqs.keys())
    warn_long(args.backbone, seqs)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_config = ClassConfig(dict(_INFERENCE_CONFIG))
    model, tokenizer = _FACTORIES[args.backbone](
        False, class_config, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    load_weights(model, args.weights_path, device)
    model = model.to(device)
    model.eval()
    print(f'Running inference on {len(names)} proteins (backbone={args.backbone})')

    dummy = [[0.0] for _ in names]
    if args.backbone == 'saprot':
        dataset = create_dataset_3d(tokenizer, [seqs[n] for n in names], dummy)
        collator = DataCollatorForTokenRegression3D(tokenizer)
    else:
        # AA-only, space-separated, no directional prefix.
        dataset = create_dataset(tokenizer, [' '.join(seqs[n]) for n in names], dummy)
        collator = DataCollatorForTokenRegression(tokenizer)

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
                if args.backbone == 'saprot':
                    # SAProt: [<cls>, aa0, ..., aaN-1, <eos>] -> residues at 1 .. n_real-2.
                    pred = logits[i, 1:n_real - 1, 0].cpu().numpy()
                else:
                    # ProstT5 / ProtT5 (no prefix): [r1, ..., rN, </s>] -> residues 0 .. n_real-2.
                    pred = logits[i, :n_real - 1, 0].cpu().numpy()
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
