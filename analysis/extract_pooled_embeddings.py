#!/usr/bin/env python3
"""Mean-pooled protein embeddings (one vector per chain) for Figure 5a/b.

Each chain's last-encoder-layer per-residue representations are averaged over its residues
(padding and ``</s>`` excluded), giving one 1024-d vector per FASTA record. Every model gets the
same input, the space-separated amino-acid sequence alone (token layout ``[r1..rN, </s>]``). Four
models:

  flexpert        : Flexpert-Seq on ProstT5 (LoRA, ``--weights_path``); the vector is the mean of
                    the regression-head input, i.e. the fine-tuned encoder's final hidden state.
  flexpert_prott5 : Flexpert-Seq on ProtT5, same readout.
  prott5          : vanilla ``Rostlab/prot_t5_xl_uniref50`` (no fine-tuning).
  prostt5         : vanilla ``Rostlab/ProstT5`` (no fine-tuning), no ``<AA2fold>`` prefix (as Flexpert-Seq).

No length filter: every FASTA record gets a row, in FASTA order, so the three files line up.
Batches are built from length-sorted sequences to keep padding low; padded positions are
masked in attention and excluded from the mean.

    python3 -m analysis.extract_pooled_embeddings --model flexpert \
        --weights_path weights/flexpert_seq_prostt5_aa_weights.bin --fasta_path data/pdb_sequences.fasta
    # -> analysis/figures/pooled_flexpert.npz  (names, features)
"""
import argparse
import os

from utils.env import setup_env
setup_env()

import numpy as np
import torch
from tqdm import tqdm

HUB_IDS = {'prott5': 'Rostlab/prot_t5_xl_uniref50', 'prostt5': 'Rostlab/ProstT5'}
# Fine-tuned models: (backbone, default weights).
FLEXPERT = {'flexpert': ('prostt5', 'weights/flexpert_seq_prostt5_aa_weights.bin'),
            'flexpert_prott5': ('prott5', 'weights/flexpert_seq_prott5_weights.bin')}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=[*FLEXPERT, *HUB_IDS])
    p.add_argument('--fasta_path', default='data/pdb_sequences.fasta')
    p.add_argument('--weights_path', default=None,
                   help='Flexpert-Seq weights for --model flexpert / flexpert_prott5 '
                        '(default: the released ATLAS weights of that backbone).')
    p.add_argument('--output_path', default=None, help='Default: analysis/figures/pooled_<model>.npz')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--limit', type=int, default=None, help='Only the first N records (smoke test).')
    return p.parse_args()


def load_model(name, weights_path, device):
    """Return ``(tokenizer, embed_fn)``; ``embed_fn(input_ids, attention_mask) -> (B, T, 1024)``."""
    from transformers import T5EncoderModel, T5Tokenizer

    if name in FLEXPERT:
        from run_inference import load_weights, _FACTORIES, _INFERENCE_CONFIG
        from utils.utils import ClassConfig

        backbone, default_weights = FLEXPERT[name]
        model, tokenizer = _FACTORIES[backbone](False, ClassConfig(dict(_INFERENCE_CONFIG)))
        load_weights(model, weights_path or default_weights, device)
        model = model.to(device).eval()
        captured = {}
        # The regression head's input is the fine-tuned encoder's final hidden state.
        model.classifier.register_forward_pre_hook(lambda m, a: captured.__setitem__('h', a[0]))

        def embed(ids, mask):
            model(input_ids=ids, attention_mask=mask)
            return captured['h']
        return tokenizer, embed

    tokenizer = T5Tokenizer.from_pretrained(HUB_IDS[name], do_lower_case=False)
    encoder = T5EncoderModel.from_pretrained(HUB_IDS[name]).to(device).eval()
    return tokenizer, lambda ids, mask: encoder(input_ids=ids, attention_mask=mask).last_hidden_state


def main():
    args = parse_args()
    from run_inference import read_fasta, clean_aa

    out = args.output_path or f'analysis/figures/pooled_{args.model}.npz'
    seqs = read_fasta(args.fasta_path)
    names = list(seqs)[:args.limit] if args.limit else list(seqs)
    seqs = [clean_aa(seqs[n]) for n in names]
    print(f'{args.model}: pooling {len(names):,} chains from {args.fasta_path}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, embed = load_model(args.model, args.weights_path, device)

    order = np.argsort([len(s) for s in seqs])[::-1]   # longest first: OOM shows up early
    feats = np.zeros((len(names), 1024), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(order), args.batch_size)):
            idx = order[start:start + args.batch_size]
            enc = tokenizer([' '.join(seqs[i]) for i in idx], padding=True, return_tensors='pt').to(device)
            h = embed(enc['input_ids'], enc['attention_mask']).float()
            for row, i in enumerate(idx):
                feats[i] = h[row, :len(seqs[i])].mean(0).cpu().numpy()   # [r1..rN, </s>]

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, names=np.array(names), features=feats)
    print(f'Saved {feats.shape} -> {out}')


if __name__ == '__main__':
    main()
