"""Per-token regression head for Flexpert.

The public release uses a single sequence/structure-only head: a linear layer on top
of the per-residue PLM embeddings. (Earlier ENM-adaptor heads — attention/conv/direct —
are not part of this release; both Flexpert-Seq and Flexpert-3D are ENM-free.)
"""
import torch.nn as nn


class ENMNoAdaptorClassifier(nn.Module):
    """Linear regression head over per-residue embeddings (ignores any ENM input)."""

    def __init__(self, seq_embedding_dim, out_dim):
        super(ENMNoAdaptorClassifier, self).__init__()
        self.adapted_classifier = nn.Linear(seq_embedding_dim, out_dim)

    def forward(self, seq_embedding, enm_input=None, attention_mask=None):
        _ = enm_input  # kept for call-signature compatibility; unused
        logits = self.adapted_classifier(seq_embedding)
        return logits
