"""Token-level regression backbones for Flexpert.

Three backbones are exposed:

* **ProstT5** (Flexpert-Seq) — AA-only input, space-separated residues, no directional prefix.
* **ProtT5** (Flexpert-Seq, original backbone; ``Rostlab/prot_t5_xl_uniref50``) — same input
  format and the same model class as ProstT5 (both are T5-XL encoders).
* **SAProt** (Flexpert-3D) — structure-aware 3Di+AA tokens.

All are LoRA fine-tuned (rank 4) with a ``MultiplicativeScaling`` wrapper around the
injected adapters; the base PLM is frozen and only LoRA + scaling + LayerNorm are trained.
A single linear regression head (``ENMNoAdaptorClassifier``) sits on top of the per-residue
embeddings — the release is ENM-free.

``ProstT5EncoderForTokenClassification`` also exposes optional hooks for downstream fine-tuning
recipes that read out an intermediate encoder layer (``class_config.readout_layer``, the
``capture_layer`` forward argument) or train only the head (``class_config.trainable_scope``).
They are inactive by default, so plain Flexpert training/inference is unaffected.
"""
import re
import copy

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    T5Config, T5PreTrainedModel, T5EncoderModel, T5Tokenizer,
    EsmConfig, EsmModel, EsmPreTrainedModel, EsmTokenizer,
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import TokenClassifierOutput
try:
    from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
except ModuleNotFoundError:  # removed in transformers >= 5; only used by legacy parallelize()
    assert_device_map = get_device_map = None
from peft import LoraConfig, inject_adapter_in_model

from models.enm_adaptor_heads import ENMNoAdaptorClassifier
from utils.lora_utils import MultiplicativeScaling


def _device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# ---------------------------------------------------------------------------
# ProstT5 / ProtT5 (Flexpert-Seq, AA-only)
# ---------------------------------------------------------------------------
class ProstT5EncoderForTokenClassification(T5PreTrainedModel):
    """T5 encoder + linear per-residue head; used for both ProstT5 and ProtT5."""

    # The encoder reuses `shared` as its input embedding (as in T5EncoderModel). Declaring the tie
    # lets transformers >= 5 save Trainer checkpoints (safetensors) without a shared-tensor error.
    _tied_weights_keys = {"encoder.embed_tokens.weight": "shared.weight"}

    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # transformers >= 5 dropped the T5Stack(config, embed_tokens) form; construct with
        # config alone and wire in the shared embedding via the standard setter.
        self.encoder = T5Stack(encoder_config)
        self.encoder.set_input_embeddings(self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = ENMNoAdaptorClassifier(config.hidden_size, class_config.num_labels)

        rl = getattr(class_config, 'readout_layer', None)
        self.readout_layer = None if rl in (None, -1) else int(rl)

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        enm_vals=None,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        capture_layer='default',
    ):
        return_dict = return_dict if return_dict is not None else True
        layer = self.readout_layer if capture_layer == 'default' else capture_layer
        layer = None if layer in (None, -1) else int(layer)
        want_hidden = bool(output_hidden_states) or (layer == 0)

        if layer is not None and layer >= 1:
            _cap = {}
            _h = self.encoder.block[layer - 1].register_forward_hook(
                lambda _m, _i, _o: _cap.__setitem__('h', _o[0]))
            try:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, head_mask=head_mask,
                    output_attentions=output_attentions, output_hidden_states=False,
                    return_dict=True)
            finally:
                _h.remove()
            sequence_output = _cap['h']
            if capture_layer != 'default' or output_hidden_states:
                outputs.hidden_states = (sequence_output,)
            sequence_output = self.encoder.final_layer_norm(sequence_output)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=want_hidden,
                return_dict=return_dict,
            )
            sequence_output = outputs.hidden_states[0] if layer == 0 else outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output, enm_vals, attention_mask)

        if not return_dict:
            return (logits,) + outputs[2:]
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _t5_classification_model(hub_id, label, half_precision, class_config, lora_r, lora_alpha):
    """Build a T5-encoder token-regression model with LoRA + multiplicative scaling."""
    device = _device()
    tokenizer = T5Tokenizer.from_pretrained(hub_id, do_lower_case=False)
    if half_precision:
        model = T5EncoderModel.from_pretrained(hub_id, torch_dtype=torch.float16).to(device)
    else:
        model = T5EncoderModel.from_pretrained(hub_id).to(device)

    class_model = ProstT5EncoderForTokenClassification(model.config, class_config)
    class_model.shared = model.shared
    class_model.encoder = model.encoder
    model = class_model
    del class_model

    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=["q", "k", "v", "o"], bias="none",
    )
    model = inject_adapter_in_model(peft_config, model)

    for module in model.modules():
        if hasattr(module, 'lora_A'):
            nn.init.normal_(module.lora_A.default.weight, std=0.01)

    # Wrap LoRA-injected attention projections with multiplicative scaling.
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(".*SelfAttention", m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch("q|k|v|o", c_name):
                    setattr(module, c_name, MultiplicativeScaling(layer, init_scale=0.01))

    # Freeze base; train LoRA + scaling + layer_norm only.
    for param in model.shared.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if re.fullmatch(".*lora.*|.*scale_in.*|.*scale_out.*|.*layer_norm.*", name):
            param.requires_grad = True

    trainable_scope = getattr(class_config, 'trainable_scope', None)
    if trainable_scope == 'classifier':
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith('classifier')

    trainable = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print(f"{label}_LoRA_Classifier — trainable parameters: " + str(trainable))
    return model, tokenizer


def ProstT5_classification_model(half_precision, class_config, lora_r=4, lora_alpha=1):
    """Flexpert-Seq on ProstT5 (``Rostlab/ProstT5``)."""
    return _t5_classification_model("Rostlab/ProstT5", "ProstT5", half_precision, class_config,
                                    lora_r, lora_alpha)


def ProtT5_classification_model(half_precision, class_config, lora_r=4, lora_alpha=1):
    """Flexpert-Seq on ProtT5 (``Rostlab/prot_t5_xl_uniref50``), the original Flexpert backbone."""
    return _t5_classification_model("Rostlab/prot_t5_xl_uniref50", "ProtT5", half_precision,
                                    class_config, lora_r, lora_alpha)


# ---------------------------------------------------------------------------
# SAProt (Flexpert-3D, 3Di+AA)
# ---------------------------------------------------------------------------
class SAProtEncoderForTokenClassification(EsmPreTrainedModel):

    def __init__(self, config: EsmConfig, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = ENMNoAdaptorClassifier(config.hidden_size, class_config.num_labels)

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embeddings.word_embeddings = new_embeddings

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        enm_vals=None,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else True
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output, enm_vals, attention_mask)

        if not return_dict:
            return (logits,) + outputs[2:]
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def SAProt_classification_model(half_precision, class_config, lora_r=4, lora_alpha=1):
    """Build a SAProt token-regression model with LoRA + multiplicative scaling."""
    device = _device()
    tokenizer = EsmTokenizer.from_pretrained('westlake-repl/SaProt_650M_PDB')
    if half_precision:
        model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_PDB", torch_dtype=torch.float16).to(device)
    else:
        model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_PDB").to(device)

    class_model = SAProtEncoderForTokenClassification(model.config, class_config)
    class_model.encoder = model  # base EsmModel, no LoRA yet
    model = class_model
    del class_model

    # Inject LoRA into attention q/k/v and the attention output projection only
    # (regex avoids catching the FFN intermediate/output dense layers).
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=r".*attention\.(self\.(query|key|value)|output\.dense)",
        bias="none",
    )
    model = inject_adapter_in_model(peft_config, model)

    for module in model.modules():
        if hasattr(module, 'lora_A'):
            nn.init.normal_(module.lora_A.default.weight, std=0.01)

    # ESM splits q/k/v and dense into separate submodules → two wrapping patterns.
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(r".*attention\.self", m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch("query|key|value", c_name):
                    setattr(module, c_name, MultiplicativeScaling(layer, init_scale=0.01))
        elif re.fullmatch(r".*attention\.output", m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch("dense", c_name):
                    setattr(module, c_name, MultiplicativeScaling(layer, init_scale=0.01))

    # Freeze base; train LoRA + scaling + LayerNorm only (ESM uses "LayerNorm").
    for param in model.encoder.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if re.fullmatch(r".*lora.*|.*scale_in.*|.*scale_out.*|.*LayerNorm.*", name):
            param.requires_grad = True

    trainable = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print("SAProt_LoRA_Classifier — trainable parameters: " + str(trainable))
    return model, tokenizer
