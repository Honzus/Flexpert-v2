"""CPU tests for the Flexpert core (Flexpert-Seq / Flexpert-3D training + inference).

No GPU, no network, no model downloads: every model here is a tiny random-weight T5 / ESM and
the PLM factories are exercised with ``from_pretrained`` monkeypatched to return those.

Run from the repo root:

    HF_HUB_OFFLINE=1 python3 -m pytest tests/test_core.py -q
"""
import argparse
import importlib
import inspect
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from transformers import EsmConfig, EsmModel, T5Config, T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]

# Strings that must not appear in the core / tracked files. Built from pieces so that this test
# file does not itself trip the repo-wide release greps.
FORBIDDEN_IN_CORE = [
    're' + 'sep', 'steer' + 'ing', 'Reactivity' + 'Trainer', 'ddg' + '_pairs',
    'models/' + 'weights', 'prediction' + '_results', 'results' + '/',
]
ABS_PATH_MARKERS = ['/' + 'mnt/', '/' + 'home/']
SAVE_SAFETENSORS = 'save_' + 'safetensors'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_t5_config(vocab_size=40):
    return T5Config(d_model=32, d_ff=64, num_layers=2, num_heads=2, d_kv=16, vocab_size=vocab_size)


def _tiny_esm_config(vocab_size=40):
    # SaProt is a rotary ESM-2 with pad_token_id=1.
    return EsmConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, vocab_size=vocab_size, max_position_embeddings=64,
                     pad_token_id=1, position_embedding_type='rotary')


def _class_config(**kw):
    from utils.utils import ClassConfig
    cfg = {'num_labels': 1, 'dropout_rate': 0.0}
    cfg.update(kw)
    return ClassConfig(cfg)


class _StubTokenizer:
    """Whitespace tokenizer: 'A C D' -> [id(A), id(C), id(D), </s>=1]; pads with 0."""

    def __init__(self, padding_side='right'):
        self.padding_side = padding_side

    def __call__(self, seqs, max_length=None, padding=False, truncation=False):
        ids = [[4 + (ord(c) % 30) for c in s.split()] + [1] for s in seqs]
        if truncation and max_length is not None:
            ids = [x[:max_length - 1] + [1] if len(x) > max_length else x for x in ids]
        return {'input_ids': ids, 'attention_mask': [[1] * len(x) for x in ids]}

    def pad(self, features, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt'):
        width = max(len(f['input_ids']) for f in features)
        out = {'input_ids': [], 'attention_mask': []}
        for f in features:
            n = width - len(f['input_ids'])
            ids, am = list(f['input_ids']), list(f['attention_mask'])
            if self.padding_side == 'right':
                ids, am = ids + [0] * n, am + [0] * n
            else:
                ids, am = [0] * n + ids, [0] * n + am
            out['input_ids'].append(ids)
            out['attention_mask'].append(am)
        return {k: torch.tensor(v) for k, v in out.items()}


def _patch_prostt5_factory(monkeypatch):
    import models.T5_encoder_per_token as mt

    class _TinyT5:
        @staticmethod
        def from_pretrained(*a, **kw):
            return T5EncoderModel(_tiny_t5_config(vocab_size=160))

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _StubTokenizer()

    monkeypatch.setattr(mt, 'T5EncoderModel', _TinyT5)
    monkeypatch.setattr(mt, 'T5Tokenizer', _Tok)
    monkeypatch.setattr(mt, '_device', lambda: torch.device('cpu'))
    return mt


def _write_fasta(path, records):
    path.write_text(''.join(f'>{h}\n{seq}\n' for h, seq in records))
    return str(path)


def _patch_saprot_factory(monkeypatch):
    import models.T5_encoder_per_token as mt

    class _TinyEsm:
        @staticmethod
        def from_pretrained(*a, **kw):
            return EsmModel(_tiny_esm_config(vocab_size=446))

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _StubTokenizer()

    monkeypatch.setattr(mt, 'EsmModel', _TinyEsm)
    monkeypatch.setattr(mt, 'EsmTokenizer', _Tok)
    monkeypatch.setattr(mt, '_device', lambda: torch.device('cpu'))
    return mt


def _read_label_file_like_train(path):
    """The parsing used by train.py / train_3d.py."""
    names, labels = [], []
    with open(path) as f:
        for line in f:
            _split_line = line.split(':\t')
            names.append(_split_line[0])
            labels.append([float(v) for v in _split_line[1].split(', ')])
    return dict(zip(names, labels))


def _core_files():
    files = [REPO_ROOT / f for f in ('train.py', 'train_3d.py', 'run_inference.py',
                                     'download_flexpert_weights.py')]
    for pattern in ('utils/*.py', 'models/*.py', 'configs/*', 'slurm/*.sh', 'evaluation/*.py'):
        files += sorted(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return files


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('module', [
    'utils.utils', 'utils.env', 'utils.lora_utils',
    'models.T5_encoder_per_token', 'models.enm_adaptor_heads',
    'evaluation.evaluate_ablations', 'evaluation.get_correlation_analysis',
    'evaluation.get_correlation_analysis_mdCATH',
    'analysis.extract_cath_labels', 'analysis.extract_embeddings',
    'analysis.plot_flexibility_universe', 'analysis.plot_embedding_universe',
    'analysis.plot_embedding_universe_cath', 'analysis.extract_pooled_embeddings',
    'analysis.fig5_tsne', 'analysis.fig5_neighbours', 'analysis.fig5_termini',
    'analysis.fig5_hydrophobicity',
    'data.scripts.extract_rmsf_labels', 'data.scripts.build_sa_fasta',
    'data.scripts.prepare_dataset',
    'run_inference', 'download_flexpert_weights',
])
def test_module_imports(module):
    importlib.import_module(module)


# ---------------------------------------------------------------------------
# 2. Core purity: the core never depends on private extensions or internal paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('path', _core_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_core_file_is_pure(path):
    text = path.read_text(errors='replace')
    hits = [s for s in FORBIDDEN_IN_CORE if s in text]
    assert not hits, f'{path.relative_to(REPO_ROOT)} contains {hits}'


# ---------------------------------------------------------------------------
# 3. Label format
# ---------------------------------------------------------------------------
class TestLabelFormat:
    def test_bundled_label_file_format(self):
        with open(REPO_ROOT / 'data' / 'rmsf_atlas_data_prottransready.txt') as f:
            first = [next(f) for _ in range(5)]
        for line in first:
            assert re.match(r'^[\w.]+:\t', line), line[:40]

    def test_format_label_line_round_trip(self, tmp_path):
        from data.scripts.extract_rmsf_labels import format_label_line
        vals = {'1k5n.A': [0.39, 0.3253333333333333, 999.0], '2abc.B': [1.0, 2.5]}
        out = tmp_path / 'labels.txt'
        out.write_text(''.join(format_label_line(k, v) for k, v in vals.items()))
        assert _read_label_file_like_train(out) == vals

    def test_extract_rmsf_labels_source_uses_colon_tab(self):
        src = (REPO_ROOT / 'data' / 'scripts' / 'extract_rmsf_labels.py').read_text()
        assert ':\\t' in src

    def test_extract_rmsf_labels_atlas_file(self, tmp_path):
        """ATLAS id keeps the chain ('1k5n_A' -> '1k5n.A') and RMSF is the replica mean."""
        from data.scripts.extract_rmsf_labels import extract_rmsf_labels, format_label_line
        d = tmp_path / '1k5n_A'
        d.mkdir()
        tsv = d / '1k5n_A_RMSF.tsv'
        tsv.write_text('position\tRMSF_R1\tRMSF_R2\tRMSF_R3\n1\t0.3\t0.6\t0.9\n2\t1.0\t1.0\t1.0\n')
        name, vals = extract_rmsf_labels(str(tsv))
        assert name == '1k5n.A'
        assert np.allclose(vals, [0.6, 1.0])
        line = format_label_line(name, vals)
        assert re.match(r'^[\w.]+:\t', line)


# ---------------------------------------------------------------------------
# 4. Datasets, collators, metrics, small helpers
# ---------------------------------------------------------------------------
class TestCreateDataset:
    def test_truncates_to_1023_labels(self):
        from utils.utils import create_dataset
        seq = ' '.join('A' * 1500)
        ds = create_dataset(_StubTokenizer(), [seq], [[0.1] * 1500], max_length=1024)
        assert len(ds[0]['input_ids']) == 1024
        assert len(ds[0]['labels']) == 1023

    def test_no_truncation_when_max_length_none(self):
        from utils.utils import create_dataset
        seq = ' '.join('A' * 1500)
        ds = create_dataset(_StubTokenizer(), [seq], [[0.1] * 1500], names=['x'], max_length=None)
        assert len(ds[0]['input_ids']) == 1501
        assert len(ds[0]['labels']) == 1500
        assert ds[0]['name'] == 'x'


class TestCollators:
    FEATS = [{'input_ids': [5, 6, 7, 1], 'attention_mask': [1, 1, 1, 1], 'labels': [0.1, 0.2, 0.3]},
             {'input_ids': [5, 1], 'attention_mask': [1, 1], 'labels': [0.4]}]

    def test_seq_collator_right(self):
        from utils.utils import DataCollatorForTokenRegression
        b = DataCollatorForTokenRegression(_StubTokenizer('right'))(self.FEATS)
        assert b['input_ids'].shape == (2, 4)
        assert b['labels'].dtype == torch.float
        assert b['labels'][0].tolist() == pytest.approx([0.1, 0.2, 0.3, -100])
        assert b['labels'][1].tolist() == pytest.approx([0.4, -100, -100, -100])

    def test_seq_collator_left(self):
        from utils.utils import DataCollatorForTokenRegression
        b = DataCollatorForTokenRegression(_StubTokenizer('left'))(self.FEATS)
        assert b['labels'][1].tolist() == pytest.approx([-100, -100, -100, 0.4])

    def test_3d_collator_right_masks_cls(self):
        from utils.utils import DataCollatorForTokenRegression3D
        feats = [{'input_ids': [0, 5, 6, 2], 'attention_mask': [1, 1, 1, 1], 'labels': [0.1, 0.2]},
                 {'input_ids': [0, 5, 2], 'attention_mask': [1, 1, 1], 'labels': [0.4]}]
        b = DataCollatorForTokenRegression3D(_StubTokenizer('right'))(feats)
        assert b['labels'][0].tolist() == pytest.approx([-100, 0.1, 0.2, -100])
        assert b['labels'][1].tolist() == pytest.approx([-100, 0.4, -100, -100])

    def test_3d_collator_left(self):
        from utils.utils import DataCollatorForTokenRegression3D
        feats = [{'input_ids': [0, 5, 6, 2], 'attention_mask': [1, 1, 1, 1], 'labels': [0.1, 0.2]},
                 {'input_ids': [0, 5, 2], 'attention_mask': [1, 1, 1], 'labels': [0.4]}]
        b = DataCollatorForTokenRegression3D(_StubTokenizer('left'))(feats)
        assert b['labels'][1].tolist() == pytest.approx([-100, -100, -100, 0.4])

    def test_no_labels_passthrough(self):
        from utils.utils import DataCollatorForTokenRegression
        feats = [{k: v for k, v in f.items() if k != 'labels'} for f in self.FEATS]
        b = DataCollatorForTokenRegression(_StubTokenizer())(feats)
        assert 'labels' not in b


class TestComputeMetrics:
    def test_perfect_correlation(self):
        from utils.utils import compute_metrics
        y = np.array([[0.1, 0.5, 0.3, 0.9]])
        m = compute_metrics((y.copy(), y))
        assert m['spearmanr'] == pytest.approx(1.0)
        assert m['pearsonr'] == pytest.approx(1.0)
        assert m['mse'] == pytest.approx(0.0)

    def test_filters_padding_and_sentinels(self):
        from utils.utils import compute_metrics
        labels = np.array([[0.1, 0.2, 0.3, -100, 950.0]])
        preds = np.array([[1.0, 2.0, 3.0, -5.0, 0.0]])  # perfectly ranked on the 3 valid residues
        m = compute_metrics((preds, labels))
        assert m['spearmanr'] == pytest.approx(1.0)
        assert m['pearsonr'] == pytest.approx(1.0)

    def test_all_invalid_returns_zeros(self):
        from utils.utils import compute_metrics
        m = compute_metrics((np.array([[1.0, 2.0]]), np.array([[-100, 999.0]])))
        assert m == {'spearmanr': 0.0, 'pearsonr': 0.0, 'mse': 0.0}


class TestHelpers:
    def test_do_topology_split(self, tmp_path):
        from utils.utils import do_topology_split
        sp = tmp_path / 'splits.json'
        sp.write_text(json.dumps({'train': ['a', 'b'], 'validation': ['c'], 'test': ['d']}))
        df = pd.DataFrame({'name': ['a', 'b', 'c', 'd', 'e'], 'x': range(5)})
        tr, va, te = do_topology_split(df, str(sp))
        assert list(tr['name']) == ['a', 'b']
        assert list(va['name']) == ['c']
        assert list(te['name']) == ['d']

    def test_update_config_ignores_none(self):
        from utils.utils import update_config
        cfg = {'batch_size': 8, 'epochs': 100, 'seed': 42}
        args = argparse.Namespace(batch_size=2, epochs=None, run_name='r')
        out = update_config(cfg, args)
        assert out == {'batch_size': 2, 'epochs': 100, 'seed': 42, 'run_name': 'r'}

    def test_class_config(self):
        from utils.utils import ClassConfig
        cc = ClassConfig({'num_labels': 1, 'dropout_rate': 0.2})
        assert cc.num_labels == 1 and cc.dropout_rate == 0.2

    def test_set_seeds_reproducible(self):
        from utils.utils import set_seeds
        draws = []
        for _ in range(2):
            set_seeds(123)
            draws.append((torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random()))
        assert draws[0] == draws[1]


class TestTrainerLoss:
    class _Dummy(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            return {'logits': self.logits}

    def _loss(self, **kw):
        from utils.utils import ENMAdaptedTrainer
        logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        inputs = {'input_ids': torch.ones(1, 4, dtype=torch.long),
                  'attention_mask': torch.tensor([[1, 1, 1, 0]]),
                  'labels': torch.tensor([[1.5, -100.0, 3.0, 9.0]])}
        trainer = ENMAdaptedTrainer.__new__(ENMAdaptedTrainer)  # compute_loss uses no state
        return ENMAdaptedTrainer.compute_loss(trainer, self._Dummy(logits), inputs, **kw)

    def test_masked_mse(self):
        # Valid residues: pos0 (1.0 vs 1.5) and pos2 (3.0 vs 3.0); pos1 is -100, pos3 is padding.
        assert self._loss().item() == pytest.approx(0.125)

    def test_accepts_num_items_in_batch(self):
        loss, outputs = self._loss(return_outputs=True, num_items_in_batch=2)
        assert loss.item() == pytest.approx(0.125)
        assert 'logits' in outputs


# ---------------------------------------------------------------------------
# 5. Tiny random models
# ---------------------------------------------------------------------------
class TestTinyModels:
    def test_prostt5_forward(self):
        from models.T5_encoder_per_token import ProstT5EncoderForTokenClassification
        m = ProstT5EncoderForTokenClassification(_tiny_t5_config(), _class_config()).eval()
        ids = torch.randint(2, 40, (1, 7))
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
        assert out.logits.shape == (1, 7, 1)

    def test_prostt5_readout_layer(self):
        from models.T5_encoder_per_token import ProstT5EncoderForTokenClassification
        torch.manual_seed(0)
        m = ProstT5EncoderForTokenClassification(_tiny_t5_config(), _class_config(readout_layer=2)).eval()
        assert m.readout_layer == 2
        ids = torch.randint(2, 40, (1, 7))
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=am)
            # Reading out the last block must match the default (final-layer) path.
            default = m(input_ids=ids, attention_mask=am, capture_layer=None)
            first = m(input_ids=ids, attention_mask=am, capture_layer=1)
        assert out.logits.shape == (1, 7, 1)
        assert torch.allclose(out.logits, default.logits, atol=1e-5)
        assert not torch.allclose(first.logits, default.logits)

    def test_saprot_forward(self):
        from models.T5_encoder_per_token import SAProtEncoderForTokenClassification
        cfg = _tiny_esm_config()
        m = SAProtEncoderForTokenClassification(cfg, _class_config())
        m.encoder = EsmModel(cfg)  # as in SAProt_classification_model
        m.eval()
        ids = torch.randint(4, 40, (1, 9))
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
        assert out.logits.shape == (1, 9, 1)

    def test_prostt5_factory_trainable_params(self, monkeypatch):
        mt = _patch_prostt5_factory(monkeypatch)
        model, tok = mt.ProstT5_classification_model(False, _class_config(), lora_r=4, lora_alpha=1)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any('lora' in n for n in trainable)
        assert any('scale_in' in n for n in trainable)
        assert any('layer_norm' in n for n in trainable)
        assert not any(n.startswith('shared.') for n in trainable)
        assert all(re.search('lora|scale_|layer_norm|classifier', n) for n in trainable)

    def test_prott5_factory_uses_prott5_hub_id(self, monkeypatch):
        mt = _patch_prostt5_factory(monkeypatch)
        seen = []

        class _Rec:
            @staticmethod
            def from_pretrained(hub_id, **kw):
                seen.append(hub_id)
                return T5EncoderModel(_tiny_t5_config(vocab_size=160))

        monkeypatch.setattr(mt, 'T5EncoderModel', _Rec)
        model, _ = mt.ProtT5_classification_model(False, _class_config(), lora_r=4, lora_alpha=1)
        assert seen == ['Rostlab/prot_t5_xl_uniref50']
        assert isinstance(model, mt.ProstT5EncoderForTokenClassification)
        assert any(n.endswith('SelfAttention.q.scale_in') for n, _ in model.named_parameters())

    def test_saprot_factory_forward(self, monkeypatch):
        mt = _patch_saprot_factory(monkeypatch)
        model, _ = mt.SAProt_classification_model(False, _class_config(), lora_r=4, lora_alpha=1)
        model.eval()
        ids = torch.randint(4, 40, (2, 9))
        assert model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.shape == (2, 9, 1)


# ---------------------------------------------------------------------------
# 6. Checkpoint round-trip: save_finetuned_model -> run_inference.load_weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('backbone', ['prostt5', 'prott5', 'saprot'])
def test_save_load_round_trip(backbone, tmp_path, monkeypatch):
    from utils.utils import save_finetuned_model
    from run_inference import load_weights
    if backbone == 'prostt5':
        factory = _patch_prostt5_factory(monkeypatch).ProstT5_classification_model
    elif backbone == 'prott5':
        factory = _patch_prostt5_factory(monkeypatch).ProtT5_classification_model
    else:
        factory = _patch_saprot_factory(monkeypatch).SAProt_classification_model

    torch.manual_seed(0)
    src, _ = factory(False, _class_config())
    with torch.no_grad():  # make the trainable parameters distinct from a fresh init
        for p in src.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    save_finetuned_model(src, str(tmp_path))
    ckpt = tmp_path / 'final_model' / 'pytorch_model.bin'
    assert ckpt.is_file()
    assert (tmp_path / 'final_model' / 'config.json').is_file()

    ids = torch.randint(4, 40, (1, 8))
    am = torch.ones_like(ids)
    src.eval()
    with torch.no_grad():
        ref = src(input_ids=ids, attention_mask=am).logits

    for path in (ckpt, tmp_path / 'final_model'):  # file and checkpoint directory
        torch.manual_seed(1)
        dst, _ = factory(False, _class_config())
        missing, unexpected = load_weights(dst, str(path), torch.device('cpu'))
        assert list(unexpected) == []
        assert list(missing) == []
        dst.eval()
        with torch.no_grad():
            assert torch.allclose(dst(input_ids=ids, attention_mask=am).logits, ref, atol=1e-6)


def test_load_weights_missing_file_message(tmp_path):
    from run_inference import load_weights
    with pytest.raises(FileNotFoundError, match='download_flexpert_weights'):
        load_weights(torch.nn.Linear(1, 1), str(tmp_path / 'nope.bin'), torch.device('cpu'))


# ---------------------------------------------------------------------------
# 7. transformers v5 contract + configs
# ---------------------------------------------------------------------------
class TestTransformersContract:
    def _yaml(self):
        with open(REPO_ROOT / 'configs' / 'train_config.yaml') as f:
            return yaml.safe_load(f)

    def test_training_arguments_from_yaml(self, tmp_path):
        from transformers import TrainingArguments
        ta = dict(self._yaml()['training_args'])
        ta['run_name'] = 'pytest'
        ta['output_dir'] = str(tmp_path / 'ckpt')
        args = TrainingArguments(**ta)
        assert args.metric_for_best_model == 'spearmanr'
        assert args.load_best_model_at_end

    def test_yaml_has_no_safetensors_flag(self):
        assert SAVE_SAFETENSORS not in (REPO_ROOT / 'configs' / 'train_config.yaml').read_text()
        assert SAVE_SAFETENSORS not in self._yaml()['training_args']

    def test_output_dir_template(self):
        assert self._yaml()['training_args']['output_dir'].startswith('./outputs/checkpoints/')

    def test_trainer_takes_processing_class(self):
        from transformers import Trainer
        params = inspect.signature(Trainer.__init__).parameters
        assert 'processing_class' in params
        assert 'tokenizer' not in params

    @pytest.mark.parametrize('script', ['train.py', 'train_3d.py'])
    def test_train_scripts_use_processing_class(self, script):
        src = (REPO_ROOT / script).read_text()
        assert 'processing_class=' in src
        assert 'tokenizer=' + 'tokenizer' not in src


# ---------------------------------------------------------------------------
# 8. run_inference / download_flexpert_weights CLIs
# ---------------------------------------------------------------------------
class TestRunInferenceCLI:
    def test_parse_args_defaults(self, monkeypatch):
        import run_inference
        monkeypatch.setattr(sys, 'argv', [
            'run_inference.py', '--backbone', 'prostt5', '--weights_path', 'w.bin',
            '--fasta_path', 'data/atlas_sequences.fasta', '--splits_path', 'data/atlas_splits.json'])
        a = run_inference.parse_args()
        assert (a.split, a.output_path, a.batch_size, a.lora_r, a.lora_alpha) == ('test', None, 4, 4, 1)

    def test_parse_args_rejects_unknown_backbone(self, monkeypatch):
        import run_inference
        monkeypatch.setattr(sys, 'argv', [
            'run_inference.py', '--backbone', 'prot5', '--weights_path', 'w',
            '--fasta_path', 'f', '--splits_path', 's'])
        with pytest.raises(SystemExit):
            run_inference.parse_args()

    @pytest.mark.parametrize('backbone,splits,split,expected', [
        ('prostt5', 'data/atlas_splits.json', 'test', 'flexpert_seq_atlas_test.txt'),
        ('saprot', 'data/atlas_splits.json', 'test', 'flexpert_3d_atlas_test.txt'),
        ('saprot', 'data/mdCATH_splits.json', 'validation', 'flexpert_3d_mdcath_validation.txt'),
    ])
    def test_default_output_path(self, backbone, splits, split, expected):
        from run_inference import default_output_path
        assert default_output_path(backbone, splits, split) == str(Path('outputs') / 'predictions' / expected)

    @pytest.mark.parametrize('backbone,weights,expected', [
        ('prostt5', 'weights/flexpert_seq_prostt5_aa_weights.bin', 'flexpert_seq_mdcath_test.txt'),
        ('prostt5', 'weights/flexpert_seq_prostt5_aa_mdcath_weights.bin', 'flexpert_seq_mdcath_mdcath_test.txt'),
        ('prostt5', 'weights/flexpert_seq_prostt5_aa_combined_weights.bin', 'flexpert_seq_combined_mdcath_test.txt'),
        ('prostt5', 'outputs/checkpoints/run_1/final_model', 'flexpert_seq_mdcath_test.txt'),
        ('prott5', 'weights/flexpert_seq_prott5_weights.bin', 'flexpert_seq_prott5_mdcath_test.txt'),
        ('prott5', 'weights/flexpert_seq_prott5_mdcath_weights.bin', 'flexpert_seq_prott5_mdcath_mdcath_test.txt'),
        ('prott5', 'weights/flexpert_seq_prott5_combined_weights.bin', 'flexpert_seq_prott5_combined_mdcath_test.txt'),
        ('saprot', 'weights/flexpert_3d_saprot_combined_weights.bin', 'flexpert_3d_combined_mdcath_test.txt'),
    ])
    def test_default_output_path_weight_variant(self, backbone, weights, expected):
        # names match what evaluation.check_reproduction and slurm/verify_core.sh expect
        from run_inference import default_output_path
        got = default_output_path(backbone, 'data/mdCATH_splits.json', 'test', weights)
        assert got == str(Path('outputs') / 'predictions' / expected)

    @pytest.mark.parametrize('backbone,weights,fasta,expected', [
        ('prostt5', 'weights/flexpert_seq_prostt5_aa_weights.bin', 'data/custom_dataset/fasta_dataset.fasta',
         'flexpert_seq_fasta_dataset.txt'),
        ('prott5', 'weights/flexpert_seq_prott5_mdcath_weights.bin', 'my/proteins.fa', 'flexpert_seq_prott5_mdcath_proteins.txt'),
        ('saprot', 'weights/flexpert_3d_saprot_weights.bin', 'outputs/custom_sa.fasta', 'flexpert_3d_custom_sa.txt'),
    ])
    def test_default_output_path_without_splits(self, backbone, weights, fasta, expected):
        from run_inference import default_output_path
        got = default_output_path(backbone, None, 'test', weights, fasta)
        assert got == str(Path('outputs') / 'predictions' / expected)

    def test_splits_path_is_optional(self, monkeypatch):
        import run_inference
        monkeypatch.setattr(sys, 'argv', [
            'run_inference.py', '--backbone', 'prott5', '--weights_path', 'w.bin', '--fasta_path', 'f.fasta'])
        a = run_inference.parse_args()
        assert a.splits_path is None and a.backbone == 'prott5'


class TestFastaInput:
    def test_ids_multiline_and_order(self, tmp_path):
        from run_inference import read_fasta
        path = _write_fasta(tmp_path / 'in.fasta', [
            ('sp|P12345|X_HUMAN Some protein OS=Homo sapiens', 'MKV\nLAA'), ('b desc', 'GG')])
        seqs = read_fasta(path)
        assert list(seqs) == ['sp|P12345|X_HUMAN', 'b']
        assert seqs['sp|P12345|X_HUMAN'] == 'MKVLAA'

    def test_duplicate_ids_raise(self, tmp_path):
        from run_inference import read_fasta
        path = _write_fasta(tmp_path / 'in.fasta', [('a first', 'MK'), ('a second', 'GG')])
        with pytest.raises(ValueError, match='duplicate'):
            read_fasta(path)

    def test_seq_cleaning(self):
        from run_inference import prepare_sequences
        out = prepare_sequences('prostt5', {'a': 'mkv-luoBZJ*', 'b': 'M K V'})
        assert out == {'a': 'MKVLXXXXX', 'b': 'MKV'}

    def test_seq_backbone_takes_aa_from_sa_pairs(self):
        from run_inference import prepare_sequences
        assert prepare_sequences('prott5', {'a': 'MdKvLp'}) == {'a': 'MKL'}

    def test_seq_invalid_characters(self):
        from run_inference import prepare_sequences
        with pytest.raises(ValueError, match="invalid characters"):
            prepare_sequences('prostt5', {'a': 'MK1V'})

    def test_sa_pair_validation(self):
        from run_inference import prepare_sequences
        assert prepare_sequences('saprot', {'a': 'MdKvL#Up'}) == {'a': 'MdKvL#Xp'}
        for bad in ('MKVL', 'MdK', 'mDkV'):
            with pytest.raises(ValueError, match='build_sa_fasta'):
                prepare_sequences('saprot', {'x': bad})

    def test_long_chain_warning(self, capsys):
        from run_inference import warn_long
        assert warn_long('prostt5', {'short': 'A' * 10, 'long': 'A' * 1024}) == ['long (1024)']
        assert 'first 1023 residues' in capsys.readouterr().out
        assert warn_long('saprot', {'x': 'Ad' * 1022}) == []
        assert warn_long('saprot', {'x': 'Ad' * 1023}) == ['x (1023)']

    def test_bundled_fastas_parse(self):
        from run_inference import read_fasta, prepare_sequences
        aa = prepare_sequences('prostt5', read_fasta(REPO_ROOT / 'data' / 'custom_dataset' / 'fasta_dataset.fasta'))
        assert len(aa) == 10 and '1a62.A' in aa
        sa = read_fasta(REPO_ROOT / 'data' / 'atlas_sa_sequences.fasta')
        assert len(prepare_sequences('saprot', sa)) == len(sa)


@pytest.mark.parametrize('backbone', ['prostt5', 'prott5'])
def test_run_inference_custom_fasta_end_to_end(backbone, tmp_path, monkeypatch):
    """No splits file: every record is predicted, names = FASTA ids, one value per residue."""
    import run_inference
    from utils.utils import save_finetuned_model
    mt = _patch_prostt5_factory(monkeypatch)
    factory = mt.ProtT5_classification_model if backbone == 'prott5' else mt.ProstT5_classification_model
    monkeypatch.setitem(run_inference._FACTORIES, backbone, factory)
    monkeypatch.setattr(run_inference.torch.cuda, 'is_available', lambda: False)
    src, _ = factory(False, _class_config())
    save_finetuned_model(src, str(tmp_path / 'ckpt'))
    fasta = _write_fasta(tmp_path / 'my_proteins.fasta', [
        ('sp|P1|A_HUMAN desc', 'MKVL\nAG*'), ('two', 'mkv'), ('long', 'A' * 1030)])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', [
        'run_inference.py', '--backbone', backbone, '--weights_path', str(tmp_path / 'ckpt' / 'final_model'),
        '--fasta_path', fasta, '--batch_size', '2'])
    run_inference.main()
    model = 'flexpert_seq_prott5' if backbone == 'prott5' else 'flexpert_seq'
    lines = (tmp_path / 'outputs' / 'predictions' / f'{model}_my_proteins.txt').read_text().splitlines()
    assert lines[::2] == ['>sp|P1|A_HUMAN', '>two', '>long']
    assert [len(l.split(', ')) for l in lines[1::2]] == [6, 3, 1023]


def test_check_reproduction_noref(tmp_path, monkeypatch):
    from evaluation import check_reproduction as cr
    from evaluation.evaluate_ablations import load_ground_truth
    gt = load_ground_truth(REPO_ROOT / 'data' / 'rmsf_atlas_data_prottransready.txt')
    with open(REPO_ROOT / 'data' / 'atlas_splits.json') as f:
        test = json.load(f)['test']
    pred = tmp_path / 'flexpert_seq_prott5_atlas_test.txt'
    pred.write_text(''.join(f'>{n}\n' + ', '.join(str(v) for v in gt[n.replace('.', '_')]) + '\n'
                            for n in test))
    assert sum(1 for m in cr.REFERENCE for _ in cr.DATASETS) == 27
    out = tmp_path / 'summary.json'
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, 'argv', ['check_reproduction', '--pred_dir', str(tmp_path),
                                      '--json_out', str(out), '--require_all'])
    assert cr.main() == 1
    rows = {(r['model'], r['dataset']): r for r in json.loads(out.read_text())}
    assert rows[('flexpert_seq_prott5', 'atlas')]['status'] == ('NOREF' if cr.REFERENCE[
        'flexpert_seq_prott5']['atlas'] is None else 'FAIL')
    assert rows[('flexpert_seq', 'atlas')]['status'] == 'MISSING'


class TestBuildSaFasta:
    def test_structure_stem(self):
        from data.scripts.build_sa_fasta import structure_stem
        assert structure_stem('x/1abc_A.pdb') == '1abc_A'
        assert structure_stem('x/1abc.A.cif.gz') == '1abc.A'
        assert structure_stem('AF-P1-F1-model_v4.mmcif') == 'AF-P1-F1-model_v4'

    @pytest.mark.parametrize('fmt', ['pdb', 'cif'])
    def test_extract_plddt_per_chain(self, tmp_path, fmt):
        from Bio.PDB import MMCIFIO, PDBIO, StructureBuilder
        from data.scripts.build_sa_fasta import extract_plddt
        sb = StructureBuilder.StructureBuilder()
        sb.init_structure('s')
        sb.init_model(0)
        for chain, n, b in (('A', 3, 90.0), ('B', 2, 40.0)):
            sb.init_chain(chain)
            sb.init_seg('    ')
            for i in range(1, n + 1):
                sb.init_residue('ALA', ' ', i, ' ')
                for atom in ('N', 'CA', 'C'):
                    sb.init_atom(atom, np.array([float(i), 0.0, 0.0]), b + i, 1.0, ' ', atom, None, 'C' if atom != 'N' else 'N')
        io = PDBIO() if fmt == 'pdb' else MMCIFIO()
        io.set_structure(sb.get_structure())
        path = tmp_path / f'af.{fmt}'
        io.save(str(path))
        plddt = extract_plddt(str(path))
        assert list(plddt) == ['A', 'B']
        assert plddt['A'].tolist() == pytest.approx([91.0, 92.0, 93.0])
        assert plddt['B'].tolist() == pytest.approx([41.0, 42.0])


def test_download_weights_layout():
    import download_flexpert_weights as dl
    assert dl.DEST == 'weights'
    assert dl.REPO_ID == 'Honzus24/Flexpert-v2'
    assert set(dl.WEIGHTS) == {'v2', 'prott5'}
    # default (v2, atlas) is unchanged
    assert dl.weight_files() == ['flexpert_seq_prostt5_aa_weights.bin', 'flexpert_3d_saprot_weights.bin']
    assert len(dl.weight_files('v2', 'all')) == 6
    assert dl.weight_files('prott5', 'all') == [
        'flexpert_seq_prott5_weights.bin', 'flexpert_seq_prott5_mdcath_weights.bin',
        'flexpert_seq_prott5_combined_weights.bin']
    everything = dl.weight_files('all', 'all')
    assert len(everything) == 9 and len(set(everything)) == 9
    assert dl.weight_files('all', 'mdcath') == [
        'flexpert_seq_prostt5_aa_mdcath_weights.bin', 'flexpert_3d_saprot_mdcath_weights.bin',
        'flexpert_seq_prott5_mdcath_weights.bin']


def test_download_weights_cli_default(monkeypatch):
    import download_flexpert_weights as dl
    monkeypatch.setattr(sys, 'argv', ['download_flexpert_weights.py'])
    a = dl.parse_args()
    assert (a.models, a.dataset) == ('v2', 'atlas')


# ---------------------------------------------------------------------------
# 9. Figure 5 analysis helpers
# ---------------------------------------------------------------------------
def test_knn_cosine_and_shared_counts():
    from analysis.fig5_neighbours import knn_cosine, shared_counts
    x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    nn = knn_cosine(x, 1, chunk=3)                   # chunk < N exercises the self-exclusion offset
    assert nn[:, 0].tolist() == [1, 0, 3, 2]
    assert shared_counts(np.array([[1, 2, 3]]), np.array([[3, 4, 1]])).tolist() == [2]


def test_knn_overlap_unique_mask_and_summary():
    from analysis.fig5_neighbours import SETS, summarise, unique_sequence_mask
    from analysis.fig5_tsne import MODELS
    assert unique_sequence_mask(['A', 'B', 'A', 'C']).tolist() == [False, True, False, True]
    s = summarise(np.array([0, 10, 20, 30, 40]))
    assert (s['n_chains'], s['mean'], s['median'], s['q25'], s['q75']) == (5, 20, 20, 10, 30)
    assert s['sd'] == pytest.approx(np.sqrt(200))
    assert all(m in MODELS for pairs in SETS.values() for p in pairs for m in p)


def test_termini_zscore_and_bins():
    from analysis.fig5_termini import relative_bins, zscore
    assert relative_bins(5, 2).tolist() == [0, 0, 1, 1, 1]  # i/(L-1) = 0, .25, .5, .75, 1
    z = zscore([1.0, np.nan, 3.0])
    assert z[0] == pytest.approx(-1) and np.isnan(z[1]) and z[2] == pytest.approx(1)


def test_hydrophobicity_his_mask_and_per_aa():
    from analysis.fig5_hydrophobicity import his_tag_mask, per_aa_stats
    assert his_tag_mask('MHHHHAHH').tolist() == [False, True, True, True, True, False, False, False]
    assert not his_tag_mask('MHHHH', 0).any()
    stats = per_aa_stats(['AAG', 'G'], [np.array([1.0, 3.0, np.nan]), np.array([5.0])])
    assert stats['A'][0] == pytest.approx(2) and stats['A'][2] == 2
    assert stats['G'][0] == pytest.approx(5) and stats['G'][2] == 1   # NaN (masked) residue dropped


# ---------------------------------------------------------------------------
# 10. Repo hygiene
# ---------------------------------------------------------------------------
def _tracked_files():
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=REPO_ROOT, check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('not a git checkout')
    return [REPO_ROOT / p for p in out.decode().split('\0') if p]


def test_no_absolute_paths_in_tracked_text_files():
    offenders = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b'\0' in data[:8192]:  # binary
            continue
        text = data.decode('utf-8', errors='replace')
        if any(m in text for m in ABS_PATH_MARKERS):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f'absolute paths in: {offenders}'


@pytest.mark.parametrize('keep', [
    'weights/.gitkeep',
    'outputs/checkpoints/.gitkeep', 'outputs/predictions/.gitkeep', 'outputs/logs/.gitkeep',
])
def test_output_dirs_exist(keep):
    assert (REPO_ROOT / keep).is_file(), f'{keep} missing'
