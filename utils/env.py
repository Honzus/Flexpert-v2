"""Process-environment setup shared by the Flexpert entry points.

Call ``setup_env()`` at the very top of a script — *before* torch / transformers / peft /
wandb are imported — so that ``HF_HOME`` actually takes effect for the HuggingFace cache:

    from utils.env import setup_env
    ENV = setup_env()

Rules:
  * ``HF_HOME`` is taken from ``configs/env_config.yaml`` only if it is not already set in the
    shell and the YAML value is not the ``/path/to/...`` placeholder.
  * ``CUDA_VISIBLE_DEVICES`` is never overwritten (SLURM sets it per job); the YAML value is
    used only when the variable is unset.
  * ``WANDB_DIR`` defaults to ``outputs/logs`` (callers may set it beforehand to log elsewhere).
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CONFIG = REPO_ROOT / 'configs' / 'env_config.yaml'


def setup_env(path=DEFAULT_ENV_CONFIG):
    """Apply configs/env_config.yaml to os.environ (without clobbering) and return it as a dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    hf = (cfg.get('huggingface') or {}).get('HF_HOME')
    if hf and not str(hf).startswith('/path/to') and 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = str(hf)

    dev = (cfg.get('gpus') or {}).get('cuda_visible_device')
    if dev is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(dev)

    os.environ.setdefault('WANDB_DIR', str(REPO_ROOT / 'outputs' / 'logs'))
    os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
    return cfg
