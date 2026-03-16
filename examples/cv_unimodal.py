"""Single-modality cross-validation example.

Run from the project root: python examples/cv_unimodal.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PathoML.config.config import RunTimeConfig
from main import build_and_run

config = RunTimeConfig()

# =========================================================================
# Edit the fields below, then run: python examples/cv_unimodal.py
# =========================================================================

# Data paths — one directory per class, containing .h5 files
config.dataset.dataset_name = "wsi_h5"
config.dataset.dataset_kwargs["data_paths"] = {
  "positive": "/path/to/positive_h5",   # e.g. MALT H5 files
  "negative": "/path/to/negative_h5",   # e.g. Reactive H5 files
}

# Model
config.model.model_name = "abmil"       # or "linear_probe"
# input_dim is auto-detected from the dataset; override here if needed:
# config.model.input_dim = 1536

# ABMIL-specific parameters (model_kwargs are filtered by model signature)
config.model.model_kwargs = {
  "gated": True,
  "attention_dim": None,    # None → hidden_dim // 2
}

# Training
config.training.device = "cuda:0"
config.training.epochs = 50
config.training.learning_rate = 5e-4
config.training.patience = 5
config.training.seed = 42

# Checkpoints
config.logging.save_dir = "./experiments/cv_unimodal"

# =========================================================================
build_and_run(config, strategy="cv", k_folds=5)
