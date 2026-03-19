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
config.dataset.dataset_name = "UnimodalPatchDataset"
config.dataset.dataset_kwargs["data_path"] = "/path/to/data_root"
# /path/to/data_root must contain one subdirectory per class, e.g.:
#   /path/to/data_root/MALT/*.h5
#   /path/to/data_root/Reactive/*.h5

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
