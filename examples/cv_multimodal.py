"""Multi-modal cross-validation example (concat strategy).

Features from each modality are concatenated along the channel dim → (N, ΣD_i).
The model's input_dim must match ΣD_i (auto-detected from the dataset).

Run from the project root: python examples/cv_multimodal.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PathoML.config.config import RunTimeConfig
from main import build_and_run

config = RunTimeConfig()

# =========================================================================
# Edit the fields below, then run: python examples/cv_multimodal.py
# =========================================================================

# Data paths — one root per modality; each root has class sub-directories
config.dataset.dataset_name = "multimodal_concat"
config.dataset.dataset_kwargs = {
  "modality_paths": {
    "HE":   "/mnt/5T/GML/Tiff/Experiments/Experiment1/GigaPath-Slide-Feature/HE",    # contains MALT/ and Reactive/ sub-dirs
    "CD20": "/mnt/5T/GML/Tiff/Experiments/Experiment1/GigaPath-Slide-Feature/CD20",
  },
  "modality_names": ["HE", "CD20"],
  "allow_missing_modalities": True,    # samples with only one modality are kept
}

# Model — input_dim is auto-detected as sum of per-modality dims
config.model.model_name = "abmil"
config.model.model_kwargs = {"gated": True}

# Training
config.training.device = "cuda:0"
config.training.epochs = 50
config.training.learning_rate = 5e-4
config.training.patience = 5

# Checkpoints
config.logging.save_dir = "./experiments/cv_multimodal_concat"

# =========================================================================
build_and_run(config, strategy="cv", k_folds=5)
