"""Full-dataset training example (no validation split).

Trains on all available data for a fixed number of epochs.
Use this to produce a deployment model after cross-validation has confirmed
the expected number of epochs.

Run from the project root: python examples/full_train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PathoML.config.config import RunTimeConfig
from main import build_and_run

config = RunTimeConfig()

# =========================================================================
# Edit the fields below, then run: python examples/full_train.py
# =========================================================================

# Data paths
config.dataset.dataset_name = "wsi_h5"
config.dataset.dataset_kwargs["data_paths"] = {
  "positive": "/path/to/positive_h5",
  "negative": "/path/to/negative_h5",
}

# Model
config.model.model_name = "abmil"
config.model.model_kwargs = {"gated": True}

# Training — set epochs to the best epoch found by cross-validation
config.training.device = "cuda:0"
config.training.epochs = 30             # set from CV result
config.training.learning_rate = 5e-4
config.training.seed = 42

# Checkpoints
config.logging.save_dir = "./experiments/full_train"

# =========================================================================
build_and_run(config, strategy="train")
