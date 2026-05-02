"""Teacher-specific concrete models."""

from .abmil import ABMIL
from .linear_probe import LinearProbe
from .mlp import MLP
from .registered_patch_fusion import RegisteredPatchFusionMIL

__all__ = ['ABMIL', 'LinearProbe', 'MLP', 'RegisteredPatchFusionMIL']
