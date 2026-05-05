"""Teacher-specific concrete models."""

from .abmil import ABMIL
from .linear_probe import LinearProbe
from .mlp import MLP
from .registered_patch_fusion import (
  RegisteredPatchCoordFusionMIL,
  RegisteredPatchFusionMIL,
  RegisteredPatchPolyCoordFusionMIL,
  RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL,
  RegisteredPatchPolyCoordStainAffineGateFusionMIL,
)

__all__ = [
  'ABMIL',
  'LinearProbe',
  'MLP',
  'RegisteredPatchFusionMIL',
  'RegisteredPatchCoordFusionMIL',
  'RegisteredPatchPolyCoordFusionMIL',
  'RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL',
  'RegisteredPatchPolyCoordStainAffineGateFusionMIL',
]
