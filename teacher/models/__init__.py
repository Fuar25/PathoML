"""Teacher-specific concrete models."""

from .abmil import ABMIL
from .linear_probe import LinearProbe
from .mlp import MLP

__all__ = ['ABMIL', 'LinearProbe', 'MLP']
