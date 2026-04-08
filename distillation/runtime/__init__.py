"""Runtime helpers for the distillation subsystem."""

from .manifest import TeacherManifest, load_manifest
from .trainer import DistillCrossValidator

__all__ = ['TeacherManifest', 'load_manifest', 'DistillCrossValidator']
