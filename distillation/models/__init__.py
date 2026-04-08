"""Model area for the distillation subsystem.

`student/` owns student architectures.
`teacher/` owns teacher checkpoint adapters only.
"""

from .student import StudentBasicABMIL, StudentTransABMIL, StudentTransABMIL_MH
from .teacher import TeacherMLP

__all__ = [
  'StudentBasicABMIL',
  'StudentTransABMIL',
  'StudentTransABMIL_MH',
  'TeacherMLP',
]
