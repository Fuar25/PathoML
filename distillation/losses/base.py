"""Base distillation loss contracts and composition helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


def format_formula_value(value: int | float) -> str:
  """Format a numeric value for human-readable loss formulas."""
  value_f = float(value)
  if value_f.is_integer():
    return str(int(value_f))
  return f"{value_f:g}"


def format_slug_value(value: int | float | str) -> str:
  """Format a value for shell-friendly condition-name slugs."""
  return str(value).replace('-', 'minus_').replace('.', 'p')


class DistillationLoss(nn.Module, ABC):
  """Stable trainer-facing contract for a distillation loss."""

  @abstractmethod
  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    ...

  @abstractmethod
  def describe(self) -> str:
    """Return the canonical human-readable formula."""

  @abstractmethod
  def slug(self) -> str:
    """Return the canonical condition-name fragment."""

  def __str__(self) -> str:
    return self.describe()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.describe()})"


class DistillationTerm(nn.Module, ABC):
  """Atomic distillation term contract."""

  @abstractmethod
  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    ...

  @abstractmethod
  def describe(self) -> str:
    """Return the canonical human-readable formula fragment."""

  @abstractmethod
  def slug(self) -> str:
    """Return the canonical condition-name fragment."""

  def __str__(self) -> str:
    return self.describe()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.describe()})"


class WeightedTerm(DistillationTerm):
  """Attach a scalar coefficient to an atomic term."""

  def __init__(self, term: DistillationTerm, weight: float) -> None:
    super().__init__()
    self.term = term
    self.weight = float(weight)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    return self.weight * self.term(s_out, t_out, labels)

  def describe(self) -> str:
    if self.weight == 1.0:
      return self.term.describe()
    return f"{format_formula_value(self.weight)}*{self.term.describe()}"

  def slug(self) -> str:
    if self.weight == 1.0:
      return self.term.slug()
    return f"{self.term.slug()}_{format_slug_value(self.weight)}"


class CompositeDistillationLoss(DistillationLoss):
  """Ordered sum of explicit atomic distillation terms."""

  def __init__(self, terms: list[DistillationTerm]) -> None:
    super().__init__()
    if not terms:
      raise ValueError("CompositeDistillationLoss requires at least one term.")
    self.terms = nn.ModuleList(terms)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    total = self.terms[0](s_out, t_out, labels)
    for term in self.terms[1:]:
      total = total + term(s_out, t_out, labels)
    return total

  def describe(self) -> str:
    return " + ".join(term.describe() for term in self.terms)

  def slug(self) -> str:
    return "_".join(term.slug() for term in self.terms)
