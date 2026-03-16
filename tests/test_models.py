"""Tests for ABMIL and LinearProbe forward passes using synthetic tensors (CPU-only)."""

import pytest
import torch

from PathoML.models.abmil import ABMIL
from PathoML.models.linear_probe import LinearProbe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(B=1, N=10, C=64):
  """Synthetic DataDict with features shape (B, N, C)."""
  return {'features': torch.randn(B, N, C)}


# ---------------------------------------------------------------------------
# ABMIL
# ---------------------------------------------------------------------------

@pytest.fixture
def abmil():
  return ABMIL(input_dim=64, hidden_dim=32, num_classes=1).eval()


@pytest.fixture
def abmil_multiclass():
  return ABMIL(input_dim=64, hidden_dim=32, num_classes=3).eval()


def test_abmil_forward_output_keys(abmil):
  out = abmil(_data())
  assert {'logits', 'bag_embeddings', 'attention'} <= set(out.keys())


def test_abmil_output_shapes(abmil):
  B, N = 1, 10
  out = abmil(_data(B=B, N=N))
  assert out['logits'].shape == (B, 1)
  assert out['attention'].shape == (B, N)
  assert out['bag_embeddings'].shape == (B, 32)


def test_abmil_gated_vs_ungated_same_output_shapes():
  # (1) gated=True/False walk different code paths in GatedAttention;
  #     both must produce identical output shapes
  data = _data()
  out_gated   = ABMIL(input_dim=64, hidden_dim=32, num_classes=1, gated=True).eval()(data)
  out_ungated = ABMIL(input_dim=64, hidden_dim=32, num_classes=1, gated=False).eval()(data)
  for key in ('logits', 'bag_embeddings', 'attention'):
    assert out_gated[key].shape == out_ungated[key].shape, \
      f"Shape mismatch for '{key}': {out_gated[key].shape} vs {out_ungated[key].shape}"


def test_abmil_multi_class(abmil_multiclass):
  out = abmil_multiclass(_data())
  assert out['logits'].shape == (1, 3)


def test_abmil_batch_size_gt_1():
  # (1) DataLoader typically has B=1 for WSI, but model must support B>1
  model = ABMIL(input_dim=64, hidden_dim=32, num_classes=1).eval()
  out = model(_data(B=3, N=10))
  assert out['logits'].shape == (3, 1)
  assert out['attention'].shape == (3, 10)


def test_abmil_no_gradient_in_eval():
  model = ABMIL(input_dim=64, hidden_dim=32, num_classes=1).eval()
  with torch.no_grad():
    out = model(_data())
  assert out['logits'].requires_grad == False


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------

def test_linear_probe_2d_input():
  model = LinearProbe(input_dim=64, num_classes=1).eval()
  out = model({'features': torch.randn(1, 64)})
  assert out['logits'].shape == (1, 1)


def test_linear_probe_3d_input():
  # (1) (B, 1, C) → squeezed to (B, C) internally
  model = LinearProbe(input_dim=64, num_classes=1).eval()
  out = model({'features': torch.randn(1, 1, 64)})
  assert out['logits'].shape == (1, 1)


def test_linear_probe_output_shape_multiclass():
  model = LinearProbe(input_dim=64, num_classes=3).eval()
  out = model({'features': torch.randn(2, 64)})
  assert out['logits'].shape == (2, 3)


def test_linear_probe_output_key():
  model = LinearProbe(input_dim=64).eval()
  out = model({'features': torch.randn(1, 64)})
  assert 'logits' in out
