"""Lightweight factory/registry for dataset and model management."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class Registry:
  """Simple string-to-callable registry for factory functions.

  Register classes or functions via decorator, then instantiate by name.
  Enables plugin-style module management without modifying core code.
  """

  def __init__(self, name: str) -> None:
    self._name = name
    self._items: Dict[str, Callable[..., Any]] = {}

  def register(self, key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that registers a class or function under the given key.

    Args:
        key: Registry key (case-insensitive). Defaults to target.__name__.lower().
    """
    def decorator(target: Callable[..., T]) -> Callable[..., T]:
      registry_key = (key or target.__name__).lower()
      if registry_key in self._items:
        raise ValueError(f"{self._name} '{registry_key}' already registered")
      self._items[registry_key] = target
      return target
    return decorator

  def get(self, key: str) -> Callable[..., Any]:
    """Retrieve a registered callable by key."""
    registry_key = key.lower()
    if registry_key not in self._items:
      available = ", ".join(sorted(self._items)) or "<empty>"
      raise KeyError(f"Unknown {self._name} '{key}'. Available: {available}")
    return self._items[registry_key]

  def create(self, key: str, **kwargs: Any) -> Any:
    """Instantiate a registered object, filtering unsupported kwargs automatically."""
    target = self.get(key)
    filtered_kwargs, dropped_keys = self._filter_kwargs_for_target(target, kwargs)
    if dropped_keys:
      print(
        f"Info: Ignoring unsupported kwargs for {self._name} '{key}': "
        f"{', '.join(sorted(dropped_keys))}"
      )
    return target(**filtered_kwargs)

  @staticmethod
  def _filter_kwargs_for_target(
    target: Callable[..., Any],
    kwargs: Dict[str, Any],
  ) -> tuple[Dict[str, Any], set[str]]:
    """Filter kwargs to only those accepted by target's signature.

    If target accepts **kwargs, all args pass through unchanged.
    """
    try:
      signature = inspect.signature(target)
    except (TypeError, ValueError):
      return dict(kwargs), set()

    accepts_var_kwargs = any(
      p.kind == inspect.Parameter.VAR_KEYWORD
      for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
      return dict(kwargs), set()

    accepted_names = {
      p.name
      for p in signature.parameters.values()
      if p.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
      )
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted_names}
    dropped = set(kwargs.keys()) - set(filtered.keys())
    return filtered, dropped

  def available(self) -> list[str]:
    """Return sorted list of all registered keys."""
    return sorted(self._items.keys())


model_registry = Registry("model")
dataset_registry = Registry("dataset")


def register_model(key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
  """Decorator to register a class or function as a model."""
  return model_registry.register(key)


def register_dataset(key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
  """Decorator to register a class or function as a dataset."""
  return dataset_registry.register(key)


def create_model(key: str, **kwargs: Any) -> Any:
  """Instantiate a model from the registry by name."""
  return model_registry.create(key, **kwargs)


def create_dataset(key: str, **kwargs: Any) -> Any:
  """Instantiate a dataset from the registry by name."""
  return dataset_registry.create(key, **kwargs)


# Built-in modules auto-loaded on every run. Add new built-ins here.
_BUILTIN_DATASET_MODULES = [
  'PathoML.data.unimodal_dataset',
  'PathoML.data.multimodal_dataset_concat',
  'PathoML.data.multimodal_dataset_add',
]
_BUILTIN_MODEL_MODULES = [
  'PathoML.models.abmil',
  'PathoML.models.linear_probe',
]


def load_runtime_plugins(config: Any) -> None:
  """Import built-in modules and any user-specified extension modules.

  Built-ins are always loaded. User extensions are appended via
  config.dataset.dataset_module_paths / config.model.model_module_paths.

  Args:
      config: RunTimeConfig with optional dataset_module_paths / model_module_paths
              for user-defined extensions.
  """
  # (1) Always load built-ins
  for module_path in _BUILTIN_DATASET_MODULES + _BUILTIN_MODEL_MODULES:
    importlib.import_module(module_path)

  # (2) Load user extensions (empty by default)
  for module_path in getattr(config.dataset, 'dataset_module_paths', []) or []:
    importlib.import_module(module_path)
  for module_path in getattr(config.model, 'model_module_paths', []) or []:
    importlib.import_module(module_path)
