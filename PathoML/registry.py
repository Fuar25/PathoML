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


def create_dataset_from_config(cfg: Any) -> Any:
  """Instantiate a dataset directly from a DatasetConfig.

  Merges top-level config fields into dataset_kwargs before construction:
  patient_id_pattern defaults to cfg.patient_id_pattern unless already in dataset_kwargs.

  Args:
      cfg: DatasetConfig instance.
  Returns:
      Instantiated dataset.
  """
  kwargs = dict(cfg.dataset_kwargs)
  kwargs.setdefault('patient_id_pattern', cfg.patient_id_pattern)
  return create_dataset(cfg.dataset_name, **kwargs)


def model_builder_from_config(cfg: Any, dataset: Any) -> Callable[[], Any]:
  """Return a model factory closure with input_dim and num_classes inferred from dataset.

  Inference runs once at call time; the returned callable can be invoked
  repeatedly (e.g. per fold in CrossValidator) to get freshly initialised models.
  Values in cfg.model_kwargs override inferred ones if needed.

  Args:
      cfg: ModelConfig instance.
      dataset: An instantiated dataset with .classes and __getitem__ returning
               a dict with 'features' key.
  Returns:
      A zero-argument callable that creates and returns a new model instance.
  """
  input_dim = int(dataset[0]['features'].shape[-1])
  n_classes = len(dataset.classes)
  num_classes = 1 if n_classes == 2 else n_classes
  kwargs = {'input_dim': input_dim, 'num_classes': num_classes, **cfg.model_kwargs}
  return lambda: create_model(cfg.model_name, **kwargs)


# Built-in modules auto-loaded on every run. Add new built-ins here.
_BUILTIN_DATASET_MODULES = [
  'PathoML.dataset.SlideDataset',
  'PathoML.dataset.PatchDataset',
]
_BUILTIN_MODEL_MODULES = [
  'PathoML.models.abmil',
  'PathoML.models.linear_probe',
]


def load_all_module(config: Any) -> None:
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
