"""Tests for optimization.registry: register, create, kwargs filtering, error handling."""

import pytest

from PathoML.optimization.registry import Registry


def test_register_and_create(fresh_registry):
  @fresh_registry.register("myclass")
  class MyClass:
    def __init__(self, x):
      self.x = x

  obj = fresh_registry.create("myclass", x=42)
  assert isinstance(obj, MyClass)
  assert obj.x == 42


def test_register_default_key(fresh_registry):
  # (1) No explicit key → uses cls.__name__.lower()
  @fresh_registry.register()
  class FooBar:
    pass

  assert "foobar" in fresh_registry.available()


def test_case_insensitive(fresh_registry):
  @fresh_registry.register("MyKey")
  class SomeClass:
    pass

  # Registry normalises to lowercase on register
  assert fresh_registry.get("mykey") is SomeClass
  assert fresh_registry.get("MYKEY") is SomeClass


def test_unknown_key_raises(fresh_registry):
  with pytest.raises(KeyError, match="nonexistent"):
    fresh_registry.get("nonexistent")


def test_unknown_key_error_lists_available(fresh_registry):
  @fresh_registry.register("existing")
  class Existing:
    pass

  with pytest.raises(KeyError, match="existing"):
    fresh_registry.get("missing")


def test_duplicate_registration_raises(fresh_registry):
  @fresh_registry.register("dup")
  class A:
    pass

  with pytest.raises(ValueError):
    @fresh_registry.register("dup")
    class B:
      pass


def test_kwarg_filtering_drops_unknown(fresh_registry):
  @fresh_registry.register("filtered")
  class Filtered:
    def __init__(self, a, b):
      self.a = a
      self.b = b

  # 'unknown_kwarg' should be silently dropped
  obj = fresh_registry.create("filtered", a=1, b=2, unknown_kwarg=99)
  assert obj.a == 1
  assert obj.b == 2


def test_kwarg_filtering_passthrough_var_kwargs(fresh_registry):
  @fresh_registry.register("varkw")
  class VarKw:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  obj = fresh_registry.create("varkw", x=1, y=2, z=3)
  assert obj.kwargs == {"x": 1, "y": 2, "z": 3}


def test_available_sorted(fresh_registry):
  @fresh_registry.register("c")
  class C:
    pass

  @fresh_registry.register("a")
  class A:
    pass

  @fresh_registry.register("b")
  class B:
    pass

  assert fresh_registry.available() == ["a", "b", "c"]


def test_filter_kwargs_for_target_static():
  # (1) Direct test of the static helper method
  def fn(x, y):
    return x + y

  filtered, dropped = Registry._filter_kwargs_for_target(fn, {"x": 1, "y": 2, "z": 3})
  assert filtered == {"x": 1, "y": 2}
  assert dropped == {"z"}
