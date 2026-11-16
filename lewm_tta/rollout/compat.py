"""Small lazy import aliases for the former stage-specific namespaces."""

from importlib import import_module
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType


class _Alias(ModuleType):
    def __init__(self, name: str, target: str, attributes: dict):
        super().__init__(name, f"Compatibility alias for lewm_tta.rollout.{target}")
        self.__spec__ = ModuleSpec(name, loader=None)
        self.__package__ = name.rpartition(".")[0]
        self._target = target
        self._attributes = attributes

    def __getattr__(self, name):
        if name == "__path__":
            raise AttributeError(name)
        target, attribute = self._attributes.get(name, (self._target, name))
        return getattr(import_module(f"lewm_tta.rollout.{target}"), attribute)


def expose(package: str, modules: dict) -> None:
    for local, (target, attributes) in modules.items():
        name = f"{package}.{local}"
        if name not in sys.modules:
            sys.modules[name] = _Alias(name, target, attributes)
        setattr(sys.modules[package], local, sys.modules[name])
