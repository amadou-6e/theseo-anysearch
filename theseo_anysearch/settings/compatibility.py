"""Compatibility helpers for historical settings APIs."""

from typing import Any, ClassVar


class NestedFieldAccessMixin:
    """Expose selected nested fields as ``container__field`` attributes."""

    exposed_nested_fields: ClassVar[tuple[str, ...]] = ()

    def __getattr__(self, name: str) -> Any:
        container_name, separator, nested_name = name.partition("__")
        if separator and container_name in self.exposed_nested_fields:
            container = getattr(self, container_name)
            if nested_name in type(container).model_fields:
                return getattr(container, nested_name)
        return super().__getattr__(name)


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into a dictionary in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value