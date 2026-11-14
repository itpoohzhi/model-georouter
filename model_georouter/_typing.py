"""Общие типы JSON и совместимый с Python 3.11 маркер переопределения."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeVar

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
__all__ = ["JsonValue", "override"]

if TYPE_CHECKING:
    from typing import override
else:
    _Method = TypeVar("_Method")

    def override(method: _Method) -> _Method:
        """На Python 3.11 сохраняет метод без добавления зависимости."""
        return method
