"""Strict dataclass conversion, without model/runtime dependencies."""

from __future__ import annotations

import math
import re
import types
from dataclasses import fields, is_dataclass
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def to_dict(value: Any) -> Any:
    """Return only finite, JSON-compatible values; tuples become lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"Not a finite JSON value: {type(value).__name__}")


def _decode(annotation: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return to_dict(value)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(option, value, path)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"{path}: incompatible value for {annotation}")
    if origin is Literal:
        if any(type(value) is type(option) and value == option for option in args):
            return value
    elif is_dataclass(annotation):
        return from_dict(annotation, value)
    elif origin in (tuple, list):
        if isinstance(value, (list, tuple)):
            item_types = args
            if origin is list or (len(args) == 2 and args[1] is Ellipsis):
                item_types = (args[0],) * len(value)
            if len(value) == len(item_types):
                items = [_decode(kind, item, path) for kind, item in zip(item_types, value)]
                return tuple(items) if origin is tuple else items
    elif origin is dict:
        if isinstance(value, dict):
            return {
                _decode(args[0], key, path): _decode(args[1], item, path)
                for key, item in value.items()
            }
    elif annotation is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif annotation is type(None):
        if value is None:
            return None
    elif type(value) is annotation:
        return value
    raise ValueError(f"{path}: invalid type/value for {annotation}")


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    names = {field.name for field in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    hints = get_type_hints(cls)
    try:
        return cls(**{key: _decode(hints[key], value, key) for key, value in data.items()})
    except TypeError as exc:
        raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc


def validate_types(value: Any) -> None:
    """Validate direct Python construction as strictly as deserialization."""
    hints = get_type_hints(type(value))
    for field in fields(value):
        item = getattr(value, field.name)
        annotation = hints[field.name]
        _decode(annotation, to_dict(item), field.name)


def nonempty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def finite(value: float | int, name: str) -> None:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")


def positive(value: float | int, name: str, *, zero: bool = False) -> None:
    finite(value, name)
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def safe_name(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError(f"{name} must be a simple name (letters, digits, _ or -)")


def resolution(value: tuple[int, int]) -> None:
    if len(value) != 2 or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError("resolution must contain two positive integers")
