"""Base class and helpers shared by every StillHere contract.

A contract is a frozen dataclass. On construction it coerces JSON-friendly input (lists, ISO-8601
strings, enum values, nested dicts) into its declared field types, checks primitive types, and then
runs the class's own ``validate()``. Free-form ``Dict`` fields are deep-frozen (``FrozenDict``, nested lists
become tuples), so a contract cannot be changed through its inner collections after validation. The same path backs ``from_dict`` and ``from_json``, so an object
built in code and an object loaded from disk go through identical checks.

Standard library only, Python 3.10+.
"""
from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, Tuple, TypeVar, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = "0.1"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

C = TypeVar("C", bound="Contract")


class ValidationError(ValueError):
    """Raised when a contract's fields are missing, mistyped or inconsistent."""


def require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValidationError(f"{name} must match {_ID_RE.pattern!r}, got {value!r}")


def require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")


def require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValidationError(f"{name} must be a lowercase 64-character sha256 hex digest")


def require_semver(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SEMVER_RE.match(value):
        raise ValidationError(f"{name} must look like MAJOR.MINOR.PATCH, got {value!r}")


def require_unique(name: str, values) -> None:
    seen = set()
    for v in values:
        if v in seen:
            raise ValidationError(f"{name} contains duplicate value {v!r}")
        seen.add(v)


def parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"not an ISO-8601 datetime: {value!r}") from exc


class FrozenDict(dict):
    """A read-only dict. Still a ``dict`` for ``isinstance``, ``json.dumps`` and equality checks."""

    def _readonly(self, *args, **kwargs):
        raise TypeError("contract mappings are read-only; build a new contract instead")

    __setitem__ = __delitem__ = __ior__ = _readonly
    clear = pop = popitem = setdefault = update = _readonly

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo) -> "FrozenDict":
        return self

    def __reduce__(self):
        return (FrozenDict, (dict(self),))


def _deep_freeze(value: Any) -> Any:
    """Copy JSON-like data into immutable containers: dict -> FrozenDict, list/tuple -> tuple."""
    if isinstance(value, dict):
        return FrozenDict((k, _deep_freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _is_union(tp: Any) -> bool:
    return get_origin(tp) is Union or isinstance(tp, types.UnionType)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """Convert a JSON-friendly value into ``tp`` and check its type."""
    if tp is Any:
        return value
    if _is_union(tp):
        options = [a for a in get_args(tp) if a is not type(None)]
        if value is None:
            if len(options) == len(get_args(tp)):
                raise ValidationError(f"{where} must not be None")
            return None
        if len(options) != 1:
            raise TypeError(f"{where}: only Optional[X] unions are supported in contracts")
        return _coerce(options[0], value, where)
    if value is None:
        raise ValidationError(f"{where} must not be None")

    origin = get_origin(tp)
    if origin in (tuple, Tuple):
        args = get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"{where} must be a list")
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(f"{where}: contracts use Tuple[X, ...] for sequences")
        return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ValidationError(f"{where} must be an object")
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{where} must be JSON-serialisable") from exc
        return _deep_freeze(value)

    if isinstance(tp, type) and is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if isinstance(value, dict):
            return tp.from_dict(value)
        raise ValidationError(f"{where} must be a {tp.__name__} or an object")
    if isinstance(tp, type) and issubclass(tp, Enum):
        if isinstance(value, tp):
            return value
        try:
            return tp(value)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in tp)
            raise ValidationError(f"{where} must be one of: {allowed}; got {value!r}") from exc
    if tp is datetime:
        dt = value if isinstance(value, datetime) else parse_datetime(value)
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValidationError(f"{where} must be timezone-aware, got {value!r}")
        return dt
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{where} must be a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"{where} must be an integer")
        return value
    if tp in (str, bool):
        if not isinstance(value, tp):
            raise ValidationError(f"{where} must be {tp.__name__}")
        return value
    raise TypeError(f"{where}: unsupported contract field type {tp!r}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        omit = getattr(value, "_omit_when_empty", ())
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)
                if not (f.name in omit and not getattr(value, f.name))}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Contract:
    """Base for all contracts: coercion, type checks, validation and JSON round-trips."""

    _hints_cache: ClassVar[Dict[type, Dict[str, Any]]] = {}
    # Fields added after a format was in use: left out of to_dict() while empty, so data that does not use them
    # stays readable by readers that predate them (from_dict rejects unknown fields).
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        cls = type(self)
        hints = Contract._hints_cache.get(cls)
        if hints is None:
            hints = Contract._hints_cache[cls] = get_type_hints(cls)
        for f in fields(self):
            value = _coerce(hints[f.name], getattr(self, f.name), f"{cls.__name__}.{f.name}")
            object.__setattr__(self, f.name, value)
        self.validate()

    def validate(self) -> None:
        """Cross-field checks. Subclasses override; raise ValidationError on failure."""

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls: type[C], data: Dict[str, Any]) -> C:
        if not isinstance(data, dict):
            raise ValidationError(f"{cls.__name__} expects an object, got {type(data).__name__}")
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValidationError(f"{cls.__name__} got unknown field(s): {', '.join(unknown)}")
        try:
            return cls(**data)
        except TypeError as exc:  # missing required fields
            raise ValidationError(f"{cls.__name__}: {exc}") from exc

    @classmethod
    def from_json(cls: type[C], text: str) -> C:
        return cls.from_dict(json.loads(text))


def canonical_json(data: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace, UTF-8 text)."""
    return json.dumps(_to_jsonable(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
