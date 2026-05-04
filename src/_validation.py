"""Internal validation helpers. Raise stdlib exceptions with context.

Underscore prefix marks these as internal — not part of the public src/ API.
Use at the boundary of public functions only; trust internal helpers.
"""
from __future__ import annotations
from collections.abc import Iterable
import pandas as pd


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: required columns missing from dataframe: {missing}")


def require_nonempty(df: pd.DataFrame, *, where: str) -> None:
    if len(df) == 0:
        raise ValueError(f"{where}: input dataframe is empty")


def require_positive_int(name: str, value: int, *, where: str, allow_zero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{where}: {name!r} must be int, got {type(value).__name__}")
    if (value < 0) or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{where}: {name!r} must be {bound}, got {value}")


def require_in_range(name: str, value: float, low: float, high: float, *, where: str) -> None:
    if not (low <= value <= high):
        raise ValueError(f"{where}: {name!r} must be in [{low}, {high}], got {value}")
