"""Miscellaneous utility functions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def listify(x: Any) -> list:
    """If x is not a list, put it into one."""
    return [x] if not isinstance(x, (list, tuple, set)) else list(x)


def make_list_of_paths(x: str | Path | Sequence[str | Path]) -> list[Path]:
    """Convert a list of strings (or single string) into a list of Path objects."""
    return [Path(xx) for xx in listify(x)]
