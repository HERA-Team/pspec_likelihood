"""Miscellaneous utility functions."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def listify(x: Any) -> list:
    """If x is not a list, put it into one."""
    return [x] if not isinstance(x, (list, tuple, set)) else list(x)


def make_list_of_paths(x: str | Path | Sequence[str | Path]) -> list[Path]:
    """Convert a list of strings (or single string) into a list of Path objects."""
    return [Path(xx) for xx in listify(x)]


def normalize_wf(arr) -> np.array:
    """Check normalization of window functions and convert to np array."""
    sum_per_bin = np.sum(arr, axis=1)[:, None]
    if np.allclose(sum_per_bin[sum_per_bin != 0.0], 1.0):
        return np.array(arr)
    else:
        warnings.warn("Had to normalize window_function.")
        return np.divide(arr, sum_per_bin, where=sum_per_bin != 0)
