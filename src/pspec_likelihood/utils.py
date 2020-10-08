"""Miscellaneous utility functions."""
from pathlib import Path


def listify(x):
    """If x is not a list, put it into one."""
    if not isinstance(x, (list, tuple, set)):
        return [x]
    else:
        return list(x)


def make_list_of_paths(x):
    """Convert a list of strings (or single string) into a list of Path objects."""
    x = listify(x)
    return [Path(xx) for xx in x]
