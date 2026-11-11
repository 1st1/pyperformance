# flake8: noqa

# This is a vendored copy of the "immutables" package (version 0.21)
# stripped of the C accelerator module: the pyperformance "gel" benchmarks
# are meant to exercise the Python interpreter, so the pure-Python
# implementation of the HAMT-based immutable mapping is always used.

from .map import Map

from ._version import __version__

__all__ = 'Map',
