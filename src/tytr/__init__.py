from __future__ import annotations

from ._version import __version__
from .core import flatten, key_of, to_typeddict, value_of
from .testing import assert_type_equals, make_type_test
from .utilities import (
    exclude,
    extract,
    non_nullable,
    omit,
    parameters,
    partial,
    pick,
    readonly,
    required,
    return_type,
)

__all__ = [
    "__version__",
    # Testing utilities
    "assert_type_equals",
    "exclude",
    "extract",
    # Core transformation
    "flatten",
    # Type inspection utilities
    "key_of",
    "make_type_test",
    "non_nullable",
    "omit",
    "parameters",
    "partial",
    "pick",
    "readonly",
    "required",
    "return_type",
    "to_typeddict",
    "value_of"
]
