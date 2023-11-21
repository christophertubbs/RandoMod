"""
Common functions and classes used throughout the application
"""
from __future__ import annotations

import inspect
import os
import typing
import functools
import numpy

from datetime import tzinfo


@functools.cache
def get_datetime_format(timezone: tzinfo) -> str:
    if timezone is None:
        return os.environ.get("RANDOMOD_DATETIME_TZ_UNAWARE_FORMAT", "%Y-%m-%d %H:%M+0000")
    return os.environ.get("RANDOMOD_DATETIME_TZ_AWARE_FORMAT", "%Y-%m-%d %H:%M%z")


def is_sequence_type(value: typing.Any) -> bool:
    """
    Checks to see if a value is one that can be interpreted as a collection of values

    Why not just use `isinstance(value, typing.Sequence)`? Strings, bytes, and maps ALL count as sequences

    Args:
        value: The value to check

    Returns:
        Whether the passed value is a sequence
    """
    is_collection = value is not None
    is_collection = is_collection and not isinstance(value, (str, bytes, typing.Mapping))
    is_collection = is_collection and isinstance(value, typing.Sequence)

    return is_collection


def value_is_number(value: typing.Any) -> bool:
    """
    Whether the passed in value may be interpreted as a number

    Args:
        value: The value to check

    Returns:
        Whether the value may be interpreted as a number
    """
    if isinstance(value, str) and value.isnumeric():
        return True
    elif isinstance(value, bytes) and value.decode().isnumeric():
        return True
    elif hasattr(type(value), "__mro__") and numpy.number in inspect.getmro(type(value)):
        return True

    return isinstance(value, int) or isinstance(value, float) or isinstance(value, complex)