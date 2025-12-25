"""Core type transformation functions for tytr."""

from typing import Union, get_args, get_origin, get_type_hints, Literal
from typing_extensions import TypedDict, NotRequired, Required
import typing_extensions

from tytr._internal import _is_typeddict, _resolve_generic_typeddict, td_eq, td_repr


# ============================================================================
# Error Classes
# ============================================================================


class DelimiterCollisionError(ValueError):
    """
    Raised when a delimiter occurs in the strings its supposed to delimit.

    This error is raised by `flatten()` when a field name contains the chosen
    delimiter character, which would cause ambiguity in the flattened structure.
    """


# ============================================================================
# Type Inspection Utilities
# ============================================================================


def key_of(tp: type) -> type:
    """
    Extract the union of all key types from a mapping type.

    Similar to TypeScript's `keyof` operator, this function returns a Union type
    representing all possible keys in the mapping.

    For TypedDict types, this returns a Union of Literal types for each key.
    For dict types like dict[str, int], this returns the key type (str).

    Parameters
    ----------
    tp : type
        A mapping type (TypedDict, dict, etc.)

    Returns
    -------
    type
        A Union type of all possible keys, or the key type for generic mappings

    Examples
    --------
    >>> class MyDict(TypedDict):
    ...     name: str
    ...     age: int
    >>> key_of(MyDict)
    typing.Literal['name', 'age']

    >>> key_of(dict[str, int])
    <class 'str'>
    """

    # Check if it's a TypedDict
    if _is_typeddict(tp):
        # Get the keys from the TypedDict
        hints = get_type_hints(tp)
        keys = list(hints.keys())

        if not keys:
            # Empty TypedDict - return Literal with no args (Never-like)
            return Literal[()]  # type: ignore
        elif len(keys) == 1:
            return Literal[keys[0]]  # type: ignore
        else:
            # Create a Union of Literal types for each key
            return Literal[tuple(keys)]  # type: ignore

    # Check if it's a generic dict type (dict[K, V])
    origin = get_origin(tp)
    if origin is dict:
        args = get_args(tp)
        if args:
            return args[0]  # Return the key type
        return str  # Unparameterized dict defaults to str keys

    # For other mapping types, try to extract key type from __class_getitem__
    if hasattr(tp, "__annotations__"):
        # It's a regular class with annotations
        hints = get_type_hints(tp)
        keys = list(hints.keys())
        if keys:
            return Literal[tuple(keys)]  # type: ignore

    raise TypeError(f"Cannot extract key type from {tp}")


def value_of(tp: type) -> type:
    """
    Extract the union of all value types from a mapping type.

    This function returns a Union type representing all possible value types
    in the mapping.

    For TypedDict types, this returns a Union of all value types.
    For dict types like dict[str, int], this returns the value type (int).

    Parameters
    ----------
    tp : type
        A mapping type (TypedDict, dict, etc.)

    Returns
    -------
    type
        A Union type of all possible values, or the value type for generic mappings

    Examples
    --------
    >>> class MyDict(TypedDict):
    ...     name: str
    ...     age: int
    >>> value_of(MyDict)
    typing.Union[str, int]

    >>> value_of(dict[str, int])
    <class 'int'>
    """
    # Check if it's a TypedDict
    if _is_typeddict(tp):
        # Get the value types from the TypedDict
        hints = get_type_hints(tp)
        value_types = list(hints.values())

        if not value_types:
            # Empty TypedDict - no values
            from typing import Never

            return Never  # type: ignore
        elif len(value_types) == 1:
            return value_types[0]
        else:
            # Create a Union of all value types
            return Union[tuple(value_types)]  # type: ignore

    # Check if it's a generic dict type (dict[K, V])
    origin = get_origin(tp)
    if origin is dict:
        args = get_args(tp)
        if args and len(args) >= 2:
            return args[1]  # Return the value type
        from typing import Any

        return Any  # Unparameterized dict defaults to Any values

    # For other mapping types with annotations
    if hasattr(tp, "__annotations__"):
        hints = get_type_hints(tp)
        value_types = list(hints.values())
        if value_types:
            if len(value_types) == 1:
                return value_types[0]
            return Union[tuple(value_types)]  # type: ignore

    raise TypeError(f"Cannot extract value type from {tp}")


# ============================================================================
# Core Type Transformation Functions
# ============================================================================


def flatten_fields(model_cls: type, prefix: str, key_delimiter: str) -> dict[str, type]:
    """
    Recursively flatten fields from a model class.

    Parameters
    ----------
    model_cls : type
        The class to flatten
    prefix : str
        The prefix to prepend to field names
    key_delimiter : str
        The delimiter to use between nested field names

    Returns
    -------
    dict[str, type]
        A dictionary mapping field names to their types
    """

    flattened = {}
    hints = get_type_hints(model_cls, include_extras=True)

    for field_name, field_type in hints.items():
        # Check for delimiter collision
        if key_delimiter in field_name:
            raise DelimiterCollisionError(
                f"The delimiter '{key_delimiter}' occurs in the field name '{field_name}' on the class {model_cls.__name__}. "
                f"Use a different delimiter or rename the field to avoid this conflict."
            )

        full_key = f"{prefix}{key_delimiter}{field_name}" if prefix else field_name

        # Unwrap Required/NotRequired if present
        origin = get_origin(field_type)
        wrapper = None
        inner_type = field_type

        if origin is Required:
            wrapper = Required
            inner_type = get_args(field_type)[0]
        elif origin is NotRequired:
            wrapper = NotRequired
            inner_type = get_args(field_type)[0]

        # Re-check origin after unwrapping Required/NotRequired
        origin = get_origin(inner_type)

        # Check if it's a nested structure
        should_flatten = False
        type_to_flatten = inner_type

        # Check if it's a generic TypedDict (e.g., WithV[int])
        if origin is not None:
            try:
                if (
                    isinstance(origin, type)
                    and hasattr(origin, "__annotations__")
                    and hasattr(origin, "__total__")
                ):
                    # It's a specialized generic TypedDict
                    # We need to resolve the type parameters
                    should_flatten = True
                    type_to_flatten = _resolve_generic_typeddict(
                        origin, get_args(inner_type)
                    )
            except (TypeError, AttributeError):
                pass

        # Check if it's a TypedDict
        if not should_flatten:
            try:
                # TypedDict classes have __annotations__ and are not regular classes
                if (
                    isinstance(inner_type, type)
                    and hasattr(inner_type, "__annotations__")
                    and hasattr(inner_type, "__total__")
                ):
                    should_flatten = True
                    type_to_flatten = inner_type
            except (TypeError, AttributeError):
                pass

        # Check if it's a regular class with annotations
        if not should_flatten:
            try:
                if (
                    isinstance(inner_type, type)
                    and hasattr(inner_type, "__annotations__")
                    and len(inner_type.__annotations__) > 0
                ):
                    should_flatten = True
                    type_to_flatten = inner_type
            except (TypeError, AttributeError):
                pass

        if should_flatten:
            # Keep the nested object itself
            flattened[full_key] = field_type
            # Recursively flatten nested model
            nested_fields = flatten_fields(type_to_flatten, full_key, key_delimiter)
            # Re-wrap each nested field with the same wrapper if present
            if wrapper is not None:
                nested_fields = {k: wrapper[v] for k, v in nested_fields.items()}
            flattened.update(nested_fields)
        else:
            # It's a primitive type - keep the wrapper if present
            flattened[full_key] = field_type

    return flattened


def to_typeddict(
    cls: type, *, name: str | None = None, closed: bool | None = None
) -> type:
    """
    Convert a Python class's annotations to a TypedDict without flattening.

    This is useful for simple type conversion where you want to preserve the
    exact structure of the original class's annotations.

    Parameters
    ----------
    cls : type
        The class to convert (can be a regular class, Pydantic model, etc.)
    name : str, optional
        Custom name for the resulting TypedDict (defaults to cls.__name__)
    closed : bool, optional
        Whether the TypedDict should be closed (PEP 728)

    Returns
    -------
    type
        A TypedDict with the same annotations as the input class

    Examples
    --------
    >>> class User:
    ...     name: str
    ...     age: int
    >>> UserDict = to_typeddict(User)
    >>> # UserDict is TypedDict('User', {'name': str, 'age': int})

    >>> from pydantic import BaseModel
    >>> class Product(BaseModel):
    ...     id: int
    ...     title: str
    >>> ProductDict = to_typeddict(Product)
    >>> # ProductDict preserves the exact field types from Pydantic model
    """
    hints = get_type_hints(cls, include_extras=True)
    type_name = name if name is not None else cls.__name__
    return TypedDict(type_name, hints, closed=closed)  # type: ignore


def flatten(
    cls: type,
    *,
    key_delimiter: str | None = None,
    name: str | None = None,
    closed: bool | None = None,
) -> type:
    """
    Create a TypedDict from a Python class definition, flattening nested structures.

    This recursively flattens nested classes/TypedDicts into a single-level structure
    with delimited keys (e.g., 'address_street' instead of nested 'address.street').

    Parameters
    ----------
    cls : type
        The class to flatten
    key_delimiter : str, optional
        Delimiter for nested keys (default: "_")
    name : str, optional
        Custom name for the resulting TypedDict
    closed : bool, optional
        Whether the TypedDict should be closed (PEP 728)

    Returns
    -------
    type
        A flattened TypedDict with nested fields expanded

    Examples
    --------
    >>> class Address:
    ...     street: str
    ...     city: str
    >>> class Person:
    ...     name: str
    ...     address: Address
    >>> FlatPerson = flatten(Person)
    >>> # FlatPerson has: name, address, address_street, address_city
    """
    if key_delimiter is None:
        td_delimiter = "_"
    else:
        td_delimiter = key_delimiter
    flattened_fields = flatten_fields(cls, key_delimiter=td_delimiter, prefix="")
    if name is None:
        td_name = cls.__name__
    else:
        td_name = name

    return TypedDict(td_name, flattened_fields, closed=closed)
