"""Core type transformation functions for tytr."""
from __future__ import annotations

import sys
from typing import (
    Literal,
    NotRequired,
    Required,
    get_args,
    get_origin,
    get_type_hints,
)

# Always use typing_extensions.TypedDict for consistency
# It has __closed__ attribute and supports the closed parameter (PEP 728)
if sys.version_info < (3, 13):
    from typing_extensions import TypedDict
else:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        # Fallback to typing.TypedDict if typing_extensions not available
        from typing import TypedDict  # type: ignore

from tytr._internal import _is_typeddict, _resolve_generic_typeddict

# ============================================================================
# Error Classes
# ============================================================================

class RenameScheme(TypedDict):
    suffix: NotRequired[str]
    prefix: NotRequired[str]

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
        hints = get_type_hints(tp, include_extras=True)
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
        hints = get_type_hints(tp, include_extras=True)
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
        hints = get_type_hints(tp, include_extras=True)
        value_types = list(hints.values())

        if not value_types:
            # Empty TypedDict - no values
            from typing import Never

            return Never  # type: ignore
        elif len(value_types) == 1:
            return value_types[0]
        else:
            # Create a Union of all value types
            from functools import reduce
            from operator import or_

            return reduce(or_, value_types)  # type: ignore

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
        hints = get_type_hints(tp, include_extras=True)
        value_types = list(hints.values())
        if value_types:
            if len(value_types) == 1:
                return value_types[0]
            from functools import reduce
            from operator import or_

            return reduce(or_, value_types)  # type: ignore

    raise TypeError(f"Cannot extract value type from {tp}")


# ============================================================================
# Core Type Transformation Functions
# ============================================================================


def flatten_fields(
    model_cls: type,
    prefix: str,
    key_delimiter: str,
    localns: dict[str, object] | None = None,
    globalns: dict[str, object] | None = None,
) -> dict[str, type]:
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
    localns : dict, optional
        Local namespace for resolving type annotations
    globalns : dict, optional
        Global namespace for resolving type annotations

    Returns
    -------
    dict[str, type]
        A dictionary mapping field names to their types
    """

    flattened = {}
    # Use get_type_hints to resolve ForwardRefs
    hints = get_type_hints(
        model_cls, globalns=globalns, localns=localns, include_extras=True
    )

    for field_name, field_type in hints.items():
        # Check for delimiter collision
        if key_delimiter in field_name:
            raise DelimiterCollisionError(
                f"The delimiter '{key_delimiter}' occurs in the field name "
                f"'{field_name}' on the class {model_cls.__name__}. "
                f"Use a different delimiter or rename the field to avoid "
                f"this conflict."
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
                        origin, get_args(inner_type), localns=localns, globalns=globalns
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
            nested_fields = flatten_fields(
                type_to_flatten,
                full_key,
                key_delimiter,
                localns=localns,
                globalns=globalns,
            )
            # Re-wrap each nested field with the same wrapper if present
            if wrapper is not None:
                nested_fields = {k: wrapper[v] for k, v in nested_fields.items()}
            flattened.update(nested_fields)
        else:
            # It's a primitive type - keep the wrapper if present
            flattened[full_key] = field_type

    return flattened


def to_typeddict(
    cls: type,
    *,
    name: str | None = None,
    closed: bool | None = None,
    localns: dict[str, object] | None = None,
    globalns: dict[str, object] | None = None,
) -> type:
    """
    Convert a Python class's annotations to a TypedDict.

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
    localns : dict, optional
        Local namespace for resolving type annotations (pass locals())
    globalns : dict, optional
        Global namespace for resolving type annotations (pass globals())

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

    Notes
    -----
    When using `from __future__ import annotations` with classes defined in local
    scopes (e.g., inside functions), pass `localns=locals()` and `globalns=globals()`
    to resolve type annotations correctly.
    """
    # Use get_type_hints to resolve ForwardRefs
    hints = get_type_hints(
        cls, globalns=globalns, localns=localns, include_extras=True
    )
    type_name = name if name is not None else cls.__name__

    # typing_extensions.TypedDict supports the closed parameter
    if closed is not None:
        return TypedDict(type_name, hints, closed=closed)  # type: ignore
    else:
        return TypedDict(type_name, hints)  # type: ignore


def _apply_name_scheme(base_name: str, name_scheme: RenameScheme | None) -> str:
    """Apply prefix/suffix to type name."""
    if name_scheme is None:
        return base_name
    name = base_name
    if "prefix" in name_scheme:
        name = name_scheme["prefix"] + name
    if "suffix" in name_scheme:
        name = name + name_scheme["suffix"]
    return name


def _should_convert_to_typeddict(tp: type) -> bool:
    """Check if a type should be converted to TypedDict."""
    # Already a TypedDict - don't convert
    if _is_typeddict(tp):
        return False
    # Regular class with annotations - convert
    try:
        return (
            isinstance(tp, type)
            and hasattr(tp, "__annotations__")
            and len(tp.__annotations__) > 0
        )
    except (TypeError, AttributeError):
        return False


def _extract_inner_type(field_type: type) -> type:
    """Extract inner type from Required/NotRequired wrapper."""
    origin = get_origin(field_type)
    if origin is Required or origin is NotRequired:
        return get_args(field_type)[0]
    return field_type


def _substitute_types_in_annotation(
    annotation: type, type_map: dict[type, str]
) -> type | str:
    """
    Recursively substitute types in a type annotation.

    For example, converts tuple[User] to tuple["UserDict"]
    if User -> "UserDict" is in type_map.

    Returns string forward references for converted types,
    which is useful for code generation.
    """
    # Check if this type itself needs substitution
    if annotation in type_map:
        return type_map[annotation]

    # Handle Required/NotRequired wrappers
    origin = get_origin(annotation)
    if origin is Required:
        inner = get_args(annotation)[0]
        substituted_inner = _substitute_types_in_annotation(inner, type_map)
        return Required[substituted_inner]  # type: ignore
    elif origin is NotRequired:
        inner = get_args(annotation)[0]
        substituted_inner = _substitute_types_in_annotation(inner, type_map)
        return NotRequired[substituted_inner]  # type: ignore

    # Handle generic types (e.g., tuple[User], list[User], dict[str, User])
    if origin is not None:
        args = get_args(annotation)
        if args:
            # Recursively substitute types in all arguments
            new_args = tuple(
                _substitute_types_in_annotation(arg, type_map) for arg in args
            )
            # Reconstruct the generic type with substituted arguments
            if new_args != args:
                return origin[new_args]  # type: ignore

    # No substitution needed
    return annotation


def _process_class_recursive(
    cls: type,
    result: dict[str, type],
    name_scheme: RenameScheme | None,
    closed: bool | None,
    globalns: dict[str, object] | None,
    localns: dict[str, object] | None,
) -> tuple[type, dict[type, str]]:
    """
    Process a class and its nested classes recursively.

    Returns a tuple of (converted_typeddict, type_mapping) where type_mapping
    maps original classes to their TypedDict string names (for forward refs).
    """
    # Get type hints for this class
    hints = get_type_hints(
        cls, globalns=globalns, localns=localns, include_extras=True
    )

    # Type mapping: maps original types to their string names
    type_map: dict[type, str] = {}

    # First pass: identify and convert all nested classes
    for _, field_type in hints.items():
        # Extract inner type from Required/NotRequired
        inner_type = _extract_inner_type(field_type)

        # Check if we need to recursively process this type
        if _should_convert_to_typeddict(inner_type):
            # Recursively process the nested class
            nested_td, nested_type_map = _process_class_recursive(
                inner_type, result, name_scheme, closed, globalns, localns
            )

            # Store the nested TypedDict in result
            nested_name = _apply_name_scheme(inner_type.__name__, name_scheme)
            if nested_name in result:
                msg = (
                    f"Name collision: multiple types would be named '{nested_name}'. "
                    f"Consider using a more specific name_scheme to avoid conflicts."
                )
                raise ValueError(msg)
            result[nested_name] = nested_td

            # Update type mapping with string name
            type_map[inner_type] = nested_name
            type_map.update(nested_type_map)

        # Also check for types nested in generics (e.g., tuple[User], list[User])
        origin = get_origin(inner_type)
        if origin is not None:
            args = get_args(inner_type)
            for arg in args:
                if _should_convert_to_typeddict(arg):
                    # Recursively process
                    nested_td, nested_type_map = _process_class_recursive(
                        arg, result, name_scheme, closed, globalns, localns
                    )
                    nested_name = _apply_name_scheme(arg.__name__, name_scheme)
                    if nested_name in result:
                        msg = (
                            f"Name collision: multiple types would be named "
                            f"'{nested_name}'. Consider using a more specific "
                            f"name_scheme to avoid conflicts."
                        )
                        raise ValueError(msg)
                    result[nested_name] = nested_td
                    type_map[arg] = nested_name
                    type_map.update(nested_type_map)

    # Second pass: build the TypedDict with substituted types
    converted_hints = {}
    for field_name, field_type in hints.items():
        # Substitute any converted types in the annotation
        converted_hints[field_name] = _substitute_types_in_annotation(
            field_type, type_map
        )

    # Create TypedDict for this class
    td_name = _apply_name_scheme(cls.__name__, name_scheme)
    if closed is not None:
        td = TypedDict(td_name, converted_hints, closed=closed)  # type: ignore
    else:
        td = TypedDict(td_name, converted_hints)  # type: ignore

    return td, type_map


def to_typeddict_recursive(
    cls: type,
    *,
    name_scheme: RenameScheme | None = None,
    closed: bool | None = None,
    localns: dict[str, object] | None = None,
    globalns: dict[str, object] | None = None) -> dict[str, type]:
    """
    Recursively convert a Python class's annotations to a TypedDict. Any value in
    `cls.__annotations__` that has an `__annotations__` attribute will also be converted
    to a TypedDict.

    Parameters
    ----------
    cls : type
        The class to convert (can be a regular class, Pydantic model, etc.)
    name_scheme : ReameScheme
        Name transformation to apply when generating typeddicts for fields.
    closed : bool, optional
        Whether the TypedDict should be closed (PEP 728)
    localns : dict, optional
        Local namespace for resolving type annotations (pass locals())
    globalns : dict, optional
        Global namespace for resolving type annotations (pass globals())

    Returns
    -------
    dict[str, type]
        A dictionary mapping type names to TypedDict types. Simple annotations
        are preserved. Annotations for classes are converted to TypedDicts.

    Examples
    --------
    >>> class User:
    ...     name: str
    ...     age: int
    >>> class UserGroup:
    ...     name: str
    ...     users: tuple[User]
    >>> type_tree = to_typeddict_recursive(UserGroup, name_scheme={"suffix": "Dict"})
    >>> type_tree["UserDict"]
    >>> # TypedDict('User', {'name': str, 'age': int})
    >>> type_tree["UserGroupDict"]
    >>> # TypedDict('UserGroupDict', {'name': str, 'users': tuple[UserDict]})

    Notes
    -----
    When using `from __future__ import annotations` with classes defined in local
    scopes (e.g., inside functions), pass `localns=locals()` and `globalns=globals()`
    to resolve type annotations correctly.
    """
    result: dict[str, type] = {}

    # Process the root class
    root_td, _ = _process_class_recursive(
        cls, result, name_scheme, closed, globalns, localns
    )
    root_name = _apply_name_scheme(cls.__name__, name_scheme)
    result[root_name] = root_td

    return result

def flatten(
    cls: type,
    *,
    key_delimiter: str | None = None,
    name: str | None = None,
    closed: bool | None = None,
    localns: dict[str, object] | None = None,
    globalns: dict[str, object] | None = None,
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
    localns : dict, optional
        Local namespace for resolving type annotations (pass locals())
    globalns : dict, optional
        Global namespace for resolving type annotations (pass globals())

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

    Notes
    -----
    When using `from __future__ import annotations` with classes defined in local
    scopes (e.g., inside functions), pass `localns=locals()` and `globalns=globals()`
    to resolve type annotations correctly.
    """
    if key_delimiter is None:
        td_delimiter = "_"
    else:
        td_delimiter = key_delimiter
    flattened_fields = flatten_fields(
        cls, key_delimiter=td_delimiter, prefix="", localns=localns, globalns=globalns
    )
    if name is None:
        td_name = cls.__name__
    else:
        td_name = name

    # typing_extensions.TypedDict supports the closed parameter
    if closed is not None:
        return TypedDict(td_name, flattened_fields, closed=closed)
    else:
        return TypedDict(td_name, flattened_fields)
