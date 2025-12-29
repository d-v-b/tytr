"""
Type transformation utilities inspired by TypeScript's utility types.

These are type-to-type functions that create new types from existing ones,
useful for test-time type validation and constraint checking.

Reference: https://www.typescriptlang.org/docs/handbook/utility-types.html
"""
from __future__ import annotations

import types
from collections.abc import Callable
from typing import (
    Any,
    NotRequired,
    Required,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypedDict

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def partial(cls: type[T], *, name: str | None = None) -> type[T]:
    """
    Constructs a type with all properties of T set to optional (NotRequired).

    TypeScript equivalent: Partial<T>

    Args:
        cls: The TypedDict class to make partial
        name: Optional custom name for the resulting type

    Returns:
        A new TypedDict with all fields wrapped in NotRequired

    Example:
        >>> class User(TypedDict):
        ...     name: str
        ...     age: int
        >>> PartialUser = partial(User)
        >>> # PartialUser has name: NotRequired[str], age: NotRequired[int]
    """
    hints = get_type_hints(cls, include_extras=True)

    # Make all fields NotRequired
    partial_fields = {}
    for field_name, field_type in hints.items():
        origin = get_origin(field_type)

        # If already wrapped in Required/NotRequired, unwrap and rewrap with NotRequired
        if origin is Required:
            inner = get_args(field_type)[0]
            partial_fields[field_name] = NotRequired[inner]
        elif origin is NotRequired:
            # Already NotRequired, keep as is
            partial_fields[field_name] = field_type
        else:
            # Not wrapped, wrap with NotRequired
            partial_fields[field_name] = NotRequired[field_type]

    type_name = name if name is not None else f"Partial{cls.__name__}"
    return TypedDict(type_name, partial_fields)  # type: ignore


def required(cls: type[T], *, name: str | None = None) -> type[T]:
    """
    Constructs a type with all properties of T set to required.

    TypeScript equivalent: Required<T>

    Args:
        cls: The TypedDict class to make all fields required
        name: Optional custom name for the resulting type

    Returns:
        A new TypedDict with all fields wrapped in Required

    Example:
        >>> class User(TypedDict):
        ...     name: str
        ...     age: NotRequired[int]
        >>> RequiredUser = required(User)
        >>> # RequiredUser has name: Required[str], age: Required[int]
    """
    hints = get_type_hints(cls, include_extras=True)

    # Make all fields Required
    required_fields = {}
    for field_name, field_type in hints.items():
        origin = get_origin(field_type)

        # If wrapped in NotRequired/Required, unwrap and rewrap with Required
        if origin is NotRequired:
            inner = get_args(field_type)[0]
            required_fields[field_name] = Required[inner]
        elif origin is Required:
            # Already Required, keep as is
            required_fields[field_name] = field_type
        else:
            # Not wrapped, wrap with Required
            required_fields[field_name] = Required[field_type]

    type_name = name if name is not None else f"Required{cls.__name__}"
    return TypedDict(type_name, required_fields)  # type: ignore


def readonly(cls: type[T], *, name: str | None = None) -> type[T]:
    """
    Constructs a type with all properties of T set to readonly.

    Uses typing_extensions.ReadOnly (PEP 705) to mark fields as readonly.

    TypeScript equivalent: Readonly<T>

    Args:
        cls: The TypedDict class
        name: Optional custom name for the resulting type

    Returns:
        A new TypedDict with all fields wrapped in ReadOnly

    Example:
        >>> class User(TypedDict):
        ...     name: str
        ...     age: int
        >>> ReadonlyUser = readonly(User)
        >>> # ReadonlyUser has name: ReadOnly[str], age: ReadOnly[int]
    """
    from typing_extensions import ReadOnly

    hints = get_type_hints(cls, include_extras=True)

    # Wrap all fields with ReadOnly, preserving Required/NotRequired wrappers
    readonly_fields = {}
    for field_name, field_type in hints.items():
        origin = get_origin(field_type)

        # Check if already wrapped in ReadOnly
        if origin is ReadOnly:
            # Already readonly, keep as is
            readonly_fields[field_name] = field_type
        elif origin is Required:
            # Preserve Required wrapper: Required[ReadOnly[T]]
            inner = get_args(field_type)[0]
            inner_origin = get_origin(inner)
            if inner_origin is ReadOnly:
                # Already ReadOnly inside Required
                readonly_fields[field_name] = field_type
            else:
                readonly_fields[field_name] = Required[ReadOnly[inner]]
        elif origin is NotRequired:
            # Preserve NotRequired wrapper: NotRequired[ReadOnly[T]]
            inner = get_args(field_type)[0]
            inner_origin = get_origin(inner)
            if inner_origin is ReadOnly:
                # Already ReadOnly inside NotRequired
                readonly_fields[field_name] = field_type
            else:
                readonly_fields[field_name] = NotRequired[ReadOnly[inner]]
        else:
            # Not wrapped, wrap with ReadOnly
            readonly_fields[field_name] = ReadOnly[field_type]

    type_name = name if name is not None else f"Readonly{cls.__name__}"
    return TypedDict(type_name, readonly_fields)  # type: ignore


def pick(cls: type[T], keys: tuple[str, ...], *, name: str | None = None) -> type[T]:
    """
    Constructs a type by picking the set of properties K from T.

    TypeScript equivalent: Pick<T, K>

    Args:
        cls: The TypedDict class to pick from
        keys: Tuple of field names to pick
        name: Optional custom name for the resulting type

    Returns:
        A new TypedDict with only the specified fields

    Example:
        >>> class User(TypedDict):
        ...     name: str
        ...     age: int
        ...     email: str
        >>> UserPreview = pick(User, ('name', 'age'))
        >>> # UserPreview has only name: str, age: int
    """
    hints = get_type_hints(cls, include_extras=True)

    picked_fields = {k: v for k, v in hints.items() if k in keys}

    # Check that all requested keys exist
    missing_keys = set(keys) - set(hints.keys())
    if missing_keys:
        raise ValueError(f"Keys {missing_keys} not found in {cls.__name__}")

    type_name = name if name is not None else f"Pick{cls.__name__}"
    return TypedDict(type_name, picked_fields)  # type: ignore


def omit(cls: type[T], keys: tuple[str, ...], *, name: str | None = None) -> type[T]:
    """
    Constructs a type by omitting the set of properties K from T.

    TypeScript equivalent: Omit<T, K>

    Args:
        cls: The TypedDict class to omit from
        keys: Tuple of field names to omit
        name: Optional custom name for the resulting type

    Returns:
        A new TypedDict without the specified fields

    Example:
        >>> class User(TypedDict):
        ...     name: str
        ...     age: int
        ...     password: str
        >>> PublicUser = omit(User, ('password',))
        >>> # PublicUser has only name: str, age: int
    """
    hints = get_type_hints(cls, include_extras=True)

    omitted_fields = {k: v for k, v in hints.items() if k not in keys}

    type_name = name if name is not None else f"Omit{cls.__name__}"
    return TypedDict(type_name, omitted_fields)  # type: ignore


def exclude(
    union_type: type | types.UnionType, excluded_type: type
) -> type | types.UnionType:
    """
    Exclude types from a Union.

    Constructs a type by excluding from UnionType all union members that are
    assignable to ExcludedMembers.

    TypeScript equivalent: Exclude<UnionType, ExcludedMembers>

    Args:
        union_type: A Union type to filter
        excluded_type: The type(s) to exclude

    Returns:
        A new Union type without the excluded types

    Example:
        >>> T = Union[str, int, bool]
        >>> StringOrInt = exclude(T, bool)
        >>> # Result: Union[str, int]
    """
    origin = get_origin(union_type)

    # Check if it's a Union - handles both typing.Union and types.UnionType (from |)
    if origin is Union or origin is types.UnionType:
        args = get_args(union_type)
        # Filter out the excluded types
        filtered = [arg for arg in args if arg != excluded_type]

        if len(filtered) == 0:
            raise ValueError("Cannot exclude all types from union")
        elif len(filtered) == 1:
            return filtered[0]
        else:
            from functools import reduce
            from operator import or_

            return reduce(or_, filtered)  # type: ignore
    else:
        # Not a union, check if it matches excluded_type
        if union_type == excluded_type:
            raise ValueError("Cannot exclude the only type")
        return union_type


def extract(
    union_type: type | types.UnionType, extracted_type: type
) -> type | types.UnionType:
    """
    Extract types from a Union.

    Constructs a type by extracting from UnionType all union members that are
    assignable to Union.

    TypeScript equivalent: Extract<UnionType, Union>

    Args:
        union_type: A Union type to filter
        extracted_type: The type(s) to extract

    Returns:
        A new type containing only the extracted types

    Example:
        >>> T = Union[str, int, bool]
        >>> StringType = extract(T, str)
        >>> # Result: str
    """
    origin = get_origin(union_type)

    # Check if it's a Union - handles both typing.Union and types.UnionType (from |)
    if origin is Union or origin is types.UnionType:
        args = get_args(union_type)
        # Keep only the types matching extracted_type
        filtered = [arg for arg in args if arg == extracted_type]

        if len(filtered) == 0:
            raise ValueError(f"Type {extracted_type} not found in union")
        elif len(filtered) == 1:
            return filtered[0]
        else:
            from functools import reduce
            from operator import or_

            return reduce(or_, filtered)  # type: ignore
    else:
        # Not a union, check if it matches extracted_type
        if union_type == extracted_type:
            return union_type
        raise ValueError(f"Type {extracted_type} not found")


def non_nullable(t: type | types.UnionType) -> type | types.UnionType:
    """
    Constructs a type by excluding None from T.

    TypeScript equivalent: NonNullable<T>

    Args:
        t: A type (possibly a Union with None)

    Returns:
        The type without None

    Example:
        >>> T = Union[str, None]
        >>> NonNullableT = non_nullable(T)
        >>> # Result: str
    """
    origin = get_origin(t)

    # Check if it's a Union - handles both typing.Union and types.UnionType (from |)
    if origin is Union or origin is types.UnionType:
        args = get_args(t)
        # Filter out None - note that type(None) is NoneType
        none_type = type(None)
        filtered = [arg for arg in args if arg is not none_type]

        if len(filtered) == 0:
            raise ValueError("Cannot remove None from a type that is only None")
        elif len(filtered) == 1:
            return filtered[0]
        else:
            from functools import reduce
            from operator import or_

            return reduce(or_, filtered)  # type: ignore
    else:
        # Not a union, return as is (unless it's None itself)
        if t is type(None):
            raise ValueError("Cannot make None non-nullable")
        return t


def parameters(func: Callable[..., Any]) -> tuple[type, ...]:
    """
    Constructs a tuple type from the types used in the parameters of a function type.

    TypeScript equivalent: Parameters<T>

    Args:
        func: A callable

    Returns:
        A tuple of parameter types

    Example:
        >>> def greet(name: str, age: int) -> str:
        ...     return f"Hello {name}"
        >>> Params = parameters(greet)
        >>> # Result: (str, int)
    """
    # Use get_type_hints to resolve string annotations from __future__ imports
    hints = get_type_hints(func)

    import inspect

    sig = inspect.signature(func)

    param_types = []
    for param_name, param in sig.parameters.items():
        if param_name in hints:
            param_types.append(hints[param_name])
        elif param.annotation != inspect.Parameter.empty:
            param_types.append(param.annotation)
        else:
            param_types.append(Any)

    return tuple(param_types)


def return_type(func: Callable[..., T]) -> type[T]:
    """
    Constructs a type consisting of the return type of function T.

    TypeScript equivalent: ReturnType<T>

    Args:
        func: A callable

    Returns:
        The return type of the function

    Example:
        >>> def greet(name: str) -> str:
        ...     return f"Hello {name}"
        >>> RT = return_type(greet)
        >>> # Result: str
    """
    # Use get_type_hints to resolve string annotations from __future__ imports
    hints = get_type_hints(func)

    # get_type_hints returns a dict with 'return' key for the return type
    if "return" in hints:
        return hints["return"]

    import inspect

    sig = inspect.signature(func)

    if sig.return_annotation != inspect.Signature.empty:
        return sig.return_annotation
    else:
        return Any  # type: ignore
