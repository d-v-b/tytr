"""Internal utility functions - not part of the public API."""

from typing import Generic, TypeVar, get_args, get_origin, get_type_hints
from typing_extensions import get_original_bases
import typing_extensions


def _substitute_type_vars(type_hint, type_map: dict):
    """
    Recursively substitute type variables in a type hint with concrete types.

    Internal helper for resolving generic TypedDicts.
    """
    # Check if it's a TypeVar
    if isinstance(type_hint, TypeVar):
        return type_map.get(type_hint, type_hint)

    # Check if it's a generic type with arguments
    origin = get_origin(type_hint)
    if origin is not None:
        args = get_args(type_hint)
        if args:
            # Recursively substitute in the arguments
            new_args = tuple(_substitute_type_vars(arg, type_map) for arg in args)
            # Reconstruct the type with new arguments
            return origin[new_args]

    # Return the type hint as-is if no substitution needed
    return type_hint


def _resolve_generic_typeddict(
    generic_class: type, type_args: tuple[object, ...]
) -> type:
    """
    Resolve a generic TypedDict by substituting type parameters with concrete types.

    For example, given WithV[int] where WithV is Generic[V], this creates a new
    class with V replaced by int in all annotations.

    Internal helper used by flatten_fields to handle generic TypedDicts.
    """
    # Get the type parameters from original bases
    type_params = []
    for base in get_original_bases(generic_class):
        origin = get_origin(base)
        # Check if this is a Generic[T, U, ...] base
        if origin is not None:
            if origin is Generic or (
                isinstance(origin, type) and issubclass(origin, Generic)
            ):
                type_params.extend(get_args(base))

    # Create a mapping from type variables to concrete types
    type_map = dict(zip(type_params, type_args))

    # Get annotations with generics included
    annotations = get_type_hints(generic_class, include_extras=True)

    # Substitute type variables with concrete types
    resolved_annotations = {}
    for key, value in annotations.items():
        resolved_annotations[key] = _substitute_type_vars(value, type_map)

    # Create a new TypedDict with resolved annotations
    # We'll return a simple class with __annotations__ that can be used by flatten_fields
    class ResolvedTypedDict:
        __annotations__ = resolved_annotations
        __total__ = getattr(generic_class, "__total__", True)
        __name__ = generic_class.__name__

    return ResolvedTypedDict


def _is_typeddict(tp: type) -> bool:
    """
    Check if a type is a TypedDict.

    Internal helper for type inspection utilities.
    """
    return (
        hasattr(tp, "__annotations__")
        and hasattr(tp, "__total__")
        and hasattr(tp, "__required_keys__")
        and hasattr(tp, "__optional_keys__")
    )


def td_repr(td: typing_extensions._TypedDictMeta) -> str:
    """
    Return a string representation of a TypedDict including its name, fields, types, and parameters.

    Internal helper for testing and debugging.
    """
    lines = [f"TypedDict {td.__name__}("]
    for field_name, field_type in td.__annotations__.items():
        lines.append(f"  {field_name}: {field_type}")
    lines.append(f") closed={td.__closed__}")
    return "\n".join(lines)


def td_eq(a: typing_extensions._TypedDictMeta, b: type) -> bool:
    """
    Test if two typeddicts have the same name, fields and types, and parameters.

    Internal helper used for test assertions.
    """
    if a.__name__ != b.__name__:
        return False
    if a.__annotations__.keys() != b.__annotations__.keys():
        return False
    for key in a.__annotations__.keys():
        if a.__annotations__[key] != b.__annotations__[key]:
            return False
    if a.__closed__ != b.__closed__:
        return False
    return True


def td_diff(a: typing_extensions._TypedDictMeta, b: type) -> str:
    """
    Generate a human-readable diff between two TypedDicts.

    This function compares two TypedDict types and produces a detailed
    summary of the differences, including:
    - Different names
    - Missing or extra fields
    - Type mismatches for fields
    - Different closed parameter values

    Parameters
    ----------
    a : typing_extensions._TypedDictMeta
        First TypedDict to compare
    b : type
        Second TypedDict to compare

    Returns
    -------
    str
        A human-readable description of the differences. Returns empty string
        if the types are equal.

    Examples
    --------
    >>> from typing_extensions import TypedDict
    >>> A = TypedDict('User', {'name': str, 'age': int})
    >>> B = TypedDict('User', {'name': str, 'age': str})
    >>> print(td_diff(A, B))
    TypedDict differences:
      Field 'age': int != str
    """
    differences = []

    # Check names
    if a.__name__ != b.__name__:
        differences.append(f"Different names: '{a.__name__}' != '{b.__name__}'")

    # Get annotations
    a_annots = a.__annotations__
    b_annots = b.__annotations__

    # Check for missing/extra fields
    a_keys = set(a_annots.keys())
    b_keys = set(b_annots.keys())

    missing_in_b = a_keys - b_keys
    extra_in_b = b_keys - a_keys
    common_keys = a_keys & b_keys

    if missing_in_b:
        for key in sorted(missing_in_b):
            differences.append(
                f"Field '{key}' missing in {b.__name__} (expected: {a_annots[key]})"
            )

    if extra_in_b:
        for key in sorted(extra_in_b):
            differences.append(f"Extra field '{key}' in {b.__name__}: {b_annots[key]}")

    # Check type mismatches for common fields
    for key in sorted(common_keys):
        a_type = a_annots[key]
        b_type = b_annots[key]
        if a_type != b_type:
            # Format the types nicely
            a_type_str = _format_type(a_type)
            b_type_str = _format_type(b_type)
            differences.append(f"Field '{key}': {a_type_str} != {b_type_str}")

    # Check closed parameter
    if a.__closed__ != b.__closed__:
        differences.append(
            f"Different 'closed' parameter: {a.__closed__} != {b.__closed__}"
        )

    if not differences:
        return ""

    return "TypedDict differences:\n  " + "\n  ".join(differences)


def _format_type(tp: type) -> str:
    """
    Format a type annotation as a readable string.

    Internal helper for td_diff.
    """
    # Handle common typing constructs
    if hasattr(tp, "__origin__"):
        origin = get_origin(tp)
        args = get_args(tp)

        # Special handling for common types
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))

            # Handle NotRequired, Required, etc.
            if hasattr(tp, "__class__"):
                class_name = tp.__class__.__name__
                if "NotRequired" in class_name or "Required" in class_name:
                    inner = _format_type(args[0]) if args else "..."
                    return f"{class_name}[{inner}]"

            if args:
                args_str = ", ".join(_format_type(arg) for arg in args)
                return f"{origin_name}[{args_str}]"
            return origin_name

    # Handle regular types
    if hasattr(tp, "__name__"):
        return tp.__name__

    # Fallback to string representation
    return str(tp)
