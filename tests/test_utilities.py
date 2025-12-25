"""Tests for TypeScript-style utility types."""

from typing import Union
from typing_extensions import NotRequired, Required, TypedDict
import pytest

from tytr.utilities import (
    awaited,
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
from tytr._internal import td_eq


class User(TypedDict):
    name: str
    age: int
    email: str


class PartialUser(TypedDict):
    name: NotRequired[str]
    age: NotRequired[int]


class MixedRequiredness(TypedDict):
    required_field: Required[str]
    optional_field: NotRequired[int]


def test_partial():
    """Test that partial makes all fields NotRequired."""
    result = partial(User)

    expected = TypedDict(
        "PartialUser",
        {
            "name": NotRequired[str],
            "age": NotRequired[int],
            "email": NotRequired[str],
        },
    )

    assert td_eq(result, expected)


def test_partial_with_name():
    """Test that partial respects custom names."""
    result = partial(User, name="CustomPartial")
    assert result.__name__ == "CustomPartial"


def test_partial_already_optional():
    """Test that partial handles already NotRequired fields."""
    result = partial(PartialUser)

    expected = TypedDict(
        "PartialPartialUser",
        {
            "name": NotRequired[str],
            "age": NotRequired[int],
        },
    )

    assert td_eq(result, expected)


def test_required():
    """Test that required makes all fields Required."""
    result = required(PartialUser)

    expected = TypedDict(
        "RequiredPartialUser",
        {
            "name": Required[str],
            "age": Required[int],
        },
    )

    assert td_eq(result, expected)


def test_required_with_name():
    """Test that required respects custom names."""
    result = required(PartialUser, name="CustomRequired")
    assert result.__name__ == "CustomRequired"


def test_required_mixed():
    """Test that required handles mixed Required/NotRequired fields."""
    result = required(MixedRequiredness)

    expected = TypedDict(
        "RequiredMixedRequiredness",
        {
            "required_field": Required[str],
            "optional_field": Required[int],
        },
    )

    assert td_eq(result, expected)


def test_readonly():
    """Test that readonly wraps all fields with ReadOnly."""
    from typing_extensions import ReadOnly

    result = readonly(User)

    expected = TypedDict(
        "ReadonlyUser",
        {
            "name": ReadOnly[str],
            "age": ReadOnly[int],
            "email": ReadOnly[str],
        },
    )
    assert td_eq(result, expected)
    assert result.__name__ == "ReadonlyUser"


def test_readonly_preserves_required():
    """Test that readonly preserves Required/NotRequired wrappers."""
    from typing_extensions import ReadOnly

    result = readonly(MixedRequiredness)

    expected = TypedDict(
        "ReadonlyMixedRequiredness",
        {
            "required_field": Required[ReadOnly[str]],
            "optional_field": NotRequired[ReadOnly[int]],
        },
    )
    assert td_eq(result, expected)


def test_readonly_idempotent():
    """Test that readonly on already readonly type is idempotent."""
    from typing_extensions import ReadOnly

    first = readonly(User)
    second = readonly(first, name="ReadonlyUser")

    expected = TypedDict(
        "ReadonlyUser",
        {
            "name": ReadOnly[str],
            "age": ReadOnly[int],
            "email": ReadOnly[str],
        },
    )
    assert td_eq(second, expected)


def test_pick():
    """Test that pick selects only specified fields."""
    result = pick(User, ("name", "age"))

    expected = TypedDict(
        "PickUser",
        {
            "name": str,
            "age": int,
        },
    )

    assert td_eq(result, expected)


def test_pick_single_field():
    """Test that pick works with a single field."""
    result = pick(User, ("name",))

    expected = TypedDict(
        "PickUser",
        {
            "name": str,
        },
    )

    assert td_eq(result, expected)


def test_pick_missing_key():
    """Test that pick raises error for missing keys."""
    with pytest.raises(ValueError, match="Keys {'missing'} not found"):
        pick(User, ("name", "missing"))


def test_omit():
    """Test that omit removes specified fields."""
    result = omit(User, ("email",))

    expected = TypedDict(
        "OmitUser",
        {
            "name": str,
            "age": int,
        },
    )

    assert td_eq(result, expected)


def test_omit_multiple_fields():
    """Test that omit works with multiple fields."""
    result = omit(User, ("age", "email"))

    expected = TypedDict(
        "OmitUser",
        {
            "name": str,
        },
    )

    assert td_eq(result, expected)


def test_exclude():
    """Test that exclude removes types from a Union."""
    union_type = Union[str, int, bool]
    result = exclude(union_type, bool)

    assert result == Union[str, int]


def test_exclude_single_type():
    """Test that exclude works on non-union types."""
    result = exclude(str, int)
    assert result == str


def test_exclude_all():
    """Test that exclude raises error when excluding all types."""
    # Exclude one type at a time to remove both
    result = exclude(Union[str, int], str)
    assert result == int

    # Now test excluding the only remaining type
    with pytest.raises(ValueError, match="Cannot exclude"):
        exclude(int, int)


def test_extract():
    """Test that extract keeps only specified types from a Union."""
    union_type = Union[str, int, bool]
    result = extract(union_type, str)

    assert result == str


def test_extract_not_found():
    """Test that extract raises error when type not found."""
    union_type = Union[str, int]
    with pytest.raises(ValueError, match="Type .* not found"):
        extract(union_type, bool)


def test_non_nullable():
    """Test that non_nullable removes None from Union."""
    nullable_type = Union[str, None]
    result = non_nullable(nullable_type)

    assert result == str


def test_non_nullable_union():
    """Test that non_nullable works with multiple types."""
    nullable_type = Union[str, int, None]
    result = non_nullable(nullable_type)

    assert result == Union[str, int]


def test_non_nullable_not_nullable():
    """Test that non_nullable returns non-nullable types as-is."""
    result = non_nullable(str)
    assert result == str


def test_non_nullable_only_none():
    """Test that non_nullable raises error for None-only type."""
    with pytest.raises(ValueError, match="Cannot make None non-nullable"):
        non_nullable(type(None))


def test_parameters():
    """Test that parameters extracts function parameter types."""

    def greet(name: str, age: int) -> str:
        return f"Hello {name}, age {age}"

    result = parameters(greet)
    assert result == (str, int)


def test_parameters_no_annotations():
    """Test that parameters handles unannotated parameters."""
    from typing import Any

    def func(x, y):
        return x + y

    result = parameters(func)
    assert result == (Any, Any)


def test_return_type():
    """Test that return_type extracts function return type."""

    def greet(name: str) -> str:
        return f"Hello {name}"

    result = return_type(greet)
    assert result == str


def test_return_type_no_annotation():
    """Test that return_type handles unannotated return."""
    from typing import Any

    def func(x):
        return x

    result = return_type(func)
    assert result == Any


def test_awaited_coroutine():
    """Test that awaited unwraps Coroutine types."""
    from collections.abc import Coroutine
    from typing import Any

    coro_type = Coroutine[Any, Any, str]
    result = awaited(coro_type)
    assert result == str


def test_awaited_awaitable():
    """Test that awaited unwraps Awaitable types."""
    from collections.abc import Awaitable

    awaitable_type = Awaitable[int]
    result = awaited(awaitable_type)
    assert result == int


def test_awaited_non_async():
    """Test that awaited returns non-async types as-is."""
    result = awaited(str)
    assert result == str
