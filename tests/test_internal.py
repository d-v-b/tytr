"""Tests for internal utility functions (_internal module)."""

import pytest
from typing_extensions import TypedDict

from tytr._internal import td_eq, td_repr


@pytest.mark.parametrize("closed", [None, True, False])
def test_td_eq(closed: bool | None) -> None:
    """
    Test that td_eq correctly tests for structural equality
    between typeddict types.
    """
    kwargs: dict[str, object] = {}

    if closed is not None:
        kwargs["closed"] = closed

    # Create two identical TypedDicts
    TD1 = TypedDict("MyDict", {"x": int, "y": str}, **kwargs)
    TD2 = TypedDict("MyDict", {"x": int, "y": str}, **kwargs)

    # They should be equal
    assert td_eq(TD1, TD2)


def test_td_eq_different_names() -> None:
    """Test that td_eq returns False for TypedDicts with different names."""
    TD1 = TypedDict("Dict1", {"x": int})
    TD2 = TypedDict("Dict2", {"x": int})

    assert not td_eq(TD1, TD2)


def test_td_eq_different_fields() -> None:
    """Test that td_eq returns False for TypedDicts with different fields."""
    TD1 = TypedDict("MyDict", {"x": int, "y": str})
    TD2 = TypedDict("MyDict", {"x": int, "z": str})

    assert not td_eq(TD1, TD2)


def test_td_eq_different_types() -> None:
    """Test that td_eq returns False for TypedDicts with different field types."""
    TD1 = TypedDict("MyDict", {"x": int})
    TD2 = TypedDict("MyDict", {"x": str})

    assert not td_eq(TD1, TD2)


def test_td_eq_different_closed() -> None:
    """Test that td_eq returns False for TypedDicts with different closed values."""
    TD1 = TypedDict("MyDict", {"x": int}, closed=True)
    TD2 = TypedDict("MyDict", {"x": int}, closed=False)

    assert not td_eq(TD1, TD2)


def test_td_repr() -> None:
    """Test that td_repr produces a readable string representation."""
    TD = TypedDict("MyDict", {"x": int, "y": str}, closed=True)

    result = td_repr(TD)

    # Check that the repr contains the expected information
    assert "TypedDict MyDict(" in result
    assert "x: " in result
    assert "y: " in result
    assert "closed=True" in result


def test_td_repr_with_closed_false() -> None:
    """Test td_repr with closed=False."""
    TD = TypedDict("MyDict", {"x": int}, closed=False)

    result = td_repr(TD)

    assert "TypedDict MyDict(" in result
    assert "closed=False" in result


def test_td_diff_equal_types() -> None:
    """Test that td_diff returns empty string for equal types."""
    from tytr._internal import td_diff

    TD1 = TypedDict("MyDict", {"x": int, "y": str})
    TD2 = TypedDict("MyDict", {"x": int, "y": str})

    result = td_diff(TD1, TD2)
    assert result == ""


def test_td_diff_different_field_types() -> None:
    """Test td_diff with different field types."""
    from tytr._internal import td_diff

    TD1 = TypedDict("MyDict", {"x": int, "y": str})
    TD2 = TypedDict("MyDict", {"x": int, "y": int})

    result = td_diff(TD1, TD2)
    assert "Field 'y': str != int" in result


def test_td_diff_missing_field() -> None:
    """Test td_diff with missing field."""
    from tytr._internal import td_diff

    TD1 = TypedDict("MyDict", {"x": int, "y": str})
    TD2 = TypedDict("MyDict", {"x": int})

    result = td_diff(TD1, TD2)
    assert "Field 'y' missing in MyDict" in result


def test_td_diff_extra_field() -> None:
    """Test td_diff with extra field."""
    from tytr._internal import td_diff

    TD1 = TypedDict("MyDict", {"x": int})
    TD2 = TypedDict("MyDict", {"x": int, "z": str})

    result = td_diff(TD1, TD2)
    assert "Extra field 'z' in MyDict" in result


def test_assert_type_equals_success() -> None:
    """Test assert_type_equals with equal types."""
    from tytr._internal import assert_type_equals

    TD1 = TypedDict("MyDict", {"x": int})
    TD2 = TypedDict("MyDict", {"x": int})

    # Should not raise
    assert_type_equals(TD1, TD2)


def test_assert_type_equals_failure() -> None:
    """Test assert_type_equals with different types."""
    from tytr._internal import assert_type_equals

    TD1 = TypedDict("MyDict", {"x": int})
    TD2 = TypedDict("MyDict", {"x": str})

    with pytest.raises(AssertionError) as exc_info:
        assert_type_equals(TD1, TD2)

    assert "Field 'x': int != str" in str(exc_info.value)


def test_make_type_test_basic() -> None:
    """Test make_type_test with basic to_typeddict."""
    from tytr._internal import make_type_test

    class User:
        name: str
        age: int

    class UserDict(TypedDict):
        name: str
        age: int

    test_func = make_type_test(User, UserDict)
    assert callable(test_func)
    assert "test_User_transforms_to_UserDict" in test_func.__name__

    # Should not raise
    test_func()


def test_make_type_test_with_transform() -> None:
    """Test make_type_test with a transformation."""
    from tytr._internal import make_type_test
    from typing_extensions import NotRequired

    class User:
        name: str
        age: int

    class UserPartial(TypedDict):
        name: NotRequired[str]
        age: NotRequired[int]

    test_func = make_type_test(User, UserPartial, transform='partial')
    assert callable(test_func)

    # Should not raise
    test_func()


def test_make_type_test_failure() -> None:
    """Test that make_type_test generates a function that fails appropriately."""
    from tytr._internal import make_type_test

    class User:
        name: str
        age: int

    class WrongUserDict(TypedDict):
        name: str
        age: str  # Wrong type

    test_func = make_type_test(User, WrongUserDict)

    with pytest.raises(AssertionError) as exc_info:
        test_func()

    assert "Field 'age': int != str" in str(exc_info.value)
