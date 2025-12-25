"""Tests for core module (flatten, to_typeddict, key_of, value_of)."""

from typing import get_args, Generic, TypeVar, List, Dict, Optional
from typing_extensions import TypedDict, Required, NotRequired, Literal
from pydantic import BaseModel
import pytest

from tytr.core import flatten, to_typeddict, key_of, value_of, DelimiterCollisionError
from tytr._internal import td_eq


# ============================================================================
# Tests for flatten()
# ============================================================================


@pytest.mark.parametrize("delimiter", [None, ".", "/"])
def test_delimeter(delimiter: str | None) -> None:
    """
    Test that the keys of the generated typeddict use the specified delimiter.
    """

    class Child:
        x: float
        y: float

    class MyClass:
        a: int
        b: str
        c: Child

    kwargs: dict[str, str] = {}
    if delimiter is not None:
        kwargs["key_delimiter"] = delimiter
    result = flatten(MyClass, **kwargs)
    expected_fields = {
        "a": int,
        "b": str,
        "c": Child,
        f"c{delimiter or '_'}x": float,
        f"c{delimiter or '_'}y": float,
    }
    expected = TypedDict("MyClass", expected_fields)

    assert td_eq(result, expected)


def test_pydantic() -> None:
    class ChildModel(BaseModel):
        x: float
        y: float

    class MyModel(BaseModel):
        a: int
        b: str
        c: ChildModel

    result = flatten(MyModel)

    expected = TypedDict(
        "MyModel",
        {
            "a": int,
            "b": str,
            "c": ChildModel,
            "c_x": float,
            "c_y": float,
        },
    )

    assert td_eq(result, expected)


def test_requirement() -> None:
    """
    Test that the Required and NotRequired annotations are propagated correctly.
    """

    class C(TypedDict):
        d: Required[float]
        e: NotRequired[bool]

    class X(TypedDict):
        a: Required[int]
        b: NotRequired[str]
        c: C

    class Expected(TypedDict):
        a: Required[int]
        b: NotRequired[str]
        c: C
        c_d: Required[float]
        c_e: NotRequired[bool]

    observed = flatten(X, key_delimiter="_", name="Expected")

    assert td_eq(observed, Expected)


def test_delimiter_collision() -> None:
    """
    Test that a delimiter collision raises an appropriate error.
    """

    class Child:
        x_y: float

    class MyClass:
        a: int
        b: str
        c: Child

    with pytest.raises(
        DelimiterCollisionError,
        match="The delimiter '_' occurs in the field name 'x_y' on the class Child",
    ):
        flatten(MyClass, key_delimiter="_")


def test_generic() -> None:
    """
    Test that we properly generate generic typeddicts
    """
    T = TypeVar("T")
    V = TypeVar("V")

    class WithV(TypedDict, Generic[V]):
        value: V

    class WithG(TypedDict, Generic[T]):
        value: T
        child: WithV[int]

    class Expected(TypedDict, Generic[T]):
        value: T
        child: WithV[int]
        child_value: int

    observed = flatten(WithG, key_delimiter="_", name="Expected")
    assert td_eq(observed, Expected)


# ============================================================================
# Tests for to_typeddict()
# ============================================================================


def test_simple_class():
    """Test converting a simple class to TypedDict."""

    class User:
        name: str
        age: int
        email: str

    result = to_typeddict(User)

    expected = TypedDict(
        "User",
        {
            "name": str,
            "age": int,
            "email": str,
        },
    )

    assert td_eq(result, expected)
    assert result.__name__ == "User"


def test_with_custom_name():
    """Test that custom name is respected."""

    class User:
        name: str
        age: int

    result = to_typeddict(User, name="UserDict")
    assert result.__name__ == "UserDict"


def test_pydantic_model():
    """Test converting a Pydantic model to TypedDict."""

    class Product(BaseModel):
        id: int
        title: str
        price: float

    result = to_typeddict(Product)

    expected = TypedDict(
        "Product",
        {
            "id": int,
            "title": str,
            "price": float,
        },
    )

    assert td_eq(result, expected)


def test_preserves_required_notrequired():
    """Test that Required/NotRequired annotations are preserved."""

    class PartialUser(TypedDict):
        name: Required[str]
        age: NotRequired[int]

    result = to_typeddict(PartialUser)

    expected = TypedDict(
        "PartialUser",
        {
            "name": Required[str],
            "age": NotRequired[int],
        },
    )

    assert td_eq(result, expected)


def test_nested_not_flattened():
    """Test that nested structures are NOT flattened."""

    class Address:
        street: str
        city: str

    class Person:
        name: str
        address: Address

    result = to_typeddict(Person)

    expected = TypedDict(
        "Person",
        {
            "name": str,
            "address": Address,
        },
    )

    assert td_eq(result, expected)

    # Verify that nested fields are NOT in the result
    assert "address_street" not in result.__annotations__
    assert "address_city" not in result.__annotations__


def test_closed_parameter():
    """Test that closed parameter is passed through."""

    class User:
        name: str
        age: int

    # Note: We can't easily test the closed attribute at runtime,
    # but we can verify the function accepts it
    result = to_typeddict(User, closed=True)
    assert result.__name__ == "User"

    # The closed attribute should be set
    assert getattr(result, "__closed__", None) is True


def test_empty_class():
    """Test converting a class with no annotations."""

    class Empty:
        pass

    result = to_typeddict(Empty)

    expected = TypedDict("Empty", {})
    assert td_eq(result, expected)


def test_complex_types():
    """Test that complex type annotations are preserved."""

    class ComplexTypes:
        items: List[str]
        mapping: Dict[str, int]
        optional: Optional[float]

    result = to_typeddict(ComplexTypes)

    expected = TypedDict(
        "ComplexTypes",
        {
            "items": List[str],
            "mapping": Dict[str, int],
            "optional": Optional[float],
        },
    )

    assert td_eq(result, expected)


def test_comparison_with_flatten():
    """Test to show the difference between to_typeddict and flatten."""

    class Address:
        street: str
        city: str

    class Person:
        name: str
        address: Address

    # to_typeddict preserves nesting
    simple = to_typeddict(Person)
    assert "address" in simple.__annotations__
    assert "address_street" not in simple.__annotations__

    # flatten expands nesting
    flat = flatten(Person)
    assert "address" in flat.__annotations__
    assert "address_street" in flat.__annotations__
    assert "address_city" in flat.__annotations__


# ============================================================================
# Tests for key_of()
# ============================================================================


def test_key_of_typeddict() -> None:
    """Test key_of with a TypedDict."""

    class Person(TypedDict):
        name: str
        age: int
        email: str

    result = key_of(Person)
    # Should be Literal['name', 'age', 'email']
    assert get_args(result) == ("name", "age", "email")


def test_key_of_single_key_typeddict() -> None:
    """Test key_of with a single-key TypedDict."""

    class Single(TypedDict):
        value: int

    result = key_of(Single)
    # Should be Literal['value']
    assert get_args(result) == ("value",)


def test_key_of_dict_type() -> None:
    """Test key_of with a generic dict type."""
    result = key_of(dict[str, int])
    assert result is str


def test_key_of_dict_with_int_keys() -> None:
    """Test key_of with integer keys."""
    result = key_of(dict[int, str])
    assert result is int


def test_key_of_nested_typeddict() -> None:
    """Test key_of with nested TypedDict (should only return top-level keys)."""

    class Address(TypedDict):
        street: str
        city: str

    class Person(TypedDict):
        name: str
        address: Address

    result = key_of(Person)
    args = get_args(result)
    assert "name" in args
    assert "address" in args
    assert len(args) == 2
    # Should NOT include 'street' or 'city'


# ============================================================================
# Tests for value_of()
# ============================================================================


def test_value_of_typeddict() -> None:
    """Test value_of with a TypedDict."""

    class Person(TypedDict):
        name: str
        age: int
        active: bool

    result = value_of(Person)
    # Should be Union[str, int, bool]
    args = get_args(result)
    assert str in args
    assert int in args
    assert bool in args
    assert len(args) == 3


def test_value_of_single_value_typeddict() -> None:
    """Test value_of with a single value type TypedDict."""

    class AllStrings(TypedDict):
        first: str
        second: str
        third: str

    result = value_of(AllStrings)
    # Should be just str (not Union[str, str, str])
    # Actually, it will be Union but we can check the type
    # The function returns str if all are the same type
    # Let's check what we actually get
    # Based on implementation, it returns Union[tuple(value_types)]
    # which would be Union[str, str, str] but Union collapses duplicates
    args = get_args(result)
    # Union of same types collapses to a single type in some Python versions
    # So we should just verify str is in there
    assert str in args or result is str


def test_value_of_dict_type() -> None:
    """Test value_of with a generic dict type."""
    result = value_of(dict[str, int])
    assert result is int


def test_value_of_dict_with_multiple_types() -> None:
    """Test value_of with a dict that has union values."""
    result = value_of(dict[str, int | str])
    # The value type is int | str (Union[int, str])
    args = get_args(result)
    assert int in args
    assert str in args


def test_value_of_nested_typeddict() -> None:
    """Test value_of with nested TypedDict."""

    class Address(TypedDict):
        street: str
        city: str

    class Person(TypedDict):
        name: str
        address: Address

    result = value_of(Person)
    args = get_args(result)
    # Should be Union[str, Address]
    assert str in args
    assert Address in args
