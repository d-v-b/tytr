from typing import Callable

from tytr._internal import td_diff, td_eq


def assert_type_equals(actual: type, expected: type, *, msg: str | None = None) -> None:
    """
    Assert that two TypedDict types are equal, with helpful error messages.

    This function is designed for use in tests to validate type transformations.
    When types don't match, it provides a detailed diff showing the differences.

    Parameters
    ----------
    actual : type
        The actual TypedDict type (typically the result of a transformation)
    expected : type
        The expected TypedDict type
    msg : str, optional
        Optional custom message to prepend to the diff

    Raises
    ------
    AssertionError
        If the types are not equal, with a detailed diff in the message

    Examples
    --------
    >>> from typing_extensions import TypedDict
    >>> from tytr import to_typeddict, partial
    >>> from pydantic import BaseModel
    >>>
    >>> class MyModel(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> class MyModelDict(TypedDict):
    ...     name: str
    ...     age: int
    >>>
    >>> assert_type_equals(to_typeddict(MyModel), MyModelDict)
    """
    if not td_eq(actual, expected):
        diff = td_diff(actual, expected)
        error_msg = diff if msg is None else f"{msg}\n{diff}"
        raise AssertionError(error_msg)


def make_type_test(
    base_type: type,
    expected_type: type,
    transform: str | Callable[[type], type] | None = None,
    *,
    test_name: str | None = None,
) -> Callable[[], None]:
    """
    Create a test function that validates a type transformation.

    This is a test helper that generates a pytest-compatible test function
    to validate that a type transformation produces the expected result.

    Parameters
    ----------
    base_type : type
        The base type to transform (e.g., a Pydantic model or class)
    expected_type : type
        The expected TypedDict after transformation
    transform : str, Callable, or None
        The transformation to apply. Can be:
        - None: just convert to TypedDict with to_typeddict()
        - str: name of a tytr function (e.g., 'partial', 'required', 'readonly')
        - Callable: custom transformation function
    test_name : str, optional
        Custom name for the test function (defaults to auto-generated name)

    Returns
    -------
    Callable[[], None]
        A test function that can be run by pytest

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from typing_extensions import TypedDict, NotRequired
    >>>
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> class UserPartial(TypedDict):
    ...     name: NotRequired[str]
    ...     age: NotRequired[int]
    >>>
    >>> # Generate a test function
    >>> test_user_partial = make_type_test(User, UserPartial, transform='partial')
    >>>
    >>> # Run it (pytest will call this)
    >>> test_user_partial()

    Notes
    -----
    The transform always starts with to_typeddict() to convert the base type
    to a TypedDict, then applies the specified transformation if any.
    """
    # Import here to avoid circular imports
    from tytr.core import to_typeddict
    import tytr.utilities as utilities

    # Determine the transformation function
    if transform is None:
        # Just to_typeddict with name matching
        def transform_func(cls: type) -> type:
            return to_typeddict(cls, name=expected_type.__name__)

        transform_name = "to_typeddict"
    elif isinstance(transform, str):
        # Look up the function by name
        if not hasattr(utilities, transform):
            raise ValueError(f"Unknown transformation: {transform}")
        transform_name = transform
        base_transform = getattr(utilities, transform)

        # Chain: to_typeddict first, then the transformation
        def transform_func(cls: type) -> type:
            # Need to match the expected type's name
            base_td = to_typeddict(cls)
            result = base_transform(base_td, name=expected_type.__name__)
            return result
    elif callable(transform):
        # Custom transformation
        transform_func = transform
        transform_name = getattr(transform, "__name__", "custom")
    else:
        raise TypeError(
            f"transform must be None, str, or callable, got {type(transform)}"
        )

    # Generate test function name
    if test_name is None:
        base_name = base_type.__name__
        expected_name = expected_type.__name__
        test_name = f"test_{base_name}_transforms_to_{expected_name}"

    # Create the test function
    def test_func() -> None:
        actual = transform_func(base_type)
        assert_type_equals(
            actual,
            expected_type,
            msg=f"Type transformation {transform_name}({base_type.__name__}) failed",
        )

    # Set the function name for pytest
    test_func.__name__ = test_name
    return test_func
