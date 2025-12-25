# /// script
# requires-python = "==3.12"
# dependencies = [
#   "tytr@file:///Users/d-v-b/dev/tytr",
#   "pydantic==2.12",
#   "pytest==9.0"
# ]
# ///

from pydantic import BaseModel
from typing_extensions import TypedDict, NotRequired
from tytr.testing import make_type_test
import pytest


class MyModel(BaseModel):
    a: int
    b: str
    c: float


class MyModelDict(TypedDict):
    a: int
    b: str
    c: int  # wrong type - should be float


class MyModelPartialDict(TypedDict):
    a: NotRequired[int]
    b: NotRequired[int]  # wrong type - should be str
    c: NotRequired[float]


# Generate test functions using the helper
test_dict_type = make_type_test(MyModel, MyModelDict)
test_partial_dict_type = make_type_test(MyModel, MyModelPartialDict, transform="partial")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
