import copy
import datetime
import json
from pathlib import PurePath

import numpy as np
import polars as pl
import pytest

from exptoolkit.data._datamodel import (
    JSONDict,
    JSONList,
    JSONSerializationWarning,
    _normalize_json_value,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (1, 1),
        (1.5, 1.5),
        ("abc", "abc"),
    ],
)
def test_normalize_json_scalar(value, expected):
    assert _normalize_json_value(value) == expected


def test_normalize_json_container():
    value = _normalize_json_value({"a": [1, {"b": (2, 3)}]})

    assert isinstance(value, JSONDict)
    assert isinstance(value["a"], JSONList)
    assert isinstance(value["a"][1], JSONDict)
    assert isinstance(value["a"][1]["b"], JSONList)
    assert value.to_builtin() == {"a": [1, {"b": [2, 3]}]}


def test_normalize_existing_json_container_returns_same_object():
    d = JSONDict({"a": 1})
    l = JSONList([1, 2])

    assert _normalize_json_value(d) is d
    assert _normalize_json_value(l) is l


@pytest.mark.parametrize(
    ("value", "expected", "expected_type"),
    [
        (np.int64(1), 1, int),
        (np.float64(1.5), 1.5, float),
        (np.bool_(True), True, bool),
        (np.str_("abc"), "abc", str),
    ],
)
def test_normalize_numpy_scalar_without_warning(value, expected, expected_type):
    result = _normalize_json_value(value)

    assert result == expected
    assert type(result) is expected_type


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (PurePath("a", "b"), str(PurePath("a", "b"))),
        (datetime.date(2026, 8, 29), "2026-08-29"),
        (datetime.time(12, 34, 56), "12:34:56"),
    ],
)
def test_normalize_convertible_value_with_warning(value, expected):
    with pytest.warns(JSONSerializationWarning):
        result = _normalize_json_value(value)

    assert result == expected


def test_normalize_numpy_array_with_warning():
    with pytest.warns(JSONSerializationWarning):
        result = _normalize_json_value(np.array([[1, 2], [3, 4]]))

    assert isinstance(result, JSONList)
    assert result.to_builtin() == [[1, 2], [3, 4]]


def test_normalize_polars_series_with_warning():
    with pytest.warns(JSONSerializationWarning):
        result = _normalize_json_value(pl.Series("x", [1, 2]))

    assert isinstance(result, JSONList)
    assert result.to_builtin() == [1, 2]


def test_normalize_polars_dataframe_with_warning():
    with pytest.warns(JSONSerializationWarning):
        result = _normalize_json_value(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    assert isinstance(result, JSONDict)
    assert result.to_builtin() == {"a": [1, 2], "b": ["x", "y"]}


def test_normalize_rejects_unsupported_value():
    with pytest.raises(TypeError, match="not supported for JSON serialization"):
        _normalize_json_value(object())


def test_jsondict_requires_string_key():
    with pytest.raises(TypeError, match="key must be a string"):
        JSONDict({1: "a"})  # type: ignore[dict-item]


def test_jsondict_rejects_invalid_nested_key():
    with pytest.raises(TypeError, match="key must be a string"):
        JSONDict({"outer": {1: "a"}})  # type: ignore[dict-item]


def test_jsondict_normalizes_on_assignment():
    d = JSONDict()
    d["nested"] = {"values": (1, 2)}

    assert isinstance(d["nested"], JSONDict)
    assert isinstance(d["nested"]["values"], JSONList)
    assert d["nested"]["values"][1] == 2


def test_jsondict_rejects_unsupported_assignment():
    d = JSONDict()

    with pytest.raises(TypeError):
        d["bad"] = object()

    assert "bad" not in d


def test_jsondict_mapping_operations():
    d = JSONDict({"a": 1, "b": 2})

    assert len(d) == 2
    assert list(d) == ["a", "b"]
    assert d["a"] == 1

    del d["a"]
    assert d.to_builtin() == {"b": 2}


def test_jsonlist_normalizes_on_mutation():
    values = JSONList([1])

    values.append({"a": [2, 3]})
    values.insert(1, (4, 5))
    values[0] = {"b": 6}

    assert isinstance(values[0], JSONDict)
    assert isinstance(values[1], JSONList)
    assert isinstance(values[2], JSONDict)
    assert values.to_builtin() == [{"b": 6}, [4, 5], {"a": [2, 3]}]


def test_jsonlist_rejects_unsupported_mutation():
    values = JSONList([1])

    with pytest.raises(TypeError):
        values.append(object())

    assert values.to_builtin() == [1]


def test_jsonlist_sequence_operations():
    values = JSONList([1, 2, 3])

    assert len(values) == 3
    assert values[1] == 2

    del values[1]
    assert values == [1, 3]


def test_to_builtin_returns_plain_recursive_structure():
    value = JSONDict({"a": JSONList([{"b": 1}])})

    result = value.to_builtin()

    assert result == {"a": [{"b": 1}]}
    assert type(result) is dict
    assert type(result["a"]) is list
    assert type(result["a"][0]) is dict


def test_to_builtin_result_is_json_serializable():
    value = JSONDict({"a": [1, True, None, {"b": "x"}]})

    encoded = json.dumps(value.to_builtin())

    assert json.loads(encoded) == {"a": [1, True, None, {"b": "x"}]}


def test_repr_looks_like_builtin_container():
    d = JSONDict({"a": [1, 2]})
    l = JSONList([1, {"a": 2}])

    assert repr(d) == repr({"a": [1, 2]})
    assert repr(l) == repr([1, {"a": 2}])


def test_rich_returns_builtin_container():
    d = JSONDict({"a": [1, 2]})
    l = JSONList([1, {"a": 2}])

    assert d.__rich__() == {"a": [1, 2]}
    assert l.__rich__() == [1, {"a": 2}]


def test_jsondict_shallow_copy_shares_nested_container():
    original = JSONDict({"nested": {"a": 1}})
    copied = copy.copy(original)

    assert copied is not original
    assert copied["nested"] is original["nested"]

    copied["nested"]["a"] = 2
    assert original["nested"]["a"] == 2


def test_jsondict_deepcopy_copies_nested_container():
    original = JSONDict({"nested": {"a": 1}})
    copied = copy.deepcopy(original)

    assert copied is not original
    assert copied["nested"] is not original["nested"]

    copied["nested"]["a"] = 2
    assert original["nested"]["a"] == 1


def test_jsonlist_shallow_copy_shares_nested_container():
    original = JSONList([{"a": 1}])
    copied = copy.copy(original)

    assert copied is not original
    assert copied[0] is original[0]

    copied[0]["a"] = 2
    assert original[0]["a"] == 2


def test_jsonlist_deepcopy_copies_nested_container():
    original = JSONList([{"a": 1}])
    copied = copy.deepcopy(original)

    assert copied is not original
    assert copied[0] is not original[0]

    copied[0]["a"] = 2
    assert original[0]["a"] == 1


def test_normalize_upath():
    upath = pytest.importorskip("upath")

    value = upath.UPath("memory://foo/bar.txt")

    with pytest.warns(JSONSerializationWarning):
        result = _normalize_json_value(value)

    assert result == str(value)
    assert type(result) is str
