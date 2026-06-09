import pytest

from exptoolkit.data._datamodel import JSONDict, JSONList, normalize_json_value


# -------------------------
# normalize_json_value
# -------------------------

def test_normalize_none():
    assert normalize_json_value(None) is None


def test_normalize_scalar():
    assert normalize_json_value(1) == 1
    assert normalize_json_value(1.5) == 1.5
    assert normalize_json_value(True) is True
    assert normalize_json_value("abc") == "abc"


def test_normalize_list_tuple():
    v = normalize_json_value([1, (2, 3)])
    assert isinstance(v, JSONList)
    assert isinstance(v[0], int)
    assert isinstance(v[1], JSONList)
    assert v[1][0] == 2


def test_normalize_dict():
    v = normalize_json_value({"a": 1, "b": [2, 3]})
    assert isinstance(v, JSONDict)
    assert v["a"] == 1
    assert isinstance(v["b"], JSONList)
    assert v["b"][0] == 2


def test_normalize_nested_json_objects():
    d = JSONDict({"x": JSONList([1, 2])})
    v = normalize_json_value(d)
    assert isinstance(v, JSONDict)
    assert isinstance(v["x"], JSONList)


def test_normalize_invalid_key():
    with pytest.raises(TypeError):
        normalize_json_value({1: "a"})


def test_normalize_unsupported():
    with pytest.raises(TypeError):
        normalize_json_value(object())


# -------------------------
# JSONDict
# -------------------------

def test_jsondict_set_get():
    d = JSONDict()
    d["a"] = 1
    assert d["a"] == 1


def test_jsondict_init():
    d = JSONDict({"a": 1, "b": [2, 3]})
    assert d["a"] == 1
    assert isinstance(d["b"], JSONList)


def test_jsondict_nested():
    d = JSONDict()
    d["x"] = {"y": [1, 2]}
    assert isinstance(d["x"], JSONDict)
    assert isinstance(d["x"]["y"], JSONList)


def test_jsondict_del():
    d = JSONDict({"a": 1})
    del d["a"]
    assert "a" not in d


def test_jsondict_iter_len():
    d = JSONDict({"a": 1, "b": 2})
    assert len(d) == 2
    assert set(iter(d)) == {"a", "b"}


# -------------------------
# JSONList
# -------------------------

def test_jsonlist_append_via_init():
    l = JSONList([1, 2, 3])
    assert l[0] == 1
    assert len(l) == 3


def test_jsonlist_setitem():
    l = JSONList([1, 2, 3])
    l[1] = [4, 5]
    assert isinstance(l[1], JSONList)
    assert l[1][0] == 4


def test_jsonlist_insert():
    l = JSONList([1, 3])
    l.insert(1, 2)
    assert l[1] == 2


def test_jsonlist_del():
    l = JSONList([1, 2, 3])
    del l[1]
    assert l[1] == 3


# -------------------------
# to_builtin
# -------------------------

def test_to_builtin_dict():
    d = JSONDict({"a": 1, "b": [2, 3]})
    out = d.to_dict()
    assert isinstance(out, dict)
    assert out == {"a": 1, "b": [2, 3]}


def test_to_builtin_list():
    l = JSONList([1, {"a": 2}])
    out = l.to_list()
    assert isinstance(out, list)
    assert out == [1, {"a": 2}]


def test_to_builtin_nested():
    d = JSONDict({"a": JSONList([{"b": 1}])})
    out = d.to_dict()
    assert out == {"a": [{"b": 1}]}


# -------------------------
# repr
# -------------------------

def test_repr_dict():
    d = JSONDict({"a": 1})
    assert "JSONDict" in repr(d)


def test_repr_list():
    l = JSONList([1, 2])
    assert "JSONList" in repr(l)
