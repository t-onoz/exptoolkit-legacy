import polars as pl
import pytest

from exptoolkit.data import BaseData, Column


def test_schema_order():
    class A(BaseData):
        x = Column(pl.Float64)

    class B(A):
        y = Column(pl.Int64)

    assert list(B.schema.keys()) == ["x", "y"]


def test_missing_column_filled():
    class A(BaseData):
        x = Column(pl.Float64)

    a = A({})
    assert a.table["x"].null_count() == 1


def test_drop_extra(sample_df):
    from conftest import SampleData

    df = {**sample_df, "extra": [1, 2, 3]}
    d = SampleData(df, drop_extra_columns=True)

    assert "extra" not in d.table.columns

    d = SampleData(df, drop_extra_columns=False)

    assert "extra" in d.table.columns


def test_schema_creation():
    class Data(BaseData):
        a = Column(pl.Int64)
        b = Column(pl.Float64)

    assert list(Data.schema) == ["a", "b"]


def test_schema_inheritance():
    class A(BaseData):
        a = Column(pl.Int64)

    class B(A):
        b = Column(pl.Float64)

    assert list(B.schema) == ["a", "b"]


def test_column_override():
    class A(BaseData):
        x = Column(pl.Int64)

    class B(A):
        x = Column(pl.Float64)

    assert list(B.schema) == ["x"]
    assert B.schema["x"].dtype == pl.Float64


def test_column_cannot_override_inherited_non_column():
    with pytest.raises(ValueError, match="conflicts with an inherited attribute"):

        class Data(BaseData):
            save = Column(pl.Float64)  # type: ignore[assignment]


def test_column_cannot_override_annotated_non_column():
    with pytest.raises(ValueError, match="conflicts with an inherited attribute"):

        class Data(BaseData):
            norm = Column(pl.Float64)  # type: ignore[assignment]


def test_non_column_cannot_override_inherited_column():
    class A(BaseData):
        x = Column(pl.Float64)

    with pytest.raises(ValueError, match="cannot be overridden by a non-Column"):

        class B(A):
            x = "not a column"  # type: ignore[assignment]
