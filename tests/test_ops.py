import polars as pl

def test_filter(data):
    b = data.filter(pl.col("x") > 1)
    assert b.x.to_list() == [2.0, 3.0]


def test_downsample(data):
    b = data.downsample(2)
    assert b.x.to_list() == [1.0, 3.0]


def test_with_table_copy(data):
    b = data.with_table(data.table)

    b.metadata["new"] = 1
    assert "new" not in data.metadata

def test_with_table_nocopy(data):
    b = data.with_table(data.table, copy_metadata=False)

    b.metadata["new"] = 1
    assert "new" in data.metadata
