import json
from zipfile import ZIP_STORED, ZipFile

import polars as pl
import pytest

from exptoolkit.data import BaseData, Column


class SampleData(BaseData):
    voltage = Column(pl.Float64)
    current = Column(pl.Float64)


def test_save_load_roundtrip(tmp_path):
    data = SampleData(
        {
            "voltage": [3.0, 3.1],
            "current": [1.0, 2.0],
        },
        normalization=(2.0, "g"),
        metadata={
            "sample": "A",
            "nested": {"foo": [1, 2, 3]},
        },
    )

    path = tmp_path / "data.zip"
    data.save(path)

    loaded = SampleData.load(path)

    assert loaded.table.equals(data.table)
    assert loaded.norm == data.norm
    assert loaded.metadata.to_builtin() == data.metadata.to_builtin()


def test_save_writes_manifest(tmp_path):
    data = SampleData(
        {
            "voltage": [3.0],
            "current": [1.0],
        },
        metadata={"sample": "A"},
    )

    path = tmp_path / "data.zip"
    data.save(path)

    with ZipFile(path, "r") as zip_file:
        manifest = json.loads(zip_file.read("manifest.json"))

    assert manifest["format"] == "exptoolkit"
    assert manifest["version"] == 1
    assert manifest["class"] == "SampleData"
    assert manifest["metadata"] == {"sample": "A"}
    assert manifest["norm"] == [None, None]


def test_load_without_version_treated_as_version_1(tmp_path):
    path = tmp_path / "data.zip"

    manifest = {
        "format": "exptoolkit",
        "class": "SampleData",
        "metadata": {"sample": "A"},
        "norm": [None, None],
    }

    table = pl.DataFrame(
        {
            "voltage": [3.0],
            "current": [1.0],
        },
        schema={
            "voltage": pl.Float64,
            "current": pl.Float64,
        },
    )

    with ZipFile(path, "w", compression=ZIP_STORED) as zip_file:
        with zip_file.open("table.parquet", "w") as f:
            table.write_parquet(f)

        zip_file.writestr("manifest.json", json.dumps(manifest))

    loaded = SampleData.load(path)

    assert loaded.table.equals(table)
    assert loaded.metadata["sample"] == "A"


def test_load_rejects_unsupported_version(tmp_path):
    path = tmp_path / "data.zip"

    manifest = {
        "format": "exptoolkit",
        "version": 999,
        "class": "SampleData",
        "metadata": {},
        "norm": [None, None],
    }

    table = pl.DataFrame(
        {
            "voltage": [3.0],
            "current": [1.0],
        },
        schema={
            "voltage": pl.Float64,
            "current": pl.Float64,
        },
    )

    with ZipFile(path, "w", compression=ZIP_STORED) as zip_file:
        with zip_file.open("table.parquet", "w") as f:
            table.write_parquet(f)

        zip_file.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported file version"):
        SampleData.load(path)


def test_load_rejects_wrong_format(tmp_path):
    path = tmp_path / "data.zip"

    manifest = {
        "format": "unknown",
        "version": 1,
        "class": "SampleData",
        "metadata": {},
        "norm": [None, None],
    }

    with ZipFile(path, "w", compression=ZIP_STORED) as zip_file:
        zip_file.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported file format"):
        SampleData.load(path)


def test_load_warns_on_class_mismatch(tmp_path):
    class OtherData(SampleData):
        pass

    data = SampleData(
        {
            "voltage": [3.0],
            "current": [1.0],
        }
    )

    path = tmp_path / "data.zip"
    data.save(path)

    with pytest.warns(UserWarning, match="File was created as SampleData"):
        loaded = OtherData.load(path)

    assert loaded.table.equals(data.table)
