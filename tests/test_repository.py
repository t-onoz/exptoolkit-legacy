# pylint: disable=W
import re
from tempfile import NamedTemporaryFile

import polars as pl
import pytest

from exptoolkit.repository import ResourceRepo  # type: ignore[import]


def test_add_and_lookup():
    repo = ResourceRepo()
    dr1 = repo.add("file1", measurement_id="m1", samples="s1", data_type="csv")
    dr2 = repo.add("file2", measurement_id="m1", samples=["s1", "s2"], data_type="txt")
    repo._check_indexes()

    # Measurement lookup
    m1 = repo.by_measurement("m1")
    assert dr1 in m1
    assert dr2 in m1
    assert len(m1) == 2

    # Sample lookup
    s1_set = repo.by_sample("s1")
    assert dr1 in s1_set and dr2 in s1_set
    s2_set = repo.by_sample("s2")
    assert dr2 in s2_set
    assert dr1 not in s2_set

    # Samples of DataSource
    assert "s1" in repo.samples_of(dr1)
    assert "s2" not in repo.samples_of(dr1)
    assert "s1" in repo.samples_of(dr2)
    assert "s2" in repo.samples_of(dr2)


def test_remove():
    repo = ResourceRepo()
    dr = repo.add("file1", measurement_id="m1", samples="s1")

    repo._check_indexes()
    assert dr in repo.by_sample("s1")
    assert dr in repo.by_measurement("m1")
    assert dr in repo.iter_resources()

    repo.remove("file1")

    repo._check_indexes()
    assert dr not in repo.by_sample("s1")
    assert dr not in repo.by_measurement("m1")
    assert dr not in repo.iter_resources()


def test_move_resource():
    repo = ResourceRepo()
    repo.add("file1", measurement_id="m1", samples=["s1", "s2"], data_type="csv")
    repo.move_resource("file1", "file_new")
    repo._check_indexes()

    # old ref gone
    assert "file1" not in repo._ref2d

    # new ref exists
    dr_new = repo._ref2d["file_new"]
    assert dr_new.ref == "file_new"
    assert "s1" in repo.samples_of(dr_new)
    assert "s2" in repo.samples_of(dr_new)
    assert dr_new in repo.by_sample("s1")
    assert dr_new in repo.by_measurement("m1")


def test_find_predicate():
    repo = ResourceRepo()
    dr1 = repo.add("file1", measurement_id="m1", samples="s1", data_type="csv")
    dr2 = repo.add("file2", measurement_id="m1", samples="s1", data_type="txt")
    repo._check_indexes()

    # Find by type
    csv_set = [dr for dr in repo.iter_resources() if dr.type_ == "csv"]
    assert dr1 in csv_set
    assert dr2 not in csv_set


def test_empty_lookup():
    repo = ResourceRepo()
    assert not repo.by_sample("nope")
    assert not repo.by_measurement("none")


def test_multiple_measurements():
    repo = ResourceRepo()
    dr1 = repo.add("file1", measurement_id="m1", samples="s1")
    dr2 = repo.add("file2", measurement_id="m2", samples="s1")
    repo._check_indexes()

    # by sample
    s1_set = repo.by_sample("s1")
    assert dr1 in s1_set and dr2 in s1_set

    # by measurement
    m1_set = repo.by_measurement("m1")
    assert dr1 in m1_set
    assert dr2 not in m1_set


def test_save_and_load():
    repo = ResourceRepo()
    dr1 = repo.add("file1", measurement_id="m1", samples="s1")
    dr2 = repo.add("file2", measurement_id="m2", samples="s1")
    repo._check_indexes()

    with NamedTemporaryFile(mode="w+") as tmp:
        repo.save(tmp)
        tmp.seek(0)
        repo2 = ResourceRepo.load(tmp)

    repo2._check_indexes()
    assert dr1 in repo2.iter_resources()
    assert dr2 in repo2.iter_resources()
    assert dr1 in repo2.by_measurement("m1")
    assert dr2 not in repo2.by_measurement("m1")
    assert dr1 not in repo2.by_measurement("m2")
    assert dr2 in repo2.by_measurement("m2")
    assert dr1 in repo2.by_sample("s1")
    assert dr2 in repo2.by_sample("s1")


def test_by_sample_regex_basic():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1", "x1"])
    repo.add("b.csv", measurement_id="m1", samples=["s2"])
    repo.add("c.csv", measurement_id="m2", samples=["x2"])

    result = repo.by_sample_regex(r"^s")

    assert set(result.keys()) == {"s1", "s2"}
    assert {dr.ref for dr in result["s1"]} == {"a.csv"}
    assert {dr.ref for dr in result["s2"]} == {"b.csv"}


def test_by_sample_regex_multiple_hits():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1", "s2"])

    result = repo.by_sample_regex(r"^s")

    assert set(result.keys()) == {"s1", "s2"}
    assert all(dr.ref == "a.csv" for drs in result.values() for dr in drs)


def test_by_sample_regex_no_match():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1"])

    result = repo.by_sample_regex(r"^x")

    assert result == {}


def test_by_sample_regex_empty_repo():
    repo = ResourceRepo()

    result = repo.by_sample_regex(r".*")

    assert result == {}


def test_by_sample_regex_partial_match():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["abc", "def"])

    result = repo.by_sample_regex(r"b")

    assert set(result.keys()) == {"abc"}


def test_by_sample_regex_returns_distinct_lists():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1"])
    repo.add("b.csv", measurement_id="m1", samples=["s1"])

    result = repo.by_sample_regex(r"s1")

    refs = {dr.ref for dr in result["s1"]}
    assert refs == {"a.csv", "b.csv"}


def test_by_sample_regex_invalid_pattern():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1"])

    with pytest.raises(re.error):
        repo.by_sample_regex(r"[")  # invalid regex


def test_broken_repo():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=["s1"])
    repo.add("b.csv", measurement_id="m1", samples=["s2"])
    del repo._ref2d["a.csv"]
    with pytest.raises(AssertionError):
        repo._check_indexes()


def test_to_df_basic():
    repo = ResourceRepo()

    repo.add(ref="file1", measurement_id="m001", samples=["A", "B"], data_type="raw")

    df = repo.to_polars()

    expected = pl.DataFrame(
        [
            {"ref": "file1", "measurement_id": "m001", "data_type": "raw", "sample": "A"},
            {"ref": "file1", "measurement_id": "m001", "data_type": "raw", "sample": "B"},
        ]
    )

    assert df.sort(df.columns).equals(expected.sort(expected.columns))


def test_to_df_multiple_refs():
    repo = ResourceRepo()

    repo.add(ref="f1", measurement_id="m001", samples=["A", "B"])
    repo.add(ref="f2", measurement_id="m002", samples=["C"])

    df = repo.to_polars()
    assert set(df["ref"]) == {"f1", "f2"}
    assert df.shape[0] == 3


def test_add_empty_samples_does_not_modify_repo():
    repo = ResourceRepo()

    with pytest.raises(ValueError, match="samples must not be empty"):
        repo.add(
            ref="file1",
            measurement_id="M001",
            samples=[],
        )

    repo._check_indexes()

    assert len(repo._ref2d) == 0


def test_add_empty_samples_keeps_existing_repo():
    repo = ResourceRepo()

    repo.add(
        ref="file0",
        measurement_id="M000",
        samples="sample0",
    )

    with pytest.raises(ValueError):
        repo.add(
            ref="file1",
            measurement_id="M001",
            samples=[],
        )

    repo._check_indexes()
    assert "file1" not in repo._ref2d


def test_add_rejects_empty_sample_iterable():
    repo = ResourceRepo()

    with pytest.raises(ValueError, match="samples must not be empty"):
        repo.add(
            "a.csv",
            measurement_id="m1",
            samples=iter(()),
        )

    repo._check_indexes()

    assert len(repo._ref2d) == 0
