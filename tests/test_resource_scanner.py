from exptoolkit.repository import ResourceRepo, ResourceScanner, ScanResult


class DummyScanner(ResourceScanner):
    def __init__(self, results, owns_fn=lambda ref: True):
        self._results = results
        self._owns = owns_fn

    def scan(self):
        return self._results

    def owns(self, ref: str) -> bool:
        return self._owns(ref)


def test_scan_and_sync_add_only():
    repo = ResourceRepo()

    scanner = DummyScanner(
        [
            ScanResult(ref="a.csv", measurement_id="m1", samples=("s1",)),
            ScanResult(ref="b.csv", measurement_id="m2", samples=("s2",)),
        ]
    )

    scanner.scan_and_sync(repo)

    assert set(repo._ref2d.keys()) == {"a.csv", "b.csv"}


def test_scan_and_sync_remove():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=("s1",))
    repo.add("b.csv", measurement_id="m2", samples=("s2",))

    scanner = DummyScanner(
        [
            ScanResult(ref="a.csv", measurement_id="m1", samples=("s1",)),
        ]
    )

    scanner.scan_and_sync(repo)

    assert set(repo._ref2d.keys()) == {"a.csv"}


def test_scan_and_sync_no_touch_outside_ownership():
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples="s1")
    repo.add("b.zip::inner.csv", measurement_id="m2", samples="s2")

    # owns only .csv (not zip)
    scanner = DummyScanner(
        [ScanResult(ref="a.csv", measurement_id="m1", samples=("s1",))],
        owns_fn=lambda ref: ref.endswith(".csv") and "::" not in ref,
    )

    scanner.scan_and_sync(repo)

    # zip resource should remain
    assert set(repo._ref2d.keys()) == {"a.csv", "b.zip::inner.csv"}


def test_scan_and_sync_add_and_remove_mix():
    repo = ResourceRepo()
    repo.add("old.csv", measurement_id="m1", samples=("s1",))

    scanner = DummyScanner(
        [
            ScanResult(ref="new.csv", measurement_id="m2", samples=("s2",)),
        ]
    )

    scanner.scan_and_sync(repo)

    assert set(repo._ref2d.keys()) == {"new.csv"}


def test_scan_must_be_complete_contract():
    """
    If scan is incomplete, owned resources will be deleted.
    This test documents that behavior (not a failure).
    """
    repo = ResourceRepo()
    repo.add("a.csv", measurement_id="m1", samples=("s1",))

    # scan returns nothing → treated as deletion
    scanner = DummyScanner([])

    scanner.scan_and_sync(repo)

    assert repo._ref2d == {}


def test_scan_and_sync_update_existing():
    repo = ResourceRepo()
    repo.add(
        "a.csv",
        measurement_id="m1",
        samples=("old_sample",),
        data_type="old_type",
    )

    scanner = DummyScanner(
        [
            ScanResult(
                ref="a.csv",
                measurement_id="m2",
                samples=("new_sample",),
                data_type="new_type",
            )
        ]
    )

    scanner.scan_and_sync(repo)

    resource = repo._ref2d["a.csv"]

    assert resource.type_ == "new_type"
    assert repo.measurement_of("a.csv") == "m2"
    assert set(repo.samples_of("a.csv")) == {"new_sample"}

    assert repo.by_sample("old_sample") == []
    assert repo.by_sample("new_sample") == [resource]
    assert repo.by_measurement("m1") == []
    assert repo.by_measurement("m2") == [resource]

    repo._check_indexes()
