from __future__ import annotations
import re
import typing as t
import json
import os
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class DataResource:
    """
    Identifies an external resource as a string. `ref` can point to local files, URLs,
    or files inside archives (nested paths via "::" like "inner/path::archive.zip").
    The Resource itself holds no authentication or state; access and interpretation
    must be handled separately.

    `type_` represents additional information about a resource ("raw", "csv", etc.).
    """
    ref: str
    type_: str | None = None

MeasurementID = t.NewType("MeasurementID", str)

class ResourceRepo:
    """
    Repository managing data sources and their association with measurements and samples.

    Responsibilities:
        - Add/remove DataResource with a MeasurementID and sample names.
        - Lookup resources by sample (exact or regex) or by measurement.
        - Provide Measurement views combining resources and samples.
        - Maintain internal bidirectional indices for fast queries.

    Notes:
        - Each DataResource belongs to exactly one MeasurementID (many-to-one).
        - Sample names can be shared across resources and measurements.
    """
    def __init__(self) -> None:
        self._ref2d: dict[str, DataResource] = {}

        # relations
        self._ref2m: dict[str, MeasurementID] = {}
        self._m2ref: dict[MeasurementID, set[str]] = {}

        # sample index
        self._sample2ref: dict[str, set[str]] = {}
        self._ref2samples: dict[str, set[str]] = {}

    def add(self, ref: str, *,
            measurement_id: str,
            samples: str | t.Iterable[str],
            data_type: str | None = None
            ) -> DataResource:
        """Register a data source and its relations.

        Args:
            ref: reference to data resource
                (local files, URLs, or files inside archives as such).
            measurement_id: Measurement identifier.
            samples: Sample name(s).
            data_type: Optional data label.

        Returns:
            The registered DataSource.
        """
        dr = DataResource(ref=ref, type_=data_type)
        mid = MeasurementID(measurement_id)

        # enforce many-to-one relation between DataResource and Measurement
        if dr.ref in self._ref2m:
            existing_mid = self._ref2m[dr.ref]
            if existing_mid != mid:
                raise ValueError(
                    f"DataResource {dr.ref} is already assigned to Measurement {existing_mid}, "
                    f"cannot reassign to {mid}."
                )
        samples = [samples] if isinstance(samples, str) else samples
        if not samples:
            raise ValueError('samples must not be empty.')

        # commit all data at once
        self._ref2d[dr.ref] = dr
        self._ref2m[dr.ref] = mid
        self._m2ref.setdefault(mid, set()).add(dr.ref)
        for s in samples:
            self._sample2ref.setdefault(s, set()).add(dr.ref)
            self._ref2samples.setdefault(dr.ref, set()).add(s)

        return dr

    def remove(self, ref: str) -> None:
        """Remove a data source and its relations.

        Args:
            ref: reference to data resource
                (local files, URLs, or files inside archives).
        """
        try:
            dr = self._ref2d.pop(ref)
        except KeyError as e:
            raise ValueError(f'{repr(ref)} does not exist in repository.') from e

        mid = self._ref2m[ref]
        samples = self._ref2samples[ref].copy()

        self._ref2m.pop(ref)
        self._m2ref[mid].discard(dr.ref)
        self._ref2samples.pop(ref)
        for sample in samples:
            self._sample2ref[sample].discard(dr.ref)

        # remove measurement and samples if no data is associated
        if not self._m2ref[mid]:
            self._m2ref.pop(mid)
        for sample in samples:
            if not self._sample2ref[sample]:
                self._sample2ref.pop(sample)

    def move_resource(self, ref_before: str, ref_after: str) -> None:
        if ref_before not in self._ref2d:
            raise ValueError(f'{repr(ref_before)} does not exist in repository.')
        if ref_after in self._ref2d:
            raise ValueError(f'Cannot move resource because {repr(ref_after)} already exists.')

        dr_before = self._ref2d[ref_before]
        mid = self._ref2m[ref_before]
        samples = self._ref2samples[ref_before]
        data_type = dr_before.type_

        self.remove(ref_before)
        self.add(ref_after, measurement_id=mid, samples=samples, data_type=data_type)

    def by_sample(self, sample: str) -> list[DataResource]:
        """Find resources by sample name (exact match, fast)."""
        return [self._ref2d[ref] for ref in self._sample2ref.get(sample, set())]

    def by_sample_regex(self, pattern: str) -> dict[str, list[DataResource]]:
        """Find resources by sample name (O(N), slower). Returns mapping from sample to resources."""
        ptn = re.compile(pattern)
        return {s: [self._ref2d[ref] for ref in ref_set]
                for s, ref_set in self._sample2ref.items()
                if ptn.search(s)}

    def by_measurement(self, id_: str) -> list[DataResource]:
        """Return data sources belonging to a measurement."""
        mid = MeasurementID(id_)
        return [self._ref2d[ref] for ref in self._m2ref.get(mid, set())]

    def samples_by_measurement(self, id_: str) -> list[str]:
        """Return unique sample names associated with a measurement."""
        mid = MeasurementID(id_)
        refs = self._m2ref.get(mid, set())
        return list({
            s for ref in refs for s in self._ref2samples[ref]
        })

    def measurement_of(self, resource: str | DataResource) -> MeasurementID:
        """Return the measurement of a data source (ref string or `DataResource` object)."""
        ref = resource.ref if isinstance(resource, DataResource) else str(resource)
        return self._ref2m[ref]

    def samples_of(self, resource: str | DataResource) -> list[str]:
        """Return samples associated with a data source.

        Args:
            resource: Data source.

        Returns:
            Associated sample names.
        """
        ref = resource.ref if isinstance(resource, DataResource) else str(resource)
        return list(self._ref2samples.get(ref, set()))

    def iter_resources(self) -> t.Iterator[DataResource]:
        """Iterate over all registered DataResource objects."""
        return iter(self._ref2d.values())

    def _check_indexes(self) -> None:
        """Verify internal index consistency.

        This method is intended for debugging and testing purposes only.
        It runs in O(N) time and should not be used in performance-critical paths.
        """
        # _ref2d vs. _ref2m vs. _ref2samples
        for ref, dr in self._ref2d.items():
            assert ref == dr.ref, f"ref {ref!r} and dr.ref {dr.ref!r} does not match"
            assert ref in self._ref2m, f"{dr!r} missing in _ref2m"
            assert ref in self._ref2samples, f"{dr!r} missing in _ref2samples"
        for ref in self._ref2m:
            assert ref in self._ref2d, f"extra ref {ref!r} in _ref2m"
        for ref in self._ref2samples:
            assert ref in self._ref2d, f"extra ref {ref!r} in _ref2samples"

        # _ref2m vs. _m2ref
        for ref, mid in self._ref2m.items():
            assert ref in self._m2ref.get(mid, set()), f"{ref!r} missing in _m2d[{mid!r}]"
        for mid, ref_set in self._m2ref.items():
            for ref in ref_set:
                assert self._ref2m.get(ref) == mid, f"{ref!r} in _m2ref[{mid!r}] but not in _ref2m"

        # _ref2samples vs. _sample2ref
        for ref, samples in self._ref2samples.items():
            for s in samples:
                assert ref in self._sample2ref.get(s, set()), f"{ref!r} missing in _sample2d[{s!r}]"
        for s, ref_set in self._sample2ref.items():
            for ref in ref_set:
                assert s in self._ref2samples.get(ref, set()), f"{s!r} missing in _ref2samples[{ref!r}]"


    def save(self, file: str | os.PathLike | t.IO[str], **json_kw) -> None:
        """Save ResourceRepo to a JSON file."""
        data: ResourceRepoData = {
            "resources": [asdict(dr) for dr in self._ref2d.values()],
            "ref2m": dict(self._ref2m),
            "ref2samples": {ref: list(samples) for ref, samples in self._ref2samples.items()},
        }
        json_kw.setdefault("indent", 2)
        json_kw.setdefault("ensure_ascii", False)
        if isinstance(file, (str, os.PathLike)):
            with open(file, "w", encoding='utf-8') as f:
                json.dump(data, f, **json_kw)
        else:
            json.dump(data, file, **json_kw)

    @classmethod
    def load(cls, file: str | os.PathLike | t.IO[str]) -> ResourceRepo:
        """Load ResourceRepo from a JSON file."""
        if isinstance(file, (str, os.PathLike)):
            with open(file, encoding='utf-8') as f:
                data: ResourceRepoData = json.load(f)
        else:
            data = json.load(file)

        repo = ResourceRepo()

        # 1. resources
        for dr_dump in data["resources"]:
            dr = DataResource(**dr_dump)
            repo._ref2d[dr.ref] = dr

        # 2. measurements
        for ref, mid_dump in data["ref2m"].items():
            dr = repo._ref2d[ref]
            mid = MeasurementID(mid_dump)
            repo._ref2m[dr.ref] = mid
            repo._m2ref.setdefault(mid, set()).add(ref)

        # 3. samples
        for ref, samples in data["ref2samples"].items():
            repo._ref2samples.setdefault(ref, set()).update(samples)
            for s in samples:
                repo._sample2ref.setdefault(s, set()).add(ref)

        return repo

    def __len__(self) -> int:
        """Number of registered resources."""
        return len(self._ref2d)

    def __contains__(self, resource: str | DataResource) -> bool:
        ref = resource.ref if isinstance(resource, DataResource) else resource
        return ref in self._ref2d

    def _iter_rows(self):
        for ref, dr in self._ref2d.items():
            for s in sorted(self._ref2samples[ref]):
                yield {
                    "ref": ref,
                    "measurement_id": self._ref2m[ref],
                    "data_type": dr.type_,
                    "sample": s,
                }

    def to_polars(self):
        import polars as pl
        return pl.DataFrame(
            self._iter_rows(),
            schema=[
                ("ref", pl.String),
                ("measurement_id", pl.String),
                ("data_type", pl.String),
                ("sample", pl.String),
            ],
        )

    def to_pandas(self):
        import pandas as pd
        df = pd.DataFrame(self._iter_rows())
        df = df.astype({
            "ref": "string",
            "measurement_id": "string",
            "data_type": "string",
            "sample": "string",
        })
        return df

class ResourceRepoData(t.TypedDict):
    """Typed structure for JSON serialization of a ResourceRepo.

    - resources: list of serialized DataResource objects.
    - ref2m: mapping from resource ref to serialized MeasurementID.
    - ref2samples: mapping from resource ref to associated sample names.
    """
    resources: list[dict[str, t.Any]]
    ref2m: dict[str, t.Any]
    ref2samples: dict[str, list[str]]
