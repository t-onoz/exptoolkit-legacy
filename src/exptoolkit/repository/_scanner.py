from __future__ import annotations
import os
import time
import typing as t
import json
import re
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from pathlib import Path
from logging import getLogger
from pydantic import BaseModel, ConfigDict

from exptoolkit.repository._repo import ResourceRepo
from exptoolkit.repository._utils import atomic_open

logger = getLogger(__name__)


class ScanResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ref: str
    measurement_id: str
    samples: tuple[str, ...]
    data_type: t.Union[str, None] = None


class ResourceScanner(ABC):
    """
    Scan external resources and sync them to a ResourceRepo.

    Each scanner owns a subset of refs.
    `scan()` must return ALL resources in that subset.
    """

    @abstractmethod
    def owns(self, ref: str) -> bool:
        """
        Return True if this scanner is responsible for the ref.
        """

    @abstractmethod
    def scan(self) -> list[ScanResult]:
        """
        Return a complete list of resources owned by this scanner.

        Rules:
            - All refs must satisfy owns(ref) == True
            - Must be complete (no missing refs)
        """

    def scan_and_sync(self, repo: ResourceRepo) -> None:
        """
        Sync repo with current scan result.

        - Add new resources
        - Remove missing resources (only owned refs)
        """
        results = self.scan()
        new_refs = {r.ref: r for r in results}

        # add or update
        for r in results:
            repo.add(
                r.ref,
                measurement_id=r.measurement_id,
                samples=r.samples,
                data_type=r.data_type,
            )

        # remove (owned only)
        for ref in list(repo._ref2d.keys()):
            if self.owns(ref) and ref not in new_refs:
                repo.remove(ref)

class _CacheEntry(BaseModel):
    mtime: float
    scanned_at: float
    results: list[ScanResult]

class _CacheData(BaseModel):
    version: str
    root: Path
    dir_regex: re.Pattern
    file_regex: re.Pattern
    entries: dict[str, _CacheEntry]

class DirectoryScanner(ResourceScanner):
    """
    Example scanner for a directory with structure:

        root/
            [measurement1]/
                [sample1]
                [sample2]
            [measurement2]/
                ...

    - `dir_regex` and `file_regex` control which folders/files are scanned.
    - measurement_id is obtained via `f_mid([measurement folder])`
    - sample_name is obtained via `f_sample([data file])`
    - data_type is obtained via `f_type([data file])`
    - ref = absolute path of file

    Uses per-measurement cache based on folder mtime.
    """
    _version = '1.0.0'

    def __init__(self,
                 root: str | os.PathLike,
                 *,
                 dir_regex: str | re.Pattern = '.*',
                 file_regex: str | re.Pattern = '.*',
                 f_mid: t.Callable[[os.DirEntry], str] = lambda e: e.name,
                 f_sample: t.Callable[[os.DirEntry], str] = lambda e: os.path.splitext(e.name)[0],
                 f_type: t.Callable[[os.DirEntry], str | None] = lambda e: os.path.splitext(e.name)[1][1:] or None,
                 ):
        # Ensure Windows short (8.3) temp paths are resolved to a canonical path by os.path.realpath()
        self.root = Path(os.path.realpath(root))
        self._cache: dict[str, _CacheEntry] = {}
        self.dir_regex = re.compile(dir_regex)
        self.file_regex = re.compile(file_regex)
        self.f_mid = f_mid
        self.f_sample = f_sample
        self.f_type = f_type

    # --- ownership ---
    def owns(self, ref: str) -> bool:
        pr = urlparse(ref)
        if len(pr.scheme) >=2:
            # exclude http://, file://, etc.
            return False

        p = Path(ref)
        try:
            p.relative_to(self.root)
            return True
        except ValueError:
            return False

    # --- main scan ---
    def scan(self) -> list[ScanResult]:
        results: list[ScanResult] = []

        with os.scandir(self.root) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if not self.dir_regex.match(entry.name):
                    continue

                measurement_id = self.f_mid(entry)

                # NOTE:
                # entry.stat() is fast but may return stale mtime (Windows / FS cache).
                # This can occur even without concurrent writes (see CPython issue #85278).
                # Calling os.stat(path) seems to refresh it; use that if strict correctness is needed.
                mtime = entry.stat().st_mtime

                # check cache
                cache = self._cache.get(measurement_id)
                if cache and abs(cache.mtime - mtime) < 1e-3:
                    logger.debug('read measurement %r from cache', measurement_id)
                    results.extend(cache.results)
                    continue

                logger.debug('read measurement %r from disk', measurement_id)

                # scan files
                scanned = self._scan_measurement_dir(entry.path, measurement_id)

                # update cache
                self._cache[measurement_id] = _CacheEntry(
                    mtime=mtime,
                    scanned_at=time.time(),
                    results=scanned,
                )

                results.extend(scanned)

        return results

    # --- internal ---
    def _scan_measurement_dir(self, d: str, measurement_id: str) -> list[ScanResult]:
        results: list[ScanResult] = []

        for f in os.scandir(d):
            if not f.is_file():
                continue

            if not self.file_regex.match(f.name):
                continue

            sample_name = self.f_sample(f)
            data_type = self.f_type(f)

            ref = f.path

            results.append(
                ScanResult(
                    ref=ref,
                    measurement_id=measurement_id,
                    samples=(sample_name,),
                    data_type=data_type,
                )
            )

        return results

    def _dump_cache(self) -> _CacheData:
        return _CacheData(
            version=self._version,
            root=self.root,
            dir_regex=self.dir_regex,
            file_regex=self.file_regex,
            entries={mid: entry for mid, entry in self._cache.items()}
        )

    def save_cache(self, file: str | os.PathLike | t.IO[str], **json_kw) -> None:
        """Save the cache as a JSON file."""
        data = self._dump_cache().model_dump(mode='json')
        json_kw.setdefault('indent', 2)
        json_kw.setdefault('ensure_ascii', False)
        if isinstance(file, (str, os.PathLike)):
            with atomic_open(file, "w", encoding='utf-8') as f:
                json.dump(data, f, **json_kw)
        else:
            json.dump(data, file, **json_kw)

    def load_cache(
        self,
        file: str | os.PathLike | t.IO[str],
        strict: bool = False,
    ) -> None:
        """Load cache from a JSON file.

        If strict is False, invalid cache files are ignored.
        """
        try:
            if isinstance(file, (str, os.PathLike)):
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.load(file)

            cache = _CacheData.model_validate(data)

            if (
                cache.version != self._version
                or cache.root != self.root
                or cache.dir_regex != self.dir_regex
                or cache.file_regex != self.file_regex
            ):
                raise ValueError("Cache metadata mismatch.")

            self._cache = dict(cache.entries)

        except Exception:
            if strict:
                raise
            logger.warning("Failed to load cache: %s", file, exc_info=True)
