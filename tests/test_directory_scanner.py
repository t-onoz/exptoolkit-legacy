# test_directory_scanner.py
from __future__ import annotations
import os
import time
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from exptoolkit.repository import ResourceRepo, ScanResult, DirectoryScanner

@pytest.fixture
def tmp_structure():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # measurement folder
        (root / 'm001').mkdir()
        (root / 'm002').mkdir()
        # sample files
        (root / 'm001' / 'sample1.csv').write_text('1,2,3')
        (root / 'm001' / 'sample2.csv').write_text('4,5,6')
        (root / 'm002' / 'sample3.csv').write_text('7,8,9')
        # refresh mtime
        os.stat(root / 'm001')
        os.stat(root / 'm002')
        yield root

def test_owns(tmp_structure):
    scanner = DirectoryScanner(tmp_structure)

    # relative path
    assert scanner.owns(str(tmp_structure / 'm001' / 'sample1.csv'))
    # sub folder
    assert scanner.owns(str(tmp_structure / 'm002'))

    assert not scanner.owns(str((tmp_structure / '..').resolve()))
    assert not scanner.owns('D:/some/path')
    assert not scanner.owns((tmp_structure / 'm001').resolve().as_uri())
    assert not scanner.owns('file:///C:/some/path')
    assert not scanner.owns('http://example.com/data.csv')

def test_scan_basic(tmp_structure):
    scanner = DirectoryScanner(tmp_structure)
    repo = ResourceRepo()

    results = scanner.scan()
    assert all(isinstance(r, ScanResult) for r in results)
    assert len(results) == 3
    # sample names
    names = [r.samples[0] for r in results]
    assert set(names) == {'sample1', 'sample2', 'sample3'}

    # scan_and_sync
    scanner.scan_and_sync(repo)
    repo_samples = [s for r in repo.iter_resources() for s in repo.samples_of(r)]
    assert set(repo_samples) == {'sample1', 'sample2', 'sample3'}

def test_cache_behavior(tmp_structure):
    scanner = DirectoryScanner(tmp_structure)
    repo = ResourceRepo()

    # create cache
    scanner.scan_and_sync(repo)
    mtime_before = {mid: entry.mtime for mid, entry in scanner._cache.items()}

    # rescan
    scanner.scan_and_sync(repo)
    mtime_after = {mid: entry.mtime for mid, entry in scanner._cache.items()}
    assert mtime_before == mtime_after

    # add a file → check cache
    (tmp_structure / 'm002' / 'sample4.csv').write_text('10,11,12')
    time.sleep(0.5) # wait for file creation
    os.stat(tmp_structure / 'm002')  # refresh mtime of a folder
    time.sleep(0.5)
    scanner.scan_and_sync(repo)
    # new file in repo
    names = [s for r in repo.iter_resources() for s in repo.samples_of(r)]
    assert 'sample4' in names

def test_file_regex(tmp_structure):
    scanner = DirectoryScanner(tmp_structure, file_regex=r'sample1\.csv')
    results = scanner.scan()
    names = [r.samples[0] for r in results]
    assert names == ['sample1']

def test_dir_regex(tmp_structure):
    scanner = DirectoryScanner(tmp_structure, dir_regex=r'm002')
    results = scanner.scan()
    mids = {r.measurement_id for r in results}
    assert mids == {'m002'}

def test_save_load_cache(tmp_structure, tmp_path):
    scanner = DirectoryScanner(tmp_structure)
    scanner.scan()

    cache_file = tmp_path / 'cache.json'
    scanner.save_cache(cache_file)

    new_scanner = DirectoryScanner(tmp_structure)
    new_scanner.load_cache(cache_file)

    assert set(new_scanner._cache.keys()) == set(scanner._cache.keys())
    for mid in scanner._cache:
        assert len(new_scanner._cache[mid].results) == len(scanner._cache[mid].results)
