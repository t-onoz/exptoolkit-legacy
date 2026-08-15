from __future__ import annotations

import contextlib
import os
import tempfile
import time
import typing as t
from pathlib import Path


@t.overload
def atomic_open(
    path: str | os.PathLike,
    mode: t.Literal["w"] = "w",
    **kwargs,
) -> contextlib.AbstractContextManager[t.IO[str]]: ...


@t.overload
def atomic_open(
    path: str | os.PathLike,
    mode: t.Literal["wb"] = "wb",
    **kwargs,
) -> contextlib.AbstractContextManager[t.IO[bytes]]: ...


@contextlib.contextmanager
def atomic_open(
    path: str | os.PathLike,
    mode: t.Literal["w", "wb"] = "w",
    **kwargs,
) -> t.Generator[t.IO]:
    """Open a file for atomic writing.

    The file is first written to a temporary file in the same directory.
    On successful exit, the temporary file atomically replaces the target.
    If an exception occurs, the temporary file is deleted.
    """
    if "r" in mode or "+" in mode:
        raise ValueError("atomic_open() only supports write modes.")

    path = Path(path)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        with open(tmp_path, mode, **kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())

        for i in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except OSError:
                if i == 4:
                    raise
                time.sleep(0.5)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
