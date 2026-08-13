from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol, TypeVar, runtime_checkable

import polars as pl

from exptoolkit.data import BaseData

M_contra = TypeVar("M_contra", bound=BaseData, contravariant=True)
M_co = TypeVar("M_co", bound=BaseData, covariant=True)
M = TypeVar("M", bound=BaseData)


@runtime_checkable
class Modifier(Protocol[M_contra]):
    """Base modifier class which takes one data and modifies in place."""

    def __call__(self, data: M_contra, *a, **kw) -> None: ...


@runtime_checkable
class Converter(Protocol[M_contra, M_co]):
    """Base converter class which takes one data and returns another."""

    def __call__(self, data: M_contra, *a, **kw) -> M_co: ...


@runtime_checkable
class Combiner(Protocol[M_contra, M_co]):
    """Base combiner class which takes multiple data and combines them into one object."""

    def __call__(self, data_list: Iterable[M_contra], *a, **kw) -> M_co: ...


# callable dataclass
@dataclass
class Downsampler(Converter[BaseData, BaseData]):
    """example data converter. similar to data.downsample()."""

    n: int
    offset: int = 0

    def __call__(self, data: M) -> M:
        return data.with_table(
            table=data.table.gather_every(self.n, self.offset), copy_metadata=True
        )


# usual function
def downsample(data: M, n: int, offset: int = 0) -> M:
    return data.with_table(
        table=data.table.gather_every(n, offset),
    )


def concatenate(data_list: Iterable[M]) -> M:
    data_list = list(data_list)
    if not data_list:
        raise ValueError("No data to concatenate")
    norms = [d.norm for d in data_list]
    if not all(n == norms[0] for n in norms):
        raise ValueError("All data must have the same normalization parameters.")
    return data_list[0].with_table(pl.concat([d.table for d in data_list], how="diagonal"))


if TYPE_CHECKING:
    _downsample: Converter = downsample
    _concatenate: Combiner = concatenate
