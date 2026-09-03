from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray

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


@dataclass(frozen=True)
class Featurizer(ABC, Generic[M_contra]):
    """Convert experimental data into a fixed-schema feature vector.

    Subclasses define featurization hyperparameters as dataclass fields.

    For a given instance:

    - Hyperparameters are immutable.
    - `feature_names` is fixed.
    - The output is a one-dimensional numeric array.
    - The output size matches `len(feature_names)`.

    Different instances may have different feature schemas depending on
    their hyperparameters.
    """

    @cached_property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Names corresponding to the elements of the feature vector."""

    def __call__(self, data: M_contra) -> NDArray[np.float64]:
        """Convert data into a feature vector."""
        values = self._featurize(data)

        assert values.ndim == 1, f"Feature vector must be one-dimensional ({values.ndim})."
        assert values.size == len(self.feature_names), (
            f"Size mismatch between feature vector {values!r} (len={values.size}) "
            f"and feature names (len={len(self.feature_names)})."
        )

        return values

    @abstractmethod
    def _featurize(self, data: M_contra) -> NDArray[np.float64]:
        """Compute the feature vector."""


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


@dataclass(frozen=True)
class InterpolationFeaturizer(Featurizer[BaseData]):
    """Interpolate columns onto an evenly spaced x grid.

    The x values may be strictly increasing or decreasing. Each y column
    is interpolated onto the same grid, and the resulting values are
    concatenated in the order given by `y`.

    If x is not strictly monotonic, a warning is issued and an all-NaN
    feature vector is returned. Values outside the input x range are
    filled with NaN.
    """

    x: str
    y: tuple[str, ...]
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive.")
        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start.")
        if not self.y:
            raise ValueError("y must contain at least one column.")

    @cached_property
    def grid(self) -> NDArray[np.float64]:
        grid = np.arange(
            self.start,
            self.stop + self.step / 2,
            self.step,
            dtype=np.float64,
        )
        grid.flags.writeable = False
        return grid

    @cached_property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{column}_{x:g}" for column in self.y for x in self.grid)

    def _nan_features(self) -> NDArray[np.float64]:
        return np.full(len(self.feature_names), np.nan)

    def _feaaturize(self, data: BaseData) -> NDArray[np.float64]:
        x = np.asarray(data.table[self.x].to_numpy(), dtype=np.float64)

        if x.size < 2:
            warnings.warn(
                f"{self.x!r} contains fewer than two points; returning all NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._nan_features()

        dx = np.diff(x)

        if np.all(dx > 0):
            reverse = False
        elif np.all(dx < 0):
            reverse = True
        else:
            warnings.warn(
                f"{self.x!r} is not strictly monotonic; returning all NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._nan_features()

        if reverse:
            x = x[::-1]

        features = []

        for column in self.y:
            y = np.asarray(data.table[column].to_numpy(), dtype=np.float64)

            if reverse:
                y = y[::-1]

            features.append(
                np.interp(
                    self.grid,
                    x,
                    y,
                    left=np.nan,
                    right=np.nan,
                )
            )

        result = np.concatenate(features)

        if np.isnan(result).any():
            warnings.warn(
                "Interpolated feature vector contains NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

        return result


if TYPE_CHECKING:
    _downsample: Converter = downsample
    _concatenate: Combiner = concatenate
