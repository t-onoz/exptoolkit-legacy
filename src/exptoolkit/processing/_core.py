from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from exptoolkit.data import BaseData

M_contra = TypeVar("M_contra", bound=BaseData, contravariant=True)
M_co = TypeVar("M_co", bound=BaseData, covariant=True)


class Modifier(Protocol[M_contra]):
    """Callable that modifies one data object in place.

    A modifier can be an ordinary function::

        def remove_invalid_rows(data: MyData) -> None:
            data.table = data.table.filter(...)

    or any other callable object with the same interface.
    """

    def __call__(self, data: M_contra, *args, **kwargs) -> None: ...


class Converter(Protocol[M_contra, M_co]):
    """Callable that converts one data object into another.

    A converter can be an ordinary function::

        def convert(data: RawData) -> ProcessedData:
            return ProcessedData(...)

    or a configured callable object::

        @dataclass
        class ConverterWithOptions:
            scale: float

            def __call__(self, data: RawData) -> ProcessedData:
                ...
    """

    def __call__(self, data: M_contra, *args, **kwargs) -> M_co: ...


class Combiner(Protocol[M_contra, M_co]):
    """Callable that combines multiple data objects into one.

    For example::

        def concatenate(data_list: Iterable[MyData]) -> MyData:
            ...
    """

    def __call__(
        self,
        data_list: Iterable[M_contra],
        *args,
        **kwargs,
    ) -> M_co: ...


@dataclass(frozen=True)
class Featurizer(ABC, Generic[M_contra]):
    """Base class for fixed-schema feature transformations.

    A featurizer converts one experimental data object into a one-dimensional
    numeric feature vector. ``feature_names`` defines the corresponding
    feature schema.

    Featurizer instances are immutable because both the feature names and the
    size of the returned vector may depend on hyperparameters. Keeping those
    parameters fixed ensures that the feature schema remains stable for the
    lifetime of the instance.

    A minimal implementation looks like::

        @dataclass(frozen=True)
        class MeanFeaturizer(Featurizer[MyData]):
            columns: tuple[str, ...]

            @cached_property
            def feature_names(self) -> tuple[str, ...]:
                return tuple(f"{column}_mean" for column in self.columns)

            def _featurize(self, data: MyData) -> Iterable[float]:
                return (
                    data.table[column].mean()
                    for column in self.columns
                )

    Missing or unavailable feature values should generally be represented by
    NaN rather than raising an exception. Structural violations of the
    featurizer contract, such as returning the wrong number of features, are
    detected by ``__call__``.
    """

    @cached_property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Names of features in the order returned by this featurizer."""
        ...

    def __call__(self, data: M_contra) -> NDArray[np.float64]:
        """Convert data into a validated one-dimensional feature vector."""
        values = np.asarray(tuple(self._featurize(data)), dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"{type(self).__name__} returned a {values.ndim}-D array; "
                "a 1-D feature vector is required."
            )

        if values.size != len(self.feature_names):
            raise ValueError(
                f"{type(self).__name__} returned {values.size} features, "
                f"but feature_names contains {len(self.feature_names)} names."
            )

        return values

    @abstractmethod
    def _featurize(self, data: M_contra) -> Iterable[float]:
        """Compute features for one data object."""
        ...
