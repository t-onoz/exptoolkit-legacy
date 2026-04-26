from __future__ import annotations
import enum
import typing as t
from collections.abc import Mapping, Iterable
from collections import OrderedDict
from types import MappingProxyType
from dataclasses import dataclass, field
from copy import copy
from functools import lru_cache
from logging import getLogger

import polars as pl

if t.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from polars._typing import FrameInitTypes, IntoExprColumn
    from pint import UnitRegistry

logger = getLogger()

# `import pint` takes some time, so perform lazy import
@lru_cache
def load_ureg() -> UnitRegistry:
    import pint
    return pint.UnitRegistry()

M = t.TypeVar('M', bound="BaseData")

class Role(enum.IntEnum):
    """Defines role of a Column.
    EXTENSIVE: value is proportional to the amount.
    INTENSIVE: value is independent of the amount.
    INVERSE_EXTENSIVE: value is inverse proportional to the amount."""
    EXTENSIVE = 1
    INTENSIVE = 0
    INVERSE_EXTENSIVE = -1

class ColumnSpec(t.NamedTuple):
    role: int
    dtype: type[pl.DataType] | pl.DataType
    base_unit: str

class NormPolicy(t.NamedTuple):
    amount: float = float('nan')
    unit: str | None = None

@lru_cache(maxsize=1000)
def conversion_factor(
        base_unit: str | None,
        normalize_unit: str | None,
        to_unit: str | None,
        role: int,
) -> float | int:
    """returns a unit conversion factor.
    e.g. conversion_factor('m', 's', 'mm/s') -> 1000.0"""
    ureg = load_ureg()
    base = base_unit or 'dimensionless'
    norm = normalize_unit or 'dimensionless'
    to = to_unit or 'dimensionless'
    return (ureg.Quantity(1.0, base) / ureg.Quantity(1.0, norm)**role).to(to).magnitude


@dataclass
class Column:
    dtype: type[pl.DataType] | pl.DataType
    base_unit: str = 'dimensionless'
    role: int = Role.INTENSIVE
    name: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.dtype, type):
            self.dtype = self.dtype()

    def __set_name__(self, owner: type[BaseData], name: str):
        self.name = name

    @t.overload
    def __get__(self, obj: None, owner: type[BaseData] | None) -> Column: ...

    @t.overload
    def __get__(self, obj: BaseData, owner: type[BaseData] | None) -> pl.Series: ...

    def __get__(
            self, obj: BaseData | None,
            owner: type[BaseData] | None = None
    ) -> Column | pl.Series:
        if obj is None:
            return self
        return obj.table.get_column(self.name)

    def __set__(self, obj: BaseData, value: t.Any) -> None:
        if isinstance(value, (pl.Expr, pl.Series)):
            expr = value
        elif (
            hasattr(value, "__iter__")
            and not isinstance(value, (str, bytes))
        ):
            # list, tuple, numpy.ndarray, pandas.Series, etc.
            expr = pl.Series(self.name, value)
        else:
            # scalar value
            expr = pl.lit(value)
        obj.table = obj.table.with_columns(
            expr.cast(self.dtype).alias(self.name)
        )

    def get_spec(self) -> ColumnSpec:
        return ColumnSpec(role=self.role, dtype=self.dtype, base_unit=self.base_unit)

    @property
    def expr(self) -> pl.Expr:
        """returns the expression of the column."""
        return pl.col(self.name)

class SchemaMixin:
    """mixin class for data classes with a predefined schema.
    provides a class attribute `schema` which is an ordered dict of column name and ColumnSpec."""
    schema: Mapping[str, ColumnSpec]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        schema = OrderedDict()
        for base in reversed(cls.__mro__):
            for attrname, attrval in base.__dict__.items():
                if isinstance(attrval, Column):
                    schema[attrname] = attrval.get_spec()
        cls.schema = MappingProxyType(schema)

class BaseData(SchemaMixin):
    """Base class for representing data."""
    metadata: dict[str, t.Any]
    norm: NormPolicy

    def __init__(self,
                 table: FrameInitTypes,
                 *,
                 normalization: tuple[float, str | None] \
                         = NormPolicy(float('nan'), None),
                 metadata: Mapping[str, t.Any] | None = None,
                 drop_extra_columns: bool = True
                 ):
        """
        :param table: original data.
        :param normalization: Information about normalization as a tuple of (amount, unit).
            - If no normalization is applied (default), (amount, unit) = (NaN, None).
            - If normalization is applied but the amount is unknown, the amount is set to NaN.
            - If dimensionless normalization is applied, the unit is set to 'dimensionless'.
        :param metadata: Other metadata (free-form)
        :param drop_extra: If True, drop columns not included in the schema.

        Note: Specifying normalize_unit or normalize_amount does not actually perform normalization.
              These arguments only indicate that the table data is already normalized.
              Actual normalization can be performed using the normalize() method.
        """
        norm = NormPolicy(*normalization)
        self.norm = norm
        self.metadata = dict(metadata or {})
        df = pl.DataFrame(table)
        exprs = [self._col_or_null(df, col) for col in self.schema]
        for col in self.schema:
            if col not in df.columns:
                logger.info('missing column: "%s"', col)
        if not drop_extra_columns:
            exprs = exprs + [pl.col(col) for col in df.columns if col not in self.schema]
        self.table = df.select(exprs)

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    @table.setter
    def table(self, table: pl.DataFrame):
        """sets table data. checks if table schema is consistent with self."""
        is_valid = True
        errors = ["schema does not match."]
        for key, spec in self.schema.items():
            if key not in table.columns:
                is_valid = False
                errors.append(f"- table does not contain required column '{key}'")
            elif table.schema[key] != spec.dtype:
                is_valid = False
                errors.append(f"- dtype mismatch in column '{key}'."
                              f"given: {table.schema[key]}, expected: {spec.dtype}")
        if is_valid:
            self._table = table
        else:
            raise ValueError("\n".join(errors))

    def is_col_ready(self, col: str | Column):
        if isinstance(col, Column):
            col = col.name
        return not self.table[col].is_null().all()

    def col_to_unit(self, col: str | Column, unit: str | None) -> pl.Series:
        """returns a column with its unit converted.
        if unit is None, returns the original column."""
        if isinstance(col, Column):
            col = col.name
        expr = self._to_unit_expr(col, unit)
        return self.table.select(expr).to_series()

    def df_to_units(self, **units: str) -> pl.DataFrame:
        exprs = [self._to_unit_expr(col, unit) for col, unit in units.items()]
        return self.table.with_columns(exprs)

    def get_unit(self, column: str | Column, fmt='~P') -> str:
        """gets the unit associated with the given column.
        considers current normalization information."""
        if isinstance(column, Column):
            column = column.name
        ureg = load_ureg()
        base = self.schema[column].base_unit
        role = self.schema[column].role
        norm = self.norm.unit or 'dimensionless'
        return f"{ureg.Unit(base) / ureg.Unit(norm)**role:{fmt}}"

    def downsample(self: M, n: int, offset: int=0) -> M:
        """takes every n points with offset, and returns a new data object."""
        return self.with_table(self.table.gather_every(n, offset))

    def normalize(self: M, norm_amount: float, norm_unit: str) -> M:
        if self.norm.unit is not None:
            raise ValueError("data is already normalized")
        exprs = [
            pl.col(col) / norm_amount**spec.role
                for col, spec in self.schema.items()
                if spec.role != Role.INTENSIVE  # prevents normalizing non-numeric data
        ]
        new_table = self.table.with_columns(exprs)
        new_data = self.with_table(new_table)
        new_data.norm = NormPolicy(norm_amount, norm_unit)
        return new_data

    def filter(
            self: M,
            *predicates: (
                IntoExprColumn
                | Iterable[IntoExprColumn]
                | bool
                | list[bool]
                | npt.NDArray[np.bool_]
            ),
            **constraints: t.Any,
        ) -> M:
        return self.with_table(
            self.table.filter(*predicates, **constraints)
        )

    def denormalize(self: M) -> M:
        if self.norm.unit is None:
            new_data =  copy(self)
            new_data.metadata = dict(self.metadata)
        else:
            exprs = [
                pl.col(col) * self.norm.amount**spec.role
                    for col, spec in self.schema.items()
                    if spec.role != Role.INTENSIVE
            ]
            new_data = self.with_table(
                self.table.with_columns(exprs)
            )
            new_data.metadata = dict(self.metadata)
            new_data.norm = NormPolicy(1.0, None)
        return new_data

    def with_table(self: M, table: pl.DataFrame, copy_metadata: bool=True) -> M:
        """switches table and returns a new data. copies metadata by default."""
        new_data = copy(self)
        new_data.table = table
        if copy_metadata:
            new_data.metadata = copy(new_data.metadata)
        return new_data

    def _to_unit_expr(self, col: str, unit: str | None) -> pl.Expr:
        colspec = self.schema[col]
        if unit is None:
            return pl.col(col)
        c = conversion_factor(colspec.base_unit, self.norm.unit, unit, colspec.role)
        return pl.col(col) * c

    def _col_or_null(self, df: pl.DataFrame, col: str) -> pl.Expr:
        if col in df.columns:
            return pl.col(col).cast(dtype=self.schema[col].dtype)
        return pl.lit(None, dtype=self.schema[col].dtype).alias(col)
