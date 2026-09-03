from __future__ import annotations

import datetime
import enum
import json
import sys
import typing as t
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from logging import getLogger
from pathlib import PurePath
from types import MappingProxyType
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import polars as pl

if t.TYPE_CHECKING:
    import numpy.typing as npt
    from pint import UnitRegistry
    from polars._typing import FrameInitTypes, IntoExprColumn

logger = getLogger()


# `import pint` takes some time, so perform lazy import
@lru_cache
def load_ureg() -> UnitRegistry:
    import pint

    return pint.UnitRegistry()


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
    amount: float | None = None
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
    base = base_unit or "dimensionless"
    norm = normalize_unit or "dimensionless"
    to = to_unit or "dimensionless"
    return (ureg.Quantity(1.0, base) / ureg.Quantity(1.0, norm) ** role).to(to).magnitude


@dataclass
class Column:
    dtype: type[pl.DataType] | pl.DataType
    base_unit: str = "dimensionless"
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
        self, obj: BaseData | None, owner: type[BaseData] | None = None
    ) -> Column | pl.Series:
        if obj is None:
            return self
        return obj.table.get_column(self.name)

    def __set__(self, obj: BaseData, value: t.Any) -> None:
        if isinstance(value, (pl.Expr, pl.Series)):
            expr = value
        elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            # list, tuple, numpy.ndarray, pandas.Series, etc.
            expr = pl.Series(self.name, value)
        else:
            # scalar value
            expr = pl.lit(value)
        obj.table = obj.table.with_columns(expr.cast(self.dtype).alias(self.name))

    def get_spec(self) -> ColumnSpec:
        return ColumnSpec(role=self.role, dtype=self.dtype, base_unit=self.base_unit)

    @property
    def expr(self) -> pl.Expr:
        """returns the expression of the column."""
        return pl.col(self.name)


class SchemaMixin:
    schema: Mapping[str, ColumnSpec]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name, value in cls.__dict__.items():
            inherited = _find_inherited_attr(cls, name)

            if inherited is _NOT_FOUND:
                continue

            if isinstance(value, Column) and not isinstance(inherited, Column):
                raise ValueError(f"Column name {name!r} conflicts with an inherited attribute.")

            if not isinstance(value, Column) and isinstance(inherited, Column):
                raise ValueError(f"Column {name!r} cannot be overridden by a non-Column attribute.")

        schema = OrderedDict()

        # Walk bases from oldest to newest so subclass Column definitions
        # override base definitions.
        for base in reversed(cls.__mro__):
            for name, value in base.__dict__.items():
                if isinstance(value, Column):
                    schema[name] = value.get_spec()

        cls.schema = MappingProxyType(schema)


_NOT_FOUND = object()
_ANNOTATED = object()


def _find_inherited_attr(cls: type, name: str) -> object:
    for base in cls.__mro__[1:]:
        if name in base.__dict__:
            return base.__dict__[name]

        annotations = base.__dict__.get("__annotations__", {})
        if name in annotations:
            return _ANNOTATED

    return _NOT_FOUND


class BaseData(SchemaMixin):
    """Base class for representing data."""

    _MANIFEST_VERSION: t.Final = 1
    norm: NormPolicy

    def __init__(
        self,
        table: FrameInitTypes,
        *,
        normalization: tuple[float | None, str | None] = NormPolicy(None, None),
        metadata: Mapping[str, t.Any] | None = None,
        drop_extra_columns: bool = True,
    ):
        """
        :param table: original data.
        :param normalization:
            Information about normalization as a tuple of (amount, unit).
            - If no normalization is applied, the unit should be None.
            - If normalization is applied (unit is known) but the amount is unknown,
                the amount should be None.
            - If normalized by a dimensionless factor,
                the unit should be 'dimensionless'.
            Note: Specifying `normalization` does not actually perform normalization.
                This argument only indicates that the table data is already normalized.
                Actual normalization can be performed using the normalize() method.
        :param metadata: Other metadata (must be JSON-serializable).
            This can be used to store any additional information about the data.
        :param drop_extra_columns: If True, drop columns not included in the schema.
        """
        norm = NormPolicy(*normalization)
        self.norm = norm
        self._metadata = JSONDict(metadata or {})
        df = pl.DataFrame(table)
        exprs = [self._col_or_null(df, col) for col in self.schema]
        for col in self.schema:
            if col not in df.columns:
                logger.info('missing column: "%s"', col)
        if not drop_extra_columns:
            exprs = exprs + [pl.col(col) for col in df.columns if col not in self.schema]
        self.table = df.select(exprs)

    def save(self, path) -> None:
        """saves the data as .zip file."""
        with ZipFile(path, "w", compression=ZIP_STORED) as zip_file:
            with zip_file.open("table.parquet", "w") as f:
                self.table.write_parquet(f)

            manifest = {
                "metadata": self.metadata.to_builtin(),
                "norm": tuple(self.norm),
                "version": self._MANIFEST_VERSION,
                "format": "exptoolkit",
                "class": type(self).__name__,
            }
            zip_file.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> t.Self:
        """loads the data from a .zip file."""
        with ZipFile(path, "r") as zip_file:
            try:
                b = zip_file.read("manifest.json")
            except KeyError as exc:
                raise ValueError("Unsupported file (manifest.json does not exist)") from exc

            manifest = json.loads(b.decode("utf-8"))

            fmt = manifest.get("format")
            if fmt != "exptoolkit":
                raise ValueError(f"Unsupported file format: {fmt!r}")

            # Files without a version field are treated as version 1.
            version = manifest.get("version", 1)
            if version != cls._MANIFEST_VERSION:
                # Add migration logic here if backward compatibility is needed
                # when the file format changes in the future.
                raise ValueError(f"Unsupported file version: {version!r}")

            metadata = manifest.get("metadata", {})
            norm = tuple(manifest.get("norm", (None, None)))
            class_ = manifest.get("class")

            if class_ != cls.__name__:
                warnings.warn(
                    f"File was created as {class_}, but is being loaded as {cls.__name__}.",
                    UserWarning,
                    stacklevel=2,
                )

            with zip_file.open("table.parquet", "r") as f:
                table = pl.read_parquet(f)

        return cls(table, normalization=norm, metadata=metadata, drop_extra_columns=False)

    def export_excel(self, path, ws_table="table", ws_manifest="manifest") -> None:
        from xlsxwriter import Workbook

        """exports the data to a .xlsx file."""
        manifest = {
            "class": type(self).__name__,
            "norm": tuple(self.norm),
            "metadata": self.metadata.to_builtin(),
        }
        with Workbook(path) as wb:
            self.table.write_excel(wb, worksheet=ws_table)
            ws = wb.add_worksheet(ws_manifest)
            for i, (path_, obj) in enumerate(flatten_json(manifest)):
                ws.write(i, 0, path_)
                try:
                    ws.write(i, 1, obj)
                except TypeError:
                    ws.write(i, 1, str(obj))

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    @table.setter
    def table(self, table: FrameInitTypes) -> None:
        """sets table data. checks if table schema is consistent with self."""
        if not isinstance(table, pl.DataFrame):
            table = pl.DataFrame(table)
        is_valid = True
        errors = ["schema does not match."]
        for key, spec in self.schema.items():
            if key not in table.columns:
                is_valid = False
                errors.append(f"- table does not contain required column '{key}'")
            elif table.schema[key] != spec.dtype:
                is_valid = False
                errors.append(
                    f"- dtype mismatch in column '{key}'."
                    f"given: {table.schema[key]}, expected: {spec.dtype}"
                )
        if is_valid:
            self._table = table
        else:
            raise ValueError("\n".join(errors))

    df = table  # alias for convenience

    @property
    def metadata(self) -> JSONDict:
        """Free-form metadata associated with the data."""
        # Design note:
        # Metadata is intentionally schema-free. TypedDict and Pydantic-based
        # approaches to subclass-specific metadata schemas were explored, but their
        # static typing and runtime machinery added disproportionate complexity.
        #
        # State interpreted by BaseData itself should instead be represented by an
        # explicit attribute, as with `norm`.
        return self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, t.Any]) -> None:
        self._metadata = JSONDict(value)

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

    def get_unit(self, column: str | Column, fmt="~P") -> str:
        """gets the unit associated with the given column.
        considers current normalization information."""
        if isinstance(column, Column):
            column = column.name
        ureg = load_ureg()
        base = self.schema[column].base_unit
        role = self.schema[column].role
        norm = self.norm.unit or "dimensionless"
        return f"{ureg.Unit(base) / ureg.Unit(norm) ** role:{fmt}}"

    def downsample(self, n: int, offset: int = 0) -> t.Self:
        """takes every n points with offset, and returns a new data object."""
        return self.with_table(self.table.gather_every(n, offset))

    def normalize(self, norm_amount: float, norm_unit: str) -> t.Self:
        """Normalize data by dividing columns by the given amount and unit.
        Returns a new BaseData object with normalized values."""
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
        self,
        *predicates: (
            IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | npt.NDArray[np.bool_]
        ),
        **constraints: t.Any,
    ) -> t.Self:
        return self.with_table(self.table.filter(*predicates, **constraints))

    def denormalize(self) -> t.Self:
        if self.norm.unit is None:
            new_data = copy(self)
            new_data.metadata = copy(self.metadata)
        else:
            if self.norm.amount is None:
                raise ValueError("cannot denormalize because normalization amount is unknown")
            exprs = [
                pl.col(col) * self.norm.amount**spec.role
                for col, spec in self.schema.items()
                if spec.role != Role.INTENSIVE
            ]
            new_data = self.with_table(self.table.with_columns(exprs))
            new_data.norm = NormPolicy(None, None)
            new_data.metadata = copy(self.metadata)
        return new_data

    def with_table(self, table: pl.DataFrame, copy_metadata: bool = True) -> t.Self:
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


class JSONSerializationWarning(UserWarning):
    """Warning for values converted during JSON serialization."""


_JsonNode = t.Union[None, bool, int, float, str, "JSONList", "JSONDict"]


def _normalize_json_value(value) -> _JsonNode:
    if value is None:
        return None

    # Types that can be safely converted to JSON-serializable types
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)

    # JSON-serializable types
    if isinstance(value, (bool, int, float, str, JSONList, JSONDict)):
        return value
    if isinstance(value, (list, tuple)):
        return JSONList(value)
    if isinstance(value, dict):
        return JSONDict(value)

    # Other types that can be converted to JSON-serializable types with a warning
    if isinstance(value, PurePath):
        _warn_conversion(value, "string")
        return str(value)
    if isinstance(value, (datetime.time, datetime.date)):
        _warn_conversion(value, "ISO format string")
        return value.isoformat()
    if isinstance(value, np.ndarray):
        _warn_conversion(value, "list")
        return JSONList(value.tolist())
    if isinstance(value, pl.Series):
        _warn_conversion(value, "list")
        return JSONList(value.to_list())
    if isinstance(value, pl.DataFrame):
        _warn_conversion(value, "dict")
        return JSONDict(value.to_dict(as_series=False))
    pd = sys.modules.get("pandas")
    if pd is not None:
        if isinstance(value, pd.Timestamp):
            _warn_conversion(value, "ISO format string")
            return value.isoformat()
        if isinstance(value, pd.Series):
            _warn_conversion(value, "list")
            return JSONList(value.to_list())
        if isinstance(value, pd.DataFrame):
            _warn_conversion(value, "dict")
            return JSONDict(value.to_dict(orient="list"))
    try:
        from pathlib_abc import JoinablePath
    except ImportError:
        pass
    else:
        if isinstance(value, JoinablePath):
            _warn_conversion(value, "string")
            return str(value)

    # Unsupported types
    raise TypeError(f"Value of type {type(value)} is not supported for JSON serialization")


def _warn_conversion(value, to_type: str, stacklevel=4) -> None:
    msg = f"{type(value)} is not JSON-serializable. converting to {to_type}."
    warnings.warn(msg, JSONSerializationWarning, stacklevel=stacklevel)


class JSONDict(MutableMapping[str, t.Any]):
    """A dict-like class that only accepts JSON-serializable values."""

    _data: dict[str, _JsonNode]

    def __init__(self, initial: t.Mapping[str, t.Any] | None = None) -> None:
        self._data = {}
        if initial is not None:
            for k, v in initial.items():
                self[k] = v

    def __getitem__(self, key: str) -> t.Any:
        # Intentionally return Any: callers usually know the semantic type from
        # the key, while exposing _JsonNode would only add narrowing burden.
        return self._data[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)}")
        self._data[key] = _normalize_json_value(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_builtin(self) -> dict[str, t.Any]:
        return _to_builtin(self)

    def __repr__(self):
        return f"{type(self).__name__}({self._data!r})"

    def __rich__(self):
        return self.to_builtin()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        cls = type(self)
        result = cls.__new__(cls)
        result._data = self._data.copy()
        return result

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._data = deepcopy(self._data, memo)
        return result


class JSONList(MutableSequence):
    _data: list[_JsonNode]

    def __init__(self, initial: t.Iterable[t.Any] | None = None):
        self._data = []
        if initial is not None:
            for v in initial:
                self.append(v)

    @t.overload
    def __getitem__(self, index: int) -> t.Any: ...

    @t.overload
    def __getitem__(self, index: slice) -> JSONList: ...

    def __getitem__(self, index: int | slice) -> t.Any:
        if isinstance(index, slice):
            return JSONList(self._data[index])
        return self._data[index]

    def __setitem__(self, index, value: t.Any) -> None:
        self._data[index] = _normalize_json_value(value)

    def __delitem__(self, index):
        del self._data[index]

    def insert(self, index, value: t.Any) -> None:
        self._data.insert(index, _normalize_json_value(value))

    def __len__(self):
        return len(self._data)

    def to_builtin(self) -> list[t.Any]:
        return _to_builtin(self)

    def __repr__(self):
        return f"{type(self).__name__}({self._data!r})"

    def __rich__(self):
        return self.to_builtin()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        cls = type(self)
        result = cls.__new__(cls)
        result._data = self._data.copy()
        return result

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._data = deepcopy(self._data, memo)
        return result

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return list(self) == list(other)
        return NotImplemented


def _to_builtin(v: _JsonNode) -> t.Any:
    if isinstance(v, JSONDict):
        return {k: _to_builtin(val) for k, val in v.items()}
    if isinstance(v, JSONList):
        return [_to_builtin(item) for item in v]
    return v


def flatten_json(obj: t.Any, prefix="") -> list[tuple[str, t.Any]]:
    rows = []

    if isinstance(obj, dict):
        if not obj:
            rows.append((prefix, "{}"))
        else:
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(flatten_json(value, path))

    elif isinstance(obj, (list, tuple)):
        if not obj:
            rows.append((prefix, "[]"))
        else:
            for i, value in enumerate(obj):
                rows.extend(flatten_json(value, f"{prefix}[{i}]"))

    else:
        rows.append((prefix, obj))

    return rows
