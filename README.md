# exptoolkit

A small, typed toolkit for representing, processing, indexing, and plotting experimental data in Python.

`exptoolkit` provides a common data model for experimental measurements: define the columns that belong to a data type, attach physical units and normalization behavior, keep metadata with the data, and build reusable processing and plotting code on top of that model.

The library is intentionally domain-agnostic. Instrument- or experiment-specific logic can live in separate packages while sharing the same data and processing interfaces.

## Features

- **Schema-defined experimental data** built on [Polars](https://pola.rs/)
- **Physical units** and unit conversion with [Pint](https://pint.readthedocs.io/)
- **Normalization-aware columns** for extensive, intensive, and inverse-extensive quantities
- **JSON-compatible metadata** stored alongside measurement data
- **Portable save/load format** using Parquet data plus a JSON manifest
- **Reusable processing interfaces** for modifiers, converters, combiners, and featurizers
- **Resource indexing** by measurement ID and sample name
- **Directory scanning with caching** for incrementally updating resource indexes
- **Backend-independent plotting** with adapters for Matplotlib, Plotly, PyQtGraph, OpenPyXL, and XlsxWriter
- **Type information included** via `py.typed`

## Installation

```bash
pip install exptoolkit
```

Optional plotting dependencies can be installed with:

```bash
pip install "exptoolkit[plotting]"
```

For PyQtGraph/PySide6 support:

```bash
pip install "exptoolkit[qt]"
```

## Quick start

### Define an experimental data type

Columns are declared once on a `BaseData` subclass. Each column has a Polars dtype, a base unit, and optionally a physical role used during normalization.

```python
import polars as pl

from exptoolkit import BaseData, Column, Role


class MeasurementData(BaseData):
    time = Column(pl.Float64, "s")
    voltage = Column(pl.Float64, "V")
    capacity = Column(pl.Float64, "mAh", Role.EXTENSIVE)


data = MeasurementData(
    {
        "time": [0.0, 1.0, 2.0],
        "voltage": [3.10, 3.25, 3.40],
        "capacity": [0.0, 12.5, 25.0],
    },
    metadata={"sample": "sample-001"},
)

print(data.table)
print(data.voltage)
```

The declared schema is available on the class:

```python
print(MeasurementData.schema)
```

Extra input columns are dropped by default, and missing schema columns are represented as null columns. This keeps instances of the same `BaseData` subclass structurally consistent.

### Work with units

Column values can be requested in compatible units without changing the stored data:

```python
capacity_ah = data.col_to_unit("capacity", "Ah")
time_min = data.col_to_unit("time", "min")
```

### Normalize by sample amount

Experimental quantities often depend on how much material was measured. For example,
capacity may be recorded in `mAh`, but comparing samples of different mass is usually more
useful after expressing it per unit mass, such as `mAh/g`.

`BaseData.normalize()` represents this operation explicitly. A normalization amount and its
unit are stored with the data, and each column declares how it should respond through its
`Role`:

- `Role.EXTENSIVE` — proportional to sample amount; divided by the normalization amount
  (for example, `mAh` → `mAh/g`)
- `Role.INTENSIVE` — independent of sample amount; left unchanged
  (for example, voltage or temperature)
- `Role.INVERSE_EXTENSIVE` — inversely proportional to sample amount; multiplied by the
  normalization amount

For example, if a measurement was obtained from a 25 mg sample:

```python
normalized = data.normalize(25, "mg")
```

An extensive `capacity` column is then stored as capacity divided by 25, while intensive
columns such as `time` and `voltage` are unchanged. The normalization unit becomes part of
the effective physical unit:

```python
print(normalized.get_unit("capacity"))
# mAh / mg

capacity_per_g = normalized.col_to_unit("capacity", "mAh/g")
```

Normalization is therefore not a generic rescaling of every numeric column. It is a
sample-amount transformation defined by the physical role of each column and tracked as
part of the `BaseData` state. `denormalize()` can restore the original extensive values
when the normalization amount is known.

### Save and load

`BaseData.save()` writes a portable ZIP container containing the table as Parquet and metadata as JSON.

```python
data.save("measurement.zip")
loaded = MeasurementData.load("measurement.zip")
```

Metadata is restricted to JSON-compatible values so saved data remains portable and inspectable.

## Processing

`exptoolkit.processing` provides lightweight interfaces for reusable operations on `BaseData` objects:

- `Modifier` — modifies one data object in place
- `Converter` — converts one data object to another
- `Combiner` — combines multiple data objects
- `Featurizer` — converts experimental data into a fixed-schema numeric feature vector

## Resource repository

`ResourceRepo` is a small in-memory index that associates external resources with measurement IDs and sample names.

```python
from exptoolkit.repository import ResourceRepo

repo = ResourceRepo()
repo.add(
    "/data/run001/sample_a.csv",
    measurement_id="run001",
    samples=["sample_a"],
    data_type="csv",
)

print(repo.by_sample("sample_a"))
print(repo.by_measurement("run001"))
```

The repository can be serialized to JSON and restored later.

### Scan a directory

`DirectoryScanner` can build and synchronize a repository from a directory layout in which each measurement has its own directory.

```text
data/
├── run001/
│   ├── sample_a.csv
│   └── sample_b.csv
└── run002/
    └── sample_c.csv
```

```python
from exptoolkit.repository import DirectoryScanner, ResourceRepo

scanner = DirectoryScanner(
    "data",
    dir_regex=r"run\d+",
    file_regex=r".*\.csv$",
)

repo = ResourceRepo()
scanner.scan_and_sync(repo)
```

The scanner maintains a per-measurement cache so unchanged directories do not need to be rescanned. Cache files can be saved and loaded with `save_cache()` and `load_cache()`.

For layouts that do not fit `DirectoryScanner`, subclass `ResourceScanner` and implement `owns()` and `scan()`.

## Plotting

The plotting layer separates *what to plot* from the plotting backend. A plotter operates on the `Target` protocol, while backend adapters translate those operations to supported plotting libraries.

For a simple x-y plot:

```python
from exptoolkit.plotter import XyPlotter

plotter = XyPlotter("time", "voltage", xunit="s", yunit="V")
plotter.plot(data, ax)  # for example, a Matplotlib Axes
```

Supported targets depend on installed optional dependencies and include Matplotlib, Plotly, PyQtGraph, OpenPyXL, and XlsxWriter objects.

This makes it possible for experiment-specific plotting code to remain largely independent of the final output backend.

## Design scope

`exptoolkit` focuses on infrastructure shared across experimental domains:

- structured measurement data
- units and normalization metadata
- generic processing contracts
- resource discovery and indexing
- plotting abstraction

Domain-specific analysis algorithms are intentionally kept outside the core package. The goal is to provide a small common layer that experiment-specific packages can build on rather than a collection of unrelated analysis routines.

## Status

`exptoolkit` is currently a **beta** project. The core concepts are in active use, but APIs may still change between minor releases while the public interface is refined.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and local development.

Clone the repository and create the development environment:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run linting, formatting checks, and type checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The test suite is also configured with tox for Python 3.9 through 3.13:

```bash
uv run tox
```

The test suite is configured for Python 3.9 through 3.13 with tox.

## License

MIT
