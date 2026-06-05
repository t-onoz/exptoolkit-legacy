# ExpToolKit

Python library for experimental data modeling, processing, repository management, and multi-backend plotting.

## Key Features

- Unit-aware experimental data models with schema-driven columns and normalization metadata
- Data processing utilities for downsampling, filtering, normalization, denormalization, and concatenation
- Protocol-based plotting support with backend adapters for Matplotlib, Plotly, PyQtGraph, and OpenPyXL
- Repository management for external data resources, measurements, and sample associations
- Flexible color handling via CSS names, hex strings, tuple values, and Matplotlib cycle references

## Installation

```bash
pip install .
```

For detailed usage examples, please refer to the Jupyter Notebooks under the `examples/` directory.

## Core Concepts

### Data Model

- `exptoolkit.data.BaseData` is the base class for table-backed experimental data objects.
- `Column` descriptors define schema, data types, base units, and physical roles (`EXTENSIVE`, `INTENSIVE`, `INVERSE_EXTENSIVE`).
- Normalization state is tracked through `NormPolicy`, enabling unit-aware operations and metadata preservation.

### Processing

- Protocols `Modifier`, `Converter`, and `Combiner` define reusable processing abstraction patterns.

### Plotting

- `exptoolkit.plotter.Plotter` is a protocol for objects that render data to a plotting target.
- The backend registry adapts runtime plotting targets via `exptoolkit.plotter.backends.get_target()`.
- Supported backends include Matplotlib, Plotly, PyQtGraph, and OpenPyXL when installed.

### Repository Management

- `exptoolkit.repository.DataResource` represents an external data reference, such as a file, URL, or archive path.
- `exptoolkit.repository.ResourceRepo` manages mappings between resources, measurement IDs, and sample names.
- Lookup methods support exact sample queries, regex sample search, and measurement-based retrieval.

## Development

```bash
python -m build
pip install dist/exptoolkit-0.1.0-py3-none-any.whl
```

## Packages

- `exptoolkit`
- `batanalysis`
