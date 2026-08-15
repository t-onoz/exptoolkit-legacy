from __future__ import annotations

import sys
import typing as t
from importlib import import_module
from logging import getLogger

from exptoolkit.plotter.backends._base import Target

logger = getLogger(__name__)

if t.TYPE_CHECKING:
    # why "if TYPE_CHECKING"?
    # Because these imports are only needed for type checking
    # and would cause unnecessary dependencies at runtime.
    from matplotlib.axes import Axes
    from openpyxl.chart import ScatterChart
    from openpyxl.worksheet.worksheet import Worksheet
    from plotly.graph_objects import Figure as PlotlyFigure
    from pyqtgraph import PlotItem, PlotWidget
    from xlsxwriter import Workbook as XlsxWorkbook
    from xlsxwriter.chart_scatter import ChartScatter as XlsxChartScatter
    from xlsxwriter.worksheet import Worksheet as XlsxWorksheet

    TargetLike = t.Union[
        Target,
        Axes,
        PlotlyFigure,
        tuple[PlotlyFigure, int, int],
        PlotWidget,
        PlotItem,  # pyright: ignore[reportInvalidTypeForm]
        Worksheet,
        tuple[Worksheet, t.Optional[ScatterChart]],
        tuple[XlsxWorksheet | XlsxChartScatter | XlsxWorkbook, ...],
    ]
else:
    # At runtime, we don't need the specific types, so we can just use Any.
    TargetLike = t.Any

_registry: dict[str, tuple[str, str]] = {
    # dependent package: (backend module, backend class)
    "matplotlib": ("exptoolkit.plotter.backends._matplotlib", "MatplotlibTarget"),
    "plotly": ("exptoolkit.plotter.backends._plotly", "PlotlyTarget"),
    "pyqtgraph": ("exptoolkit.plotter.backends._pyqtgraph", "PyQtGraphTarget"),
    "openpyxl": ("exptoolkit.plotter.backends._openpyxl", "OpenPyXlTarget"),
    "xlsxwriter": ("exptoolkit.plotter.backends._xlsxwriter", "XlsxWriterTarget"),
}


def get_target(obj: TargetLike) -> Target:
    """Create a Target instance from a given object by checking which backend it belongs to."""
    if isinstance(obj, Target):
        return obj
    available_backends: list[str] = []
    for pkg_name, (module_name, cls_name) in _registry.items():
        # Check if the package is available in the current environment.
        # This would improve performance by avoiding unnecessary imports.
        if pkg_name not in sys.modules:
            continue

        try:
            module = import_module(module_name)
        except ImportError:
            logger.warning(
                "Backend '%s' is available but failed to load (%s)", pkg_name, module_name
            )
            logger.debug("Import error details", exc_info=True)
            continue
        available_backends.append(pkg_name)
        cls: type[Target] = getattr(module, cls_name)

        try:
            return cls.from_obj(obj)
        except (TypeError, ValueError):
            continue
    if not available_backends:
        raise RuntimeError("No plotting backend is available.")
    raise ValueError(
        f"Could not create Target from object: {obj}. "
        f"Object does not match any registered backend targets."
        f"Available backends: {available_backends}"
    )
