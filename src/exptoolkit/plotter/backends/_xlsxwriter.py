from __future__ import annotations
from collections import defaultdict
from logging import getLogger
from typing import Any, Literal, MutableMapping
from weakref import WeakKeyDictionary

try:
    from xlsxwriter import Workbook  # type: ignore[import]
    from xlsxwriter.worksheet import Worksheet  # type: ignore[import]
    from xlsxwriter.chart_scatter import ChartScatter  # type: ignore[import]
except ImportError as exc:
    raise ImportError("xlswriter is not installed. ") from exc
from exptoolkit.plotter.backends._base import Target
from exptoolkit.plotter.colors import parse_color

# Per-chart user options.
# Stored separately because Chart.x_axis/y_axis are internal state and
# must not be reused as input to set_x_axis()/set_y_axis().
# Reusing them can generate invalid DrawingML (e.g. empty <a:solidFill/>).
_chart_options: MutableMapping[
    ChartScatter,
    dict[str, dict[str, Any]]
] = WeakKeyDictionary()

logger = getLogger(__name__)

class XlsxWriterTarget(Target):
    """XlsxWriterTarget backend for plotting graphs. Implements the Target protocol."""
    def __init__(
        self,
        ws: Worksheet,
        chart: ChartScatter | None = None,
        wb: Workbook | None = None,
        addr_default: str = "E5",
        ):
        """Initialize the XlsxWriterTarget with an xlsxwriter Worksheet object.
        Args:
            ws (Worksheet): An xlsxwriter Worksheet object where the data will be written.
            chart (ChartScatter | None): An xlsxwriter chart object.
                If None, a ChartScatter will be created by default.
            wb (Workbook | None): An xlsxwriter Workbook object where the plots will be drawn.
                wb is required if chart is None, as a new chart will be created in the workbook.
            addr_default (str): The default cell address where the chart will be placed
                if a new chart is created.
        """
        if chart is None:
            if wb is None:
                raise ValueError("Either wb or chart must be provided.")
            self.chart: ChartScatter = wb.add_chart(
                {"type": "scatter", "subtype": "straight_with_markers"}
                )  # type: ignore
            ws.insert_chart(addr_default, self.chart)  # type: ignore
        else:
            self.chart = chart
        self.ws = ws

        opts = _chart_options.get(self.chart)
        if opts is None:
            opts = defaultdict(dict)
            _chart_options[self.chart] = opts
        self._options: dict[str, dict[str, Any]] = opts

        if wb is not None:
            wb.nan_inf_to_errors = True

        if chart is None:
            self._apply_default_style()

    def _apply_default_style(self):
        """Apply default styling to the chart."""
        self.chart.set_size({"width": 640, "height": 480})
        self.chart.set_plotarea({"border": {"none": True}})
        self.chart.set_chartarea({"border": {"none": True}})
        self.chart.set_legend({"font": {"size": 10, "color": "#000000"}})
        self._options["title"].update({'name_font': {'size': 18, 'bold': False}})
        axis_options = {
            'major_gridlines': {
                'visible': True,
                'line': {'color': '#D3D3D3', 'dash_type': 'dot'}
            },
            'major_tick_mark': "inside",
            'num_font': {'size': 12, 'color': '#000000'},
            'name_font': {'size': 14, 'bold': False, 'color': '#000000'},
            'line': {'color': '#000000'},
        }
        self._options["x_axis"].update(axis_options)
        self._options["y_axis"].update(axis_options)
        self._apply_options()

    def add_line(self, x, y, color=None, label=None, **kwargs):
        max_col = self.ws.dim_colmax or -1
        prefix = label + "_" if label else ""
        self.ws.write_string(row=0, col=max_col + 1, string=prefix + str(getattr(x, "name", "x")))
        self.ws.write_string(row=0, col=max_col + 2, string=prefix + str(getattr(y, "name", "y")))
        self.ws.write_column(row=1, col=max_col + 1, data=x.to_numpy() if hasattr(x, "to_numpy") else x)  # type: ignore
        self.ws.write_column(row=1, col=max_col + 2, data=y.to_numpy() if hasattr(y, "to_numpy") else y)  # type: ignore
        series_options = {
            "categories": [self.ws.name, 1, max_col + 1, len(x), max_col + 1],
            "values": [self.ws.name, 1, max_col + 2, len(y), max_col + 2],
            "name": label,
            "marker": {"type": "none"},
            "line": {"width": 1.5},
        }
        if color is not None:
            hexcolor = parse_color(color).as_hex()
            series_options["line"]["color"] = hexcolor
            series_options["marker"]["fill"] = {"color": hexcolor}
            series_options["marker"]["border"] = {"color": hexcolor}
        series_options.update(kwargs)
        self.chart.add_series(series_options)

    def add_scatter(self, x, y, c=None, color=None, label=None, color_scale="linear", **kwargs):
        if c is not None:
            logger.warning("Color mapping for scatter plots is not supported in XlsxWriter backend."
                " Ignoring color information.")
        max_col = self.ws.dim_colmax or -1
        prefix = label + "_" if label else ""
        self.ws.write_string(row=0, col=max_col + 1, string=prefix + str(getattr(x, "name", "x")))
        self.ws.write_string(row=0, col=max_col + 2, string=prefix + str(getattr(y, "name", "y")))
        self.ws.write_column(row=1, col=max_col + 1, data=x.to_numpy() if hasattr(x, "to_numpy") else x)  # type: ignore
        self.ws.write_column(row=1, col=max_col + 2, data=y.to_numpy() if hasattr(y, "to_numpy") else y)  # type: ignore
        series_options = {
            "categories": [self.ws.name, 1, max_col + 1, len(x), max_col + 1],
            "values": [self.ws.name, 1, max_col + 2, len(y), max_col + 2],
            "name": label,
            "marker": {
                "type": "circle",
                "size": 5,
            },
            "line": {"none": True},
        }
        if color is not None:
            hexcolor = parse_color(color).as_hex()
            series_options["line"]["color"] = hexcolor
            series_options["marker"]["fill"] = {"color": hexcolor}
            series_options["marker"]["border"] = {"color": hexcolor}
        series_options.update(kwargs)
        self.chart.add_series(series_options)

    def set_ax_label(self, axis: Literal["x", "y"], label: str) -> None:
        self._options[f'{axis}_axis']["name"] = label
        self._apply_options()

    def set_scale(self, axis: Literal["x", "y"], scale: Literal["linear", "log"]) -> None:
        if scale == 'log':
            self._options[f'{axis}_axis']['log_base'] = 10
        else:
            self._options[f'{axis}_axis'].pop('log_base', None)
        self._apply_options()

    def set_title(self, title: str) -> None:
        self._options["title"]["name"] = title
        self._apply_options()

    def set_aspect(self, aspect: Literal["equal", "auto"]) -> None:
        logger.warning("Setting aspect ratio is not supported in XlsxWriter backend. Skipping.")

    def reverse_axis(self, x: bool | None = None, y: bool | None = None) -> None:
        if x is not None:
            self._options["x_axis"]["reverse"] = x
        if y is not None:
            self._options["y_axis"]["reverse"] = y
        self._apply_options()

    def _apply_options(self):
        for key, opts in self._options.items():
            setter = getattr(self.chart, f"set_{key}")
            setter(opts)

    @classmethod
    def from_obj(cls, obj):
        """Create an XlsxWriterTarget from an XlsxWriter Worksheet or Chart object."""
        if isinstance(obj, XlsxWriterTarget):
            return obj
        if isinstance(obj, tuple):
            wb, ws, chart = None, None, None
            for item in obj:
                if isinstance(item, Workbook):
                    wb = item
                elif isinstance(item, Worksheet):
                    ws = item
                elif isinstance(item, ChartScatter):
                    chart = item
            if ws is not None:
                return cls(ws=ws, chart=chart, wb=wb)

        raise TypeError(
            "Unrecognized object type for XlsxWriterTarget. "
            f"Expected tuple containing Workbook, Worksheet, and/or ChartScatter objects. "
            f"Got: {repr(obj)}"
            )
