from __future__ import annotations

import numpy as np

from exptoolkit.plotter.backends._base import Target
from exptoolkit.plotter.colors import ColorLike, parse_color

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError("Plotly is not installed. ") from exc


class PlotlyTarget(Target):
    """Plotly backend for plotting graphs. Implements the Target protocol."""

    def __init__(self, fig: go.Figure, row: int | None = None, col: int | None = None):
        """Initialize the PlotlyTarget with a Plotly Figure object.
        Args:
            fig (Figure): A Plotly Figure object where the plots will be drawn.
            row (int | None): The row index of the subplot to draw on.
            col (int | None): The column index of the subplot to draw on.
        """
        self.fig = fig
        self.row = row
        self.col = col

    @staticmethod
    def _plotly_color(color: ColorLike):
        cobj = parse_color(color)
        s = ", ".join(str(x) for x in cobj.as_rgb_int())
        return f"rgba({s}, {cobj.a})"

    def add_line(self, x, y, color=None, label=None, **kwargs):
        if color:
            line_opts = kwargs.setdefault("line", {})
            line_opts["color"] = self._plotly_color(color)
        trace = go.Scatter(x=x, y=y, mode="lines", name=label, **kwargs)
        return self.fig.add_trace(trace, row=self.row, col=self.col)

    def add_scatter(self, x, y, c=None, color=None, label=None, color_scale="linear", **kwargs):
        marker = kwargs.pop("marker", {})
        if c is not None and color_scale == "log":
            c = np.log10(c)
        if c is not None:
            marker["color"] = c
        elif color is not None:
            marker["color"] = self._plotly_color(color)
        return self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=label,
                marker=marker,
                **kwargs,
            ),
            row=self.row,
            col=self.col,
        )

    def set_ax_label(self, axis, label):
        if axis == "x":
            return self.fig.update_xaxes(title=label, row=self.row, col=self.col)
        if axis == "y":
            return self.fig.update_yaxes(title=label, row=self.row, col=self.col)
        raise ValueError(f"Unknown axis: {axis}")

    def set_title(self, title):
        return self.fig.update_layout(title=title)

    def set_aspect(self, aspect):
        if aspect == "equal":
            return self.fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                row=self.row,
                col=self.col,
            )
        if aspect == "auto":
            return self.fig.update_yaxes(
                scalenachor=None,
                scaleratio=None,
                row=self.row,
                col=self.col,
            )
        raise ValueError(f"Aspect must be either 'equal' or 'auto', not '{aspect}'")

    def set_scale(self, axis, scale):
        if axis == "x":
            return self.fig.update_xaxes(type=scale)
        if axis == "y":
            return self.fig.update_yaxes(type=scale)
        raise ValueError(f"Unknown axis: {axis}")

    def reverse_axis(self, x=None, y=None):
        if x is not None:
            self.fig.update_xaxes(autorange=("reversed" if x else True), row=self.row, col=self.col)
        if y is not None:
            self.fig.update_yaxes(autorange=("reversed" if y else True), row=self.row, col=self.col)
        return self.fig

    @classmethod
    def from_obj(cls, obj) -> PlotlyTarget:
        """Create a PlotlyTarget instance from a given object.
        Args:
            obj: A Plotly Figure object, or a tuple of (Figure, row, col),
                or an existing PlotlyTarget instance.
        Returns:
            PlotlyTarget: An instance of PlotlyTarget.
        """
        if isinstance(obj, PlotlyTarget):
            return obj
        if isinstance(obj, tuple) and len(obj) == 3 and isinstance(obj[0], go.Figure):
            fig, row, col = obj
            return cls(fig, row=row, col=col)
        if isinstance(obj, go.Figure):
            return cls(obj)
        raise TypeError(f"Cannot create PlotlyTarget from object of type {type(obj)}")
