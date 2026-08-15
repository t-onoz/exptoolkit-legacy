# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import numpy as np

from exptoolkit.plotter.backends._base import Target
from exptoolkit.plotter.colors import parse_color

try:
    import pyqtgraph as pg
except ImportError as exc:
    raise ImportError("PyQtGraph is not installed. ") from exc


class PyQtGraphTarget(Target):
    """PyQtGraph backend for plotting graphs. Implements the Target protocol."""

    def __init__(self, target: pg.PlotWidget | pg.PlotItem):  # pyright: ignore[reportInvalidTypeForm]  # pyright#11481
        """Initialize the PyQtGraphTarget with a PlotWidget or PlotItem object.
        Args:
            target (PlotWidget | PlotItem): A PyQtGraph PlotWidget or PlotItem object
                where the plots will be drawn.
        """
        self.target = target

    def add_line(self, x, y, color=None, label=None, **kwargs):
        if color is not None:
            cobj = parse_color(color)
            kwargs["pen"] = cobj.as_rgb_int(include_alpha=True)
        return self.target.plot(x, y, name=label, **kwargs)

    def add_scatter(self, x, y, c=None, color=None, label=None, color_scale="linear", **kwargs):
        kwargs["pen"] = None
        kwargs["symbol"] = "o"
        if c is not None:
            if color_scale == "log":
                c = np.log10(c)
            for n in ("cmap", "palette", "map_name", "colormap_name"):
                colormap_name = kwargs.pop(n, None)
                if colormap_name is not None:
                    break
            colormap_name = colormap_name or "cividis"

            n_pts = 512
            colormap = pg.colormap.get(colormap_name)
            if colormap is None:
                raise ValueError(f"Failed to get colormap: {colormap_name}")
            value_range = np.linspace(np.min(c), np.max(c), n_pts)
            colors: np.ndarray = colormap.getLookupTable(0, 1, n_pts)  # pyright: ignore[reportAssignmentType]
            brushes = colors[np.searchsorted(value_range, c)]
            kwargs["symbolBrush"] = brushes
        elif color is not None:
            kwargs["symbolBrush"] = parse_color(color).as_rgb_int(include_alpha=True)
        return self.target.plot(x, y, name=label, **kwargs)

    def set_ax_label(self, axis, label):
        if axis == "x":
            return self.target.setLabel("bottom", label)
        if axis == "y":
            return self.target.setLabel("left", label)
        raise ValueError(f"Unknown axis: {axis}")

    def set_title(self, title):
        return self.target.setTitle(title)

    def set_aspect(self, aspect):
        if aspect == "equal":
            return self.target.setAspectLocked(True, ratio=1)  # type: ignore
        return self.target.setAspectLocked(False)

    def set_scale(self, axis, scale):
        enable = scale == "log"
        if axis == "x":
            return self.target.setLogMode(enable, None)
        if axis == "y":
            return self.target.setLogMode(None, enable)
        raise ValueError(f"Unknown axis: {axis}")

    def reverse_axis(self, x=None, y=None):
        if x is not None:
            self.target.invertX(x)
        if y is not None:
            self.target.invertY(y)

    @classmethod
    def from_obj(cls, obj):
        """Create a PyQtGraphTarget instance from a given object if it is compatible.
        Args:
            obj: An object that can be used to create a PyQtGraphTarget. This can be
                an existing PyQtGraphTarget instance, a PlotWidget or PlotItem,
                or an object that contains one of these as an attribute.
        Returns:
            PyQtGraphTarget: An instance of PyQtGraphTarget created from the given object.
        Raises:
            TypeError: If the object cannot be used to create a PyQtGraphTarget.
        """
        if isinstance(obj, PyQtGraphTarget):
            return obj
        if isinstance(obj, (pg.PlotWidget, pg.PlotItem)):  # pyright: ignore[reportArgumentType]  # pyright#11481
            return cls(obj)
        for attr in ("plot_widget", "plot_item", "widget", "item"):
            target = getattr(obj, attr, None)
            if isinstance(target, (pg.PlotWidget, pg.PlotItem)):  # pyright: ignore[reportArgumentType]  # pyright#11481
                return cls(target)
        raise TypeError(
            f"Cannot create PyQtGraphTarget from object: {obj}. "
            "Object must be a PlotWidget or PlotItem, or contain one as an attribute."
        )
