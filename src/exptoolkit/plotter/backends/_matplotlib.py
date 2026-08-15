from __future__ import annotations

from exptoolkit.plotter.backends._base import Target
from exptoolkit.plotter.colors import parse_color

try:
    from matplotlib.axes import Axes
except ImportError as exc:
    raise ImportError("Matplotlib is not installed.") from exc


class MatplotlibTarget(Target):
    """Matplotlib backend for plotting graphs. Implements the Target protocol."""

    def __init__(self, ax: Axes):
        """Initialize the MatplotlibTarget with a Matplotlib Axes object.
        Args:
            ax (Axes): A Matplotlib Axes object where the plots will be drawn.
        """
        self.ax = ax

    def add_line(self, x, y, color=None, label=None, **kwargs):
        if "fmt" in kwargs:
            args: tuple = (x, y, kwargs.pop("fmt"))
        else:
            args = (x, y)
        if color:
            color_ = parse_color(color)
            kwargs["color"] = color_.as_hex(include_alpha=True)
        return self.ax.plot(*args, label=label, **kwargs)

    def add_scatter(self, x, y, c=None, color=None, label=None, color_scale="linear", **kwargs):
        if color_scale == "log":
            kwargs["norm"] = "log"
        if color is not None:
            kwargs["color"] = parse_color(color).as_hex(include_alpha=True)
        return self.ax.scatter(x, y, c=c, label=label, **kwargs)

    def set_ax_label(self, axis, label):
        if axis == "x":
            return self.ax.set_xlabel(label)
        if axis == "y":
            return self.ax.set_ylabel(label)
        raise ValueError(f"Unknown axis: {axis}")

    def set_ylabel(self, label):
        return self.ax.set_ylabel(label)

    def set_title(self, title):
        return self.ax.set_title(title)

    def set_aspect(self, aspect):
        if aspect == "equal":
            return self.ax.set_aspect("equal", "box")
        return self.ax.set_aspect(aspect)

    def set_scale(self, axis, scale):
        if axis == "x":
            return self.ax.set_xscale(scale)
        if axis == "y":
            return self.ax.set_yscale(scale)
        raise ValueError(f"Unknown axis: {axis}")

    def reverse_axis(self, x=None, y=None):
        if x is not None:
            if x and not self.ax.xaxis_inverted():
                self.ax.invert_xaxis()
            if not x and self.ax.xaxis_inverted():
                self.ax.invert_xaxis()
        if y is not None:
            if y and not self.ax.yaxis_inverted():
                self.ax.invert_yaxis()
            if not y and self.ax.yaxis_inverted():
                self.ax.invert_yaxis()

    @classmethod
    def from_obj(cls, obj) -> MatplotlibTarget:
        """Create a MatplotlibTarget instance from a given object.
        Args:
            obj: A Matplotlib Axes object, or an existing MatplotlibTarget instance.
        Returns:
            MatplotlibTarget: An instance of MatplotlibTarget.
        """
        if isinstance(obj, MatplotlibTarget):
            return obj
        if isinstance(obj, Axes):
            return cls(obj)
        raise TypeError(f"Cannot create MatplotlibTarget from object of type {type(obj)}")
