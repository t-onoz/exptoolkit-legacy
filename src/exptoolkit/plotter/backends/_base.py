from __future__ import annotations

import typing as t

from exptoolkit.plotter.colors import ColorLike

if t.TYPE_CHECKING:
    # these are just for type annotation
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

VectorLike: t.TypeAlias = "t.Sequence[int | float] | npt.NDArray[np.number] | pd.Series | pl.Series"


@t.runtime_checkable
class Target(t.Protocol):
    """Protocol for plotting graphs, absorbing differences in backends.
    The methods implement specific plotting commands."""

    def add_line(
        self,
        x: VectorLike,
        y: VectorLike,
        color: ColorLike | None = None,
        label: str | None = None,
        **kwargs: t.Any,
    ) -> t.Any: ...
    def add_scatter(
        self,
        x: VectorLike,
        y: VectorLike,
        c: VectorLike | None = None,
        color: ColorLike | None = None,
        label: str | None = None,
        color_scale: t.Literal["log", "linear"] = "linear",
        **kwargs: t.Any,
    ) -> t.Any: ...
    def set_ax_label(self, axis: t.Literal["x", "y"], label: str) -> t.Any: ...
    def set_scale(self, axis: t.Literal["x", "y"], scale: t.Literal["linear", "log"]) -> t.Any: ...
    def set_title(self, title: str) -> t.Any: ...
    def set_aspect(self, aspect: t.Literal["equal", "auto"]) -> t.Any: ...
    def reverse_axis(self, x: bool | None = None, y: bool | None = None) -> t.Any: ...
    @classmethod
    def from_obj(cls, obj: t.Any) -> Target: ...
