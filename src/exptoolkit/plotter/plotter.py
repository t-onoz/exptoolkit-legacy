from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from string import capwords
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from exptoolkit.data import BaseData
from exptoolkit.plotter.backends import get_target

if TYPE_CHECKING:
    from exptoolkit.plotter.backends import Target, TargetLike
    from exptoolkit.plotter.colors import ColorLike

M_contra = TypeVar("M_contra", bound=BaseData, contravariant=True)


class Plotter(Protocol[M_contra]):
    def plot(
        self,
        data: M_contra,
        target_like: TargetLike,
        label: str | None = None,
        color: ColorLike | None = None,
        **opts,
    ) -> Target:

        t = get_target(target_like)
        self._plot(data, t, label, color, **opts)
        return t

    @abstractmethod
    def _plot(
        self,
        data: M_contra,
        target: Target,
        label: str | None = None,
        color: ColorLike | None = None,
        **opts: Any,
    ): ...


@dataclass
class XyPlotter(Plotter[BaseData]):
    xcol: str
    ycol: str
    xunit: str | None = None
    yunit: str | None = None
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        x = data.col_to_unit(self.xcol, self.xunit)
        y = data.col_to_unit(self.ycol, self.yunit)
        target.add_line(x, y, label=label, color=color, **opts)
        if self.add_ax_labels:
            xunit = data.get_unit(self.xcol) if self.xunit is None else self.xunit
            yunit = data.get_unit(self.ycol) if self.yunit is None else self.yunit
            xlabel = f"{capwords(self.xcol)} ({xunit or '-'})"
            ylabel = f"{capwords(self.ycol)} ({yunit or '-'})"
            target.set_ax_label("x", xlabel)
            target.set_ax_label("y", ylabel)


class TargetManager(ABC):
    """Create and cache plotting targets.

    `TargetManager` creates plotting targets on demand and reuses them
    according to `plot_name` and a grouping key derived from the input data.

    Subclasses implement `group_key()` and `factory()` to define the grouping
    rule and target creation strategy.

    Notes
    -----
    This class is experimental and is not considered part of the stable public API.
    Its interface may change without deprecation while the design is being refined.
    """

    def __init__(self) -> None:
        self._targets: dict[tuple[str, Hashable], TargetLike] = {}

    @abstractmethod
    def group_key(self, data: BaseData) -> Hashable:
        pass

    @abstractmethod
    def factory(
        self,
        *,
        plot_name: str,
        group_key: Any,
    ) -> TargetLike:
        pass

    def get(
        self,
        data: BaseData,
        plot_name: str,
    ) -> TargetLike:
        group_key = self.group_key(data)
        target_key = (plot_name, group_key)

        if target_key not in self._targets:
            self._targets[target_key] = self.factory(group_key=group_key, plot_name=plot_name)
        return self._targets[target_key]
