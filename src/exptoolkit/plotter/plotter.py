from __future__ import annotations
from string import capwords
from typing import Protocol, TypeVar, TYPE_CHECKING, Any, Callable, Hashable
from abc import abstractmethod
from dataclasses import dataclass

from exptoolkit.data import BaseData
from exptoolkit.plotter.backends import get_target

if TYPE_CHECKING:
    from exptoolkit.plotter.backends import TargetLike, Target
    from exptoolkit.plotter.colors import ColorLike

M_contra = TypeVar('M_contra', bound=BaseData, contravariant=True)

class Plotter(Protocol[M_contra]):
    def plot(self,
            data: M_contra,
            target_like: TargetLike,
            label: str | None = None,
            color: ColorLike | None = None,
            **opts) -> Target:

        t = get_target(target_like)
        self._plot(data, t, label, color, **opts)
        return t

    @abstractmethod
    def _plot(self,
        data: M_contra,
        target: Target,
        label: str | None = None ,
        color: ColorLike | None = None,
        **opts: Any): ...

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


class TargetManager:
    """Manage and cache plotting targets.

    The TargetManager holds a mapping of (plot_name, group_key) -> target created
    by the provided target_factory. Use get(data, plot_name=...) to
    obtain a target for a specific data grouping; a new target is created on
    first request and reused thereafter.
    """
    def __init__(
        self,
        target_factory: Callable[[], TargetLike],
        key_func: Callable[[BaseData], Hashable],
    ) -> None:
        self.target_factory = target_factory
        self.targets: dict[tuple[str, Hashable], TargetLike] = {}
        self.key_func = key_func

    def get(
        self,
        data: BaseData,
        plot_name: str,
    ) -> TargetLike:
        group_key = self.key_func(data)
        key = (plot_name, group_key)

        if key not in self.targets:
            self.targets[key] = self.target_factory()
        return self.targets[key]
