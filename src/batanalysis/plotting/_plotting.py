from __future__ import annotations

import typing as t
from dataclasses import dataclass
from string import capwords

import polars as pl

from batanalysis.data import ChargeDischargeData, CycleSummaryData, EISData, State
from batanalysis.processing import calc_z_theta, differentiate, integrate_capacity
from exptoolkit.plotter import Plotter


@dataclass
class ChargeDischarge(Plotter[ChargeDischargeData]):
    cycle: int | t.Collection[int] | None = None
    mode: t.Literal["step", "cycle", "total"] = "step"
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        cls = ChargeDischargeData
        if not data.is_col_ready(cls.step_capacity.name):
            integrate_capacity(data)

        if self.cycle is None:
            pass
        elif isinstance(self.cycle, int):
            data = data.filter(cls.cycle.expr == self.cycle)
        else:
            data = data.filter(cls.cycle.expr.is_in(self.cycle))

        if self.mode == "step":
            x_expr: pl.Expr = pl.when(cls.state.expr.is_in([State.CHARGE, State.DISCHARGE])).then(
                cls.step_capacity.expr
            )
        elif self.mode == "cycle":
            x_expr = cls.cycle_capacity.expr
        else:
            x_expr = cls.capacity.expr

        df = data.table.select(
            x_expr.alias("x"),
            cls.voltage.expr,
        )

        x = df["x"]
        y = df[cls.voltage.name]
        target.add_line(x, y, label=label, color=color, **opts)
        if self.add_ax_labels:
            target.set_ax_label("x", f"Capacity ({data.get_unit('step_capacity')})")
            target.set_ax_label("y", f"Voltage ({data.get_unit('voltage')})")


@dataclass
class DqDv(Plotter[ChargeDischargeData]):
    cycle: int | t.Collection[int] | None = None
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        cls = ChargeDischargeData

        if not data.is_col_ready(cls.dqdv.name):
            differentiate(data)

        if self.cycle is None:
            pass
        elif isinstance(self.cycle, int):
            data = data.filter(cls.cycle.expr == self.cycle)
        else:
            data = data.filter(cls.cycle.expr.is_in(self.cycle))

        target.add_line(
            x=data.voltage,
            y=data.dqdv,
            label=label,
            color=color,
            **opts,
        )
        if self.add_ax_labels:
            target.set_ax_label("x", f"Voltage ({data.get_unit(cls.voltage.name)})")
            target.set_ax_label("y", f"dQ/dV ({data.get_unit(cls.dqdv.name)})")


@dataclass
class ColeCole(Plotter[EISData]):
    add_ax_labels: bool = True
    set_aspect: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        cls = EISData
        x = data.re_Z
        y = data.im_Z
        target.add_line(x, y, label=label, color=color, **opts)
        target.reverse_axis(y=True)
        if self.add_ax_labels:
            target.set_ax_label("x", f"Re[Z] ({data.get_unit(cls.re_Z.name)})")
            target.set_ax_label("y", f"Im[Z] ({data.get_unit(cls.im_Z.name)})")
        if self.set_aspect:
            target.set_aspect("equal")


@dataclass
class BodeTheta(Plotter[EISData]):
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        cls = EISData
        if not data.is_col_ready(cls.theta.name):
            calc_z_theta(data)
        x = data.frequency
        y = data.col_to_unit(cls.theta.name, "deg")
        target.add_line(x, y, label=label, color=color, **opts)
        target.set_scale("x", "log")
        if self.add_ax_labels:
            target.set_ax_label("x", f"Frequency ({data.get_unit(cls.frequency.name)})")
            target.set_ax_label("y", "theta (deg.)")


@dataclass
class BodeZ(Plotter[EISData]):
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        cls = EISData
        if not data.is_col_ready(cls.theta.name):
            calc_z_theta(data)
        x = data.frequency
        y = data.abs_Z
        target.add_line(x, y, label=label, color=color, **opts)
        target.set_scale("x", "log")
        if self.add_ax_labels:
            target.set_ax_label("x", f"Frequency ({data.get_unit(cls.frequency.name)})")
            target.set_ax_label("y", f"|Z| ({data.get_unit(cls.abs_Z.name)})")


@dataclass
class CycleSummary(Plotter[CycleSummaryData]):
    state: t.Literal["charge", "discharge"] = "discharge"
    mode: t.Literal["retention", "absolute"] = "retention"
    value: t.Literal["capacity", "energy"] = "capacity"
    add_ax_labels: bool = True

    def _plot(self, data, target, label=None, color=None, **opts):
        x = data.cycle
        col_y = (
            f"{self.value}_{self.state}_retention"
            if self.mode == "retention"
            else f"{self.value}_{self.state}"
        )
        y = data.table[col_y]
        target.add_line(x, y, label=label, color=color, **opts)
        if self.add_ax_labels:
            target.set_ax_label("x", "Cycle Number")
            yunit = data.get_unit(col_y)
            ylabel = f"{capwords(self.state)} {capwords(self.value)} ({yunit})"
            target.set_ax_label("y", ylabel)
