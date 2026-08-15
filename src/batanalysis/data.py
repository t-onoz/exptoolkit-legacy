from __future__ import annotations

import typing as t

import polars as pl

from exptoolkit.data import BaseData, Column, Role


class State:
    CHARGE: t.Final[str] = "charge"
    DISCHARGE: t.Final[str] = "discharge"
    REST: t.Final[str] = "rest"
    UNKNOWN: t.Final[str] = "unknown"


class ChargeDischargeData(BaseData):
    time = Column(pl.Float64, "s", Role.INTENSIVE)
    cycle = Column(pl.UInt16, "dimensionless", Role.INTENSIVE)
    step = Column(pl.UInt32, "dimensionless", Role.INTENSIVE)
    state = Column(
        pl.Enum(["charge", "discharge", "rest", "unknown"]), "dimensionless", Role.INTENSIVE
    )
    temperature = Column(pl.Float32, "degC", Role.INTENSIVE)
    current = Column(pl.Float32, "mA", Role.EXTENSIVE)
    capacity = Column(pl.Float32, "mAh", Role.EXTENSIVE)
    energy = Column(pl.Float32, "mWh", Role.EXTENSIVE)
    cycle_time = Column(pl.Float64, "s", Role.INTENSIVE)
    cycle_capacity = Column(pl.Float32, "mAh", Role.EXTENSIVE)
    cycle_energy = Column(pl.Float32, "mWh", Role.EXTENSIVE)
    step_time = Column(pl.Float64, "s", Role.INTENSIVE)
    step_capacity = Column(pl.Float32, "mAh", Role.EXTENSIVE)
    step_energy = Column(pl.Float32, "mWh", Role.EXTENSIVE)
    voltage = Column(pl.Float32, "V", Role.INTENSIVE)
    dqdv = Column(pl.Float32, "mAh/V", Role.EXTENSIVE)
    dvdq = Column(pl.Float32, "V/mAh", Role.INVERSE_EXTENSIVE)


class CycleSummaryData(BaseData):
    cycle = Column(pl.UInt16, "dimensionless", Role.INTENSIVE)
    capacity_charge = Column(pl.Float32, "mAh", Role.EXTENSIVE)
    capacity_charge_retention = Column(pl.Float32, "percent", Role.INTENSIVE)
    capacity_discharge = Column(pl.Float32, "mAh", Role.EXTENSIVE)
    capacity_discharge_retention = Column(pl.Float32, "percent", Role.INTENSIVE)
    coulomb_efficiency = Column(pl.Float32, "percent", Role.INTENSIVE)
    energy_charge = Column(pl.Float32, "mWh", Role.EXTENSIVE)
    energy_charge_retention = Column(pl.Float32, "dimensionless", Role.INTENSIVE)
    energy_discharge = Column(pl.Float32, "mWh", Role.EXTENSIVE)
    energy_discharge_retention = Column(pl.Float32, "percent", Role.INTENSIVE)
    energy_efficiency = Column(pl.Float32, "percent", Role.INTENSIVE)


class EISData(BaseData):
    frequency = Column(pl.Float32, "Hz", Role.INTENSIVE)
    re_Z = Column(pl.Float32, "ohm", Role.INVERSE_EXTENSIVE)
    im_Z = Column(pl.Float32, "ohm", Role.INVERSE_EXTENSIVE)
    abs_Z = Column(pl.Float32, "ohm", Role.INVERSE_EXTENSIVE)
    theta = Column(pl.Float32, "rad", Role.INTENSIVE)
