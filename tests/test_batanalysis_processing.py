import numpy as np
import polars as pl
import pytest

from batanalysis import processing
from batanalysis.data import ChargeDischargeData, EISData, State


@pytest.fixture
def charge_discharge_data():
    return ChargeDischargeData(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cycle": [1, 1, 1, 1, 1, 1],
            "current": [0.0, 1000.0, 1000.0, -1000.0, -1000.0, 0.0],
            "voltage": [3.8, 3.7, 3.6, 3.5, 3.4, 3.3],
        }
    )


@pytest.fixture
def differentiation_data():
    return ChargeDischargeData(
        {
            "time": list(range(20)),
            "cycle": [1] * 20,
            "current": [
                0,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                0,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                -1000,
                0,
                0,
            ],
            "voltage": [
                3.80,
                3.75,
                3.70,
                3.68,
                3.66,
                3.64,
                3.62,
                3.60,
                3.58,
                3.57,
                3.55,
                3.53,
                3.51,
                3.49,
                3.47,
                3.45,
                3.43,
                3.41,
                3.39,
                3.38,
            ],
        }
    )


@pytest.fixture
def two_cycle_data():
    return ChargeDischargeData(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "cycle": [1, 1, 1, 1, 2, 2, 2, 2],
            "current": [1000.0, 1000.0, -1000.0, -1000.0, 1000.0, 1000.0, -1000.0, -1000.0],
            "voltage": [3.7, 3.8, 3.5, 3.4, 3.6, 3.7, 3.4, 3.3],
        }
    )


@pytest.fixture
def eis_data():
    return EISData(
        {
            "frequency": [1000.0, 100.0, 10.0],
            "re_Z": [10.0, 8.0, 5.0],
            "im_Z": [0.0, 2.0, 4.0],
        }
    )


def test_detect_states(charge_discharge_data):
    processing.detect_states(charge_discharge_data)

    assert charge_discharge_data.state.to_list() == [
        State.REST,
        State.CHARGE,
        State.CHARGE,
        State.DISCHARGE,
        State.DISCHARGE,
        State.REST,
    ]


def test_detect_steps(charge_discharge_data):
    processing.detect_states(charge_discharge_data)
    processing.detect_steps(charge_discharge_data)

    assert charge_discharge_data.step.to_list() == [0, 1, 1, 2, 2, 3]
    assert charge_discharge_data.step_time.to_list() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_integrate_capacity_and_energy(charge_discharge_data):
    processing.detect_states(charge_discharge_data)
    processing.detect_steps(charge_discharge_data)
    processing.integrate_capacity(charge_discharge_data)
    processing.integrate_energy(charge_discharge_data)

    assert charge_discharge_data.capacity.to_list()[-1] == pytest.approx(0.0, abs=1e-8)
    assert charge_discharge_data.step_capacity.to_list()[2] == pytest.approx(0.27777779, rel=1e-5)
    assert charge_discharge_data.energy.to_list()[-1] == pytest.approx(0.1111111, rel=1e-4)


def test_differentiate_smoke(differentiation_data):
    processing.detect_states(differentiation_data)
    processing.detect_steps(differentiation_data)
    processing.integrate_capacity(differentiation_data)
    processing.differentiate(differentiation_data)

    assert "dqdv" in differentiation_data.table.columns
    assert "dvdq" in differentiation_data.table.columns
    assert differentiation_data.dqdv.is_not_null().any()
    assert differentiation_data.dvdq.is_not_null().any()
    assert np.all(np.isfinite(differentiation_data.dqdv.drop_nulls().to_numpy()))
    assert np.all(np.isfinite(differentiation_data.dvdq.drop_nulls().to_numpy()))


def test_chargedischarge_to_cycle_summary(two_cycle_data):
    processing.detect_states(two_cycle_data)
    processing.detect_steps(two_cycle_data)
    processing.integrate_capacity(two_cycle_data)
    processing.integrate_energy(two_cycle_data)

    summary = processing.chargedischarge_to_cycle(two_cycle_data)

    assert summary.cycle.to_list() == [1, 2]
    assert summary.capacity_charge.to_list() == pytest.approx([0.27777779, 0.27777779], rel=1e-4)
    assert summary.capacity_discharge.to_list() == pytest.approx([0.27777779, 0.27777779], rel=1e-4)
    assert summary.capacity_charge_retention.to_list() == pytest.approx([100.0, 100.0], abs=1e-3)
    assert summary.coulomb_efficiency.to_list() == pytest.approx([100.0, 100.0], abs=1e-3)
    assert summary.energy_efficiency.to_list() == pytest.approx([91.999, 91.780], abs=1e-2)


def test_calc_z_theta(eis_data):
    processing.calc_z_theta(eis_data)

    assert {"abs_Z", "theta"}.issubset(eis_data.table.columns)
    assert np.all(np.isfinite(eis_data.abs_Z.to_numpy()))
    assert np.all(np.isfinite(eis_data.theta.to_numpy()))


def test_calc_dcr_smoke(charge_discharge_data):
    processing.detect_states(charge_discharge_data)
    processing.detect_steps(charge_discharge_data)
    processing.integrate_capacity(charge_discharge_data)

    result = processing.calc_dcr(charge_discharge_data, t_extract=None)

    assert isinstance(result, pl.DataFrame)
    required_cols = {
        "pulse_id",
        "pulse_type",
        "cycle",
        "step",
        "t0",
        "V0",
        "I0",
        "Q0",
        "Δt",
        "ΔV",
        "ΔI",
        "DCR",
        "DCR_raw",
    }
    assert required_cols.issubset(set(result.columns))
