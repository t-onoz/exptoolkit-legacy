import numpy as np
import polars as pl
import pytest

from batanalysis.data import ChargeDischargeData, State
from batanalysis.processing import ChargeDischargeFeaturizer


def _make_charge_discharge_data(
    *,
    initial_rest: bool,
) -> ChargeDischargeData:
    """Create plausible charge-discharge data for one cycle."""
    cls = ChargeDischargeData

    time_parts = []
    current_parts = []
    voltage_parts = []
    state_parts = []

    t0 = 0.0

    def add_step(
        duration: float,
        current: float,
        voltage: np.ndarray,
        state: str,
    ) -> None:
        nonlocal t0

        n = len(voltage)
        time = t0 + np.linspace(0.0, duration, n)

        time_parts.append(time)
        current_parts.append(np.full(n, current))
        voltage_parts.append(voltage)
        state_parts.append(np.full(n, state))

        # Prevent the timestamps from overlapping with the next step.
        t0 = time[-1] + 1.0

    if initial_rest:
        # Stable initial rest near 3.0 V.
        add_step(
            duration=60.0,
            current=0.0,
            voltage=np.linspace(3.00, 3.01, 21),
            state=State.REST,
        )

    q = np.linspace(0.0, 1.0, 201)

    # Approximately 1 mAh of charge:
    # 1 mA × 1 h
    #
    # A plausible nonlinear curve rising from 3.0 V to 4.2 V.
    charge_v = 3.0 + 0.85 * q + 0.35 * q**4

    add_step(
        duration=3600.0,
        current=1.0,
        voltage=charge_v,
        state=State.CHARGE,
    )

    # IR drop immediately after charging, followed by slight relaxation.
    rest_charge_t = np.linspace(0.0, 1.0, 51)
    rest_after_charge_v = 4.15 - 0.05 * (1.0 - np.exp(-5.0 * rest_charge_t))

    add_step(
        duration=300.0,
        current=0.0,
        voltage=rest_after_charge_v,
        state=State.REST,
    )

    # Approximately 0.9 mAh of discharge:
    # -0.9 mA × 1 h
    #
    # Falling from 4.1 V to 3.0 V.
    discharge_v = 4.10 - 0.75 * q - 0.35 * q**4

    add_step(
        duration=3600.0,
        current=-0.9,
        voltage=discharge_v,
        state=State.DISCHARGE,
    )

    # After discharge, the voltage relaxes upward.
    rest_discharge_t = np.linspace(0.0, 1.0, 51)
    rest_after_discharge_v = 3.05 + 0.05 * (1.0 - np.exp(-5.0 * rest_discharge_t))

    add_step(
        duration=300.0,
        current=0.0,
        voltage=rest_after_discharge_v,
        state=State.REST,
    )

    time = np.concatenate(time_parts)
    current = np.concatenate(current_parts)
    voltage = np.concatenate(voltage_parts)
    state = np.concatenate(state_parts)

    table = pl.DataFrame(
        {
            cls.time.name: time,
            cls.cycle.name: np.ones(len(time), dtype=np.int64),
            cls.current.name: current,
            cls.voltage.name: voltage,
            cls.state.name: state,
        }
    )

    return cls(table)


@pytest.mark.parametrize("initial_rest", [False, True])
def test_charge_discharge_featurizer(initial_rest: bool) -> None:
    data = _make_charge_discharge_data(
        initial_rest=initial_rest,
    )

    featurizer = ChargeDischargeFeaturizer()

    values = featurizer(data)

    assert values.shape == (len(featurizer.feature_names),)
    assert np.all(np.isfinite(values))

    features = dict(
        zip(
            featurizer.feature_names,
            values,
        )
    )

    # 1 mA × 1 h ≈ 1 mAh.
    assert features["charge_capacity"] == pytest.approx(
        1.0,
        rel=0.01,
    )

    # 0.9 mA × 1 h ≈ 0.9 mAh.
    assert features["discharge_capacity"] == pytest.approx(
        0.9,
        rel=0.01,
    )

    # Energy should also be positive and roughly equal to capacity × representative voltage.
    assert features["charge_energy"] > 3.0
    assert features["discharge_energy"] > 2.5

    # The charge curve rises with q.
    assert features["charge_v_q0.100"] < features["charge_v_q0.500"] < features["charge_v_q0.900"]

    # The discharge curve falls with q.
    assert (
        features["discharge_v_q0.100"]
        > features["discharge_v_q0.500"]
        > features["discharge_v_q0.900"]
    )

    # The sign of dV/dq is also as expected.
    assert features["charge_dvdq_q0.500"] > 0
    assert features["discharge_dvdq_q0.500"] < 0

    # The voltage drops after charging.
    assert features["post_charge_rest_delta_v_first"] < 0
    assert features["post_charge_rest_delta_v_end"] < 0

    # The voltage rises after discharge.
    assert features["post_discharge_rest_delta_v_first"] > 0
    assert features["post_discharge_rest_delta_v_end"] > 0
