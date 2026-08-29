import numpy as np
import polars as pl
import pytest

from batanalysis.data import ChargeDischargeData
from batanalysis.processing import _detect_dcr_pulses, calc_dcr


def make_data(
    time,
    current,
    voltage=None,
    capacity=None,
    cycle=None,
    step=None,
    step_time=None,
):
    """DCRテスト用の最小データを作る。"""
    n = len(time)

    if voltage is None:
        voltage = np.full(n, 3.0)
    if capacity is None:
        capacity = np.arange(n, dtype=float) * 0.01

    table = {
        "time": time,
        "current": current,
        "voltage": voltage,
        "capacity": capacity,
    }

    if cycle is not None:
        table["cycle"] = cycle
    if step is not None:
        table["step"] = step
    if step_time is not None:
        table["step_time"] = step_time

    return ChargeDischargeData(pl.DataFrame(table))


def detect(
    data,
    *,
    threshold=0.1,
    current_eps=0.01,
    first_point_elapsed=0.01,
    current_transient_time=0.1,
):
    return _detect_dcr_pulses(
        data,
        threshold=threshold,
        current_eps=current_eps,
        first_point_elapsed=first_point_elapsed,
        current_transient_time=current_transient_time,
    )


# ----------------------------------------------------------------------
# pulse detection
# ----------------------------------------------------------------------


def test_detect_ideal_rectangular_pulse():
    data = make_data(
        time=[0.0, 1.01, 1.10, 1.20, 1.30, 1.40],
        current=[0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    )

    result = detect(data)

    pulse = result.filter(pl.col("pulse_id") == 1)

    assert pulse.height == 4
    assert pulse["t0"][0] == pytest.approx(1.0)
    assert pulse["I0"][0] == pytest.approx(0.0)


def test_detect_slow_current_transient_as_one_pulse():
    """transient内の多段電流変化は1つのpulseとして扱う。"""
    data = make_data(
        time=[0.0, 1.01, 1.04, 1.07, 1.10, 1.20, 1.30, 1.40],
        current=[0.0, 0.25, 0.75, 0.95, 1.0, 1.0, 1.0, 0.0],
    )

    result = detect(data)

    pulse = result.filter(pl.col("pulse_id") == 1)

    assert pulse.height == 6
    assert pulse["pulse_id"].n_unique() == 1
    assert pulse["t0"][0] == pytest.approx(1.0)


def test_small_reverse_during_transient_is_allowed():
    """current_eps以内の逆方向変動は許容する。"""
    data = make_data(
        time=[0.0, 1.01, 1.04, 1.06, 1.08, 1.10, 1.20],
        current=[0.0, 0.2, 0.7, 0.698, 0.9, 1.0, 1.0],
    )

    result = detect(data, current_eps=0.005)

    assert not result.is_empty()


def test_large_reverse_during_transient_is_rejected():
    """current_epsを超えて逆方向へ動く過渡応答は除外する。"""
    data = make_data(
        time=[0.0, 1.01, 1.04, 1.06, 1.08, 1.10, 1.20],
        current=[0.0, 0.2, 0.7, 0.69, 0.9, 1.0, 1.0],
    )

    result = detect(data, current_eps=0.005)

    assert result.is_empty()


def test_pulse_shorter_than_transient_time_is_rejected():
    data = make_data(
        time=[0.0, 1.01, 1.05, 1.08, 1.20],
        current=[0.0, 1.0, 1.0, 0.0, 0.0],
    )

    result = detect(data)

    assert result.is_empty()


def test_pulse_does_not_restart_after_leaving_constant_current_range():
    """
    一度定電流範囲を外れた後、同じ電流値へ戻っても
    同じpulseには復帰しない。
    """
    data = make_data(
        time=[0.0, 1.01, 1.10, 1.20, 1.30, 1.40],
        current=[0.0, 1.0, 1.0, 0.98, 1.0, 1.0],
    )

    result = detect(
        data,
        current_eps=0.01,
    )

    pulse = result.filter(pl.col("pulse_id") == 1)

    assert pulse["time"].to_list() == pytest.approx([1.01, 1.10])


def test_t0_uses_first_point_elapsed_without_step_information():
    data = make_data(
        time=[0.0, 1.05, 1.10, 1.20],
        current=[0.0, 1.0, 1.0, 1.0],
    )

    result = detect(
        data,
        first_point_elapsed=0.05,
        current_transient_time=0.1,
    )

    assert result["t0"][0] == pytest.approx(1.0)


def test_t0_uses_step_time_at_program_step_start():
    data = make_data(
        time=[0.0, 1.03, 1.08, 1.13, 1.20],
        current=[0.0, 1.0, 1.0, 1.0, 1.0],
        cycle=[1, 1, 1, 1, 1],
        step=[0, 1, 1, 1, 1],
        step_time=[0.0, 0.03, 0.08, 0.13, 0.20],
    )

    result = detect(
        data,
        first_point_elapsed=0.01,
        current_transient_time=0.1,
    )

    assert result["t0"][0] == pytest.approx(1.0)
    assert result["cycle"][0] == 1
    assert result["step"][0] == 1


def test_q0_is_capacity_before_pulse():
    data = make_data(
        time=[0.0, 1.01, 1.10, 1.20],
        current=[0.0, 1.0, 1.0, 1.0],
        capacity=[0.123, 0.124, 0.125, 0.126],
    )

    result = detect(data)

    assert result["Q0"][0] == pytest.approx(0.123)


# ----------------------------------------------------------------------
# calc_dcr
# ----------------------------------------------------------------------


def test_calc_dcr_uses_measured_current_during_transient():
    """
    ΔIにはreference currentではなく各時刻の実測電流を使用する。
    """
    data = make_data(
        time=[0.0, 1.01, 1.05, 1.10, 1.20],
        current=[0.0, 0.25, 0.75, 1.0, 1.0],
        voltage=[3.0, 3.0025, 3.0075, 3.010, 3.010],
    )

    result = calc_dcr(
        data,
        current_eps=0.01,
        first_point_elapsed=0.01,
        current_transient_time=0.1,
    )

    assert result["ΔI"].to_list() == pytest.approx([0.25, 0.75, 1.0, 1.0])

    assert result["DCR"].to_list() == pytest.approx(
        [10.0] * 4,
        rel=1e-4,
    )


def test_calc_dcr_interpolates_requested_time():
    data = make_data(
        time=[0.0, 1.01, 1.10, 1.20, 1.30],
        current=[0.0, 1.0, 1.0, 1.0, 1.0],
        voltage=[3.0, 3.001, 3.010, 3.020, 3.030],
    )

    result = calc_dcr(
        data,
        t_extract=[0.15],
        current_eps=0.01,
        first_point_elapsed=0.01,
        current_transient_time=0.1,
    )

    assert result.height == 1
    assert result["Δt"][0] == pytest.approx(0.15)
    assert result["ΔI"][0] == pytest.approx(1.0)
    assert result["ΔV"][0] == pytest.approx(0.015)
    assert result["DCR"][0] == pytest.approx(15.0)


def test_calc_dcr_uses_last_point_within_extract_tolerance():
    data = make_data(
        time=[0.0, 1.01, 1.10, 1.18, 1.19],
        current=[0.0, 1.0, 1.0, 1.0, 0.0],
        voltage=[3.0, 3.01, 3.01, 3.018, 3.0],
    )

    result = calc_dcr(
        data,
        t_extract=[0.20],
        first_point_elapsed=0.01,
        current_transient_time=0.1,
        extract_time_tolerance=0.03,
    )

    assert result.height == 1
    assert result["Δt"][0] == pytest.approx(0.18)
    assert result["Δt_nearest"][0] == pytest.approx(0.18)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("first_point_elapsed", -0.01),
        ("current_transient_time", -0.01),
        ("extract_time_tolerance", -0.01),
    ],
)
def test_calc_dcr_rejects_negative_time_parameters(argument, value):
    data = make_data(
        time=[0.0, 1.0],
        current=[0.0, 1.0],
        voltage=[3.0, 3.01],
    )

    with pytest.raises(ValueError):
        calc_dcr(
            data,
            **{argument: value},
        )


@pytest.mark.parametrize(
    "t_extract",
    [
        [],
        [-0.1],
        [np.inf],
        [np.nan],
    ],
)
def test_calc_dcr_rejects_invalid_extract_times(t_extract):
    data = make_data(
        time=[0.0, 1.0],
        current=[0.0, 1.0],
        voltage=[3.0, 3.01],
    )

    with pytest.raises(ValueError):
        calc_dcr(
            data,
            t_extract=t_extract,
        )
