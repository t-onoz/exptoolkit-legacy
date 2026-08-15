import polars as pl

from batanalysis.data import ChargeDischargeData, EISData


def _make_charge_discharge_data():
    return ChargeDischargeData(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cycle": [1, 1, 1, 1, 1, 1],
            "current": [0.0, 1000.0, 1000.0, -1000.0, -1000.0, 0.0],
            "voltage": [3.8, 3.7, 3.6, 3.5, 3.4, 3.3],
        }
    )


def _make_two_cycle_data():
    return ChargeDischargeData(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "cycle": [1, 1, 1, 1, 2, 2, 2, 2],
            "current": [1000.0, 1000.0, -1000.0, -1000.0, 1000.0, 1000.0, -1000.0, -1000.0],
            "voltage": [3.7, 3.8, 3.5, 3.4, 3.6, 3.7, 3.4, 3.3],
        }
    )


def _make_eis_data():
    return EISData(
        {
            "frequency": [1000.0, 100.0, 10.0],
            "re_Z": [10.0, 8.0, 5.0],
            "im_Z": [0.0, 2.0, 4.0],
        }
    )


def test_core_data_factories_build_valid_objects():
    data = _make_charge_discharge_data()
    two_cycle = _make_two_cycle_data()
    eis = _make_eis_data()

    assert data.table.shape[0] == 6
    assert two_cycle.table["cycle"].unique().to_list() == [1, 2]
    assert set(eis.table.columns) >= {"frequency", "re_Z", "im_Z"}
    assert data.current.dtype == pl.Float32
