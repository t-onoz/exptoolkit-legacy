from __future__ import annotations

import math
from collections.abc import Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from batanalysis._savgol import savgol_filter_np
from batanalysis.data import ChargeDischargeData, CycleSummaryData, EISData, State
from exptoolkit.processing import Converter, Modifier

logger = getLogger()


def _cumulative_trapezoid(y: pl.Expr, x: pl.Expr, x0: None | float = None) -> pl.Expr:
    width = x - x.shift(fill_value=(x.first() if x0 is None else x0))
    height = y + y.shift(fill_value=y.first())
    return 0.5 * (width * height).cum_sum()


def detect_states(data: ChargeDischargeData, atol=1e-6, rtol=1e-4) -> None:
    """detects states based on current.

    I < -tolerance : discharge
    |I| <= tolerance: rest
    I > tolerance: charge
    """
    cls = ChargeDischargeData
    i_max = cast(float, data.current.abs().max() or 0.0)
    tolerance = max(i_max * rtol, atol)
    dtype = cls.state.dtype
    data.state = data.table.select(
        pl.coalesce(
            pl.when(cls.current.expr < -tolerance).then(pl.lit(State.DISCHARGE, dtype=dtype)),
            pl.when(cls.current.expr > tolerance).then(pl.lit(State.CHARGE, dtype=dtype)),
            pl.lit(State.REST, dtype=dtype),
        )
        .cast(dtype)
        .alias(cls.state.name)
    ).to_series()


def detect_steps(
    data: ChargeDischargeData,
    recalc_time: bool = True,
    *,
    first_point_elapsed: float = 0.0,
) -> None:
    cls = ChargeDischargeData

    if not data.is_col_ready("state"):
        detect_states(data)

    st = cls.state.expr.shift(fill_value=cls.state.expr.first()) != cls.state.expr
    cy = cls.cycle.expr.shift(fill_value=cls.cycle.expr.first()) != cls.cycle.expr

    data.step = data.table.select((st | cy).cum_sum().alias(cls.step.name)).to_series()

    if recalc_time:
        data.table = data.table.with_columns(
            (cls.time.expr - cls.time.expr.first().over(cls.step.expr) + first_point_elapsed).alias(
                cls.step_time.name
            ),
            (
                cls.time.expr - cls.time.expr.first().over(cls.cycle.expr) + first_point_elapsed
            ).alias(cls.cycle_time.name),
        )


def integrate_capacity(
    data: ChargeDischargeData, *, skip_step=False, skip_cycle=False, skip_total=False
) -> None:
    cls = ChargeDischargeData
    if not data.is_col_ready("step"):
        detect_steps(data)
    exprs = []
    if not skip_step:
        exprs.append(
            (
                _cumulative_trapezoid(cls.current.expr, cls.step_time.expr / 3600.0, x0=0.0)
                .over(cls.step.expr, cls.cycle.expr)
                .cast(cls.step_capacity.dtype)
                * pl.when(cls.state.expr == State.DISCHARGE)
                .then(pl.lit(-1.0))
                .otherwise(pl.lit(1.0))
            ).alias(cls.step_capacity.name)
        )
    if not skip_cycle:
        exprs.append(
            _cumulative_trapezoid(cls.current.expr, cls.cycle_time.expr / 3600.0, x0=0.0)
            .over(cls.cycle.expr)
            .cast(cls.cycle_capacity.dtype)
            .alias(cls.cycle_capacity.name)
        )
    if not skip_total:
        exprs.append(
            _cumulative_trapezoid(cls.current.expr, cls.time.expr / 3600, x0=0.0)
            .cast(cls.capacity.dtype)
            .alias(cls.capacity.name)
        )

    if exprs:
        data.table = data.table.with_columns(*exprs)


def integrate_energy(
    data: ChargeDischargeData, *, skip_step=False, skip_cycle=False, skip_total=False
) -> None:
    cls = ChargeDischargeData

    step_ready = skip_step or data.is_col_ready(cls.step_capacity)
    cycle_ready = skip_cycle or data.is_col_ready(cls.cycle_capacity)
    total_ready = skip_total or data.is_col_ready(cls.capacity)

    if not (step_ready and cycle_ready and total_ready):
        integrate_capacity(
            data,
            skip_step=step_ready,
            skip_cycle=cycle_ready,
            skip_total=total_ready,
        )

    exprs = []
    if not skip_step:
        exprs.append(
            _cumulative_trapezoid(cls.voltage.expr, cls.step_capacity.expr, x0=0.0)
            .over(cls.step.expr, cls.cycle.expr)
            .cast(cls.step_energy.dtype)
            .alias(cls.step_energy.name)
        )
    if not skip_cycle:
        exprs.append(
            _cumulative_trapezoid(cls.voltage.expr, cls.cycle_capacity.expr, x0=0.0)
            .over(cls.cycle.expr)
            .cast(cls.cycle_energy.dtype)
            .alias(cls.cycle_energy.name)
        )
    if not skip_total:
        exprs.append(
            _cumulative_trapezoid(cls.voltage.expr, cls.capacity.expr, x0=0.0)
            .cast(cls.energy.dtype)
            .alias(cls.energy.name)
        )

    if exprs:
        data.table = data.table.with_columns(*exprs)


def differentiate(
    data: ChargeDischargeData, window_in_volt: float = 0.02, polyorder: int = 2
) -> None:
    """calculates dq/dv and dv/dq using Savitzky-Golay algorithm.

    Args:
        window_in_volt: window width for smoothing. recommended value is between 0.01-0.02.
        polyorder: order of the polynomial"""
    cls = ChargeDischargeData
    if not data.is_col_ready(cls.step_capacity.name):
        integrate_capacity(data)
    data.table = data.table.group_by(cls.step.name, cls.cycle.name, maintain_order=True).map_groups(
        lambda g: _differentiate_step(g, window_in_volt, polyorder)
    )


def _differentiate_step(g: pl.DataFrame, window_in_volt, polyorder) -> pl.DataFrame:
    """Called from `polars.DataFrame.group_by(...).map_groups()`."""
    cls = ChargeDischargeData

    def _none():
        return g.with_columns(
            pl.lit(None, dtype=cls.dqdv.dtype).alias(cls.dqdv.name),
            pl.lit(None, dtype=cls.dvdq.dtype).alias(cls.dvdq.name),
        )

    cycle, step = g.select(
        pl.col(cls.cycle.name).first(),
        pl.col(cls.step.name).first(),
    ).row(0)

    try:
        if not g[cls.state.name].is_in(["charge", "discharge"]).all():
            return _none()

        q = g[cls.step_capacity.name].to_numpy()
        v = g[cls.voltage.name].to_numpy()
        dqdv_full = np.full_like(v, np.nan)
        dvdq_full = np.full_like(v, np.nan)

        # exclude constant voltage region
        mask = np.abs(v - v[-1]) >= 0.005
        v_ = v[mask]
        q_ = q[mask]

        # should be placed before np.nanmax(v_)
        # because if v_ is empty, np.nanmax(v_) will raise ValueError.
        if len(v_) < 2:
            logger.warning(
                "Skipping differentiation for (cycle, step) = (%s, %s) "
                "because there are not enough data points (%s)",
                cycle,
                step,
                len(v_),
            )
            return _none()

        # set window length based on voltage range and number of data points
        v_span = np.nanmax(v_) - np.nanmin(v_)
        try:
            wl = int(len(v_) * window_in_volt / v_span)
        except (OverflowError, ZeroDivisionError, ValueError):
            wl = 0

        if wl % 2 == 0:
            wl = wl - 1

        if 0 < polyorder < wl <= len(v_):
            logger.info("window_length of (cycle, step) = (%s, %s): %s", cycle, step, wl)
            dq = savgol_filter_np(q_, window_length=wl, polyorder=polyorder, deriv=1)
            dv = savgol_filter_np(v_, window_length=wl, polyorder=polyorder, deriv=1)
        else:
            logger.warning(
                "Disable smoothing for (cycle, step) = (%s, %s) "
                "because window_length is too large/small (%s)",
                cycle,
                step,
                wl,
            )
            dq = np.gradient(q_)
            dv = np.gradient(v_)

        dqdv_full[mask] = dq / dv
        dvdq_full[mask] = dv / dq
        return g.with_columns(
            pl.Series(cls.dqdv.name, dqdv_full, dtype=cls.dqdv.dtype).fill_nan(None),
            pl.Series(cls.dvdq.name, dvdq_full, dtype=cls.dvdq.dtype).fill_nan(None),
        )
    except Exception:
        # Catch all exceptions because map_groups hides the original traceback.
        logger.exception(
            "Failed to differentiate for (cycle, step) = (%s, %s)", cycle, step, exc_info=True
        )
        return _none()


def chargedischarge_to_cycle(
    data: ChargeDischargeData,
    base: Literal["first", "max"] = "first",
    copy_metadata: bool = True,
) -> CycleSummaryData:
    """_summary_

    Args:
        data (ChargeDischargeData): 充放電データ
        base (Literal['first', 'max'], optional):
            'fisrt'の場合、初回サイクルに対する維持率を計算する。
            'max'の場合、最大値に対する維持率を計算する。. Defaults to 'first'.

    Returns:
        CycleSummaryData:
    """
    cdd = ChargeDischargeData
    csd = CycleSummaryData

    def _ret(expr: pl.Expr) -> pl.Expr:
        if base == "first":
            return expr / expr.first(ignore_nulls=True)
        return expr / expr.max()

    new_table = (
        data.table
        # ------------------------------------------------------------
        # 充電・放電ステップのみ使用（restなどを除外）
        # ------------------------------------------------------------
        .filter(cdd.state.expr.is_in([State.CHARGE, State.DISCHARGE]))
        # ------------------------------------------------------------
        # 各stepの最終値を取得
        #    step_capacity / step_energy は通常累積値なので
        #    stepごとの最終行を取ることで step全体の値を得る
        # ------------------------------------------------------------
        .group_by(cdd.cycle.expr, cdd.step.expr, maintain_order=True)
        .last()
        # ------------------------------------------------------------
        # step → (cycle, state) に集約
        #    1 cycle 内の charge / discharge の
        #    capacity と energy をそれぞれ合計
        # ------------------------------------------------------------
        .group_by(cdd.cycle.expr, cdd.state.expr, maintain_order=True)
        .agg(
            cdd.step_capacity.expr.sum().alias("capacity"),
            cdd.step_energy.expr.sum().alias("energy"),
        )
        # ------------------------------------------------------------
        # cycle内の charge / discharge の値を横持ち列として取得
        # ------------------------------------------------------------
        .pivot(
            on=cdd.state.name,
            on_columns=[State.CHARGE, State.DISCHARGE],
            index=cdd.cycle.name,
            values=["capacity", "energy"],
        )
        # --------------------------------------------------------
        # 容量保持率 (retention)
        #
        #    各 state (charge / discharge) ごとに
        #    基準容量 (first または max) を計算し
        #    それに対する比をとる
        # --------------------------------------------------------
        .with_columns(
            (100.0 * _ret(csd.capacity_charge.expr)).alias(csd.capacity_charge_retention.name),
            (100.0 * _ret(csd.capacity_discharge.expr)).alias(
                csd.capacity_discharge_retention.name
            ),
            (100.0 * _ret(csd.energy_charge.expr)).alias(csd.energy_charge_retention.name),
            (100.0 * _ret(csd.energy_discharge.expr)).alias(csd.energy_discharge_retention.name),
        )
        # ------------------------------------------------------------
        # 効率計算
        #
        #    coulomb efficiency  = discharge capacity / charge capacity
        #    energy efficiency   = discharge energy   / charge energy
        # ------------------------------------------------------------
        .with_columns(
            (100.0 * (csd.capacity_discharge.expr / csd.capacity_charge.expr)).alias(
                csd.coulomb_efficiency.name
            ),
            (100.0 * (csd.energy_discharge.expr / csd.energy_charge.expr)).alias(
                csd.energy_efficiency.name
            ),
        )
    )
    return CycleSummaryData(
        new_table,
        normalization=data.norm,
        metadata=data.metadata.copy() if copy_metadata else None,
    )


def calc_dcr(
    data: ChargeDischargeData,
    t_extract: list[float] | Literal["last"] | None = None,
    threshold: float = 0.1,
    current_eps: float = 1e-4,
    first_point_elapsed: float = 0.01,
    current_transient_time: float = 0.5,
    extract_time_tolerance: float = 0.1,
) -> pl.DataFrame:
    """
    電流パルスを検出し、DCRを計算する。

    Parameters
    ----------
    data
        充放電データ。
    t_extract
        DCRを抽出する経過時間 (s)。

        - ``list[float]``: 指定時刻のDCRを返す。
        - ``"last"``: 各パルスの最終データ点のDCRを返す。
        - ``None``: パルス内の全データ点についてDCRを返す。

    threshold
        パルス開始を検出する最小電流変化量 (mA/[amount])。
    current_eps
        電流が一定または0とみなす許容値 (mA/[amount])。
    first_point_elapsed
        電流切替から最初のデータ点までの経過時間 (s)。
        step情報から開始時刻を求められない場合の推定に使用する。
    current_transient_time
        電流切替後の過渡応答時間の上限 (s)。
        この時間内の電流変化は同一パルスの過渡応答として扱い、
        この時間以降は電流が定常値に達しているものとみなす。
    extract_time_tolerance
        指定した抽出時刻をパルス終端がわずかに下回る場合の
        許容時間 (s)。

    Returns
    -------
    pl.DataFrame
        DCR計算結果。以下の列を含む。

        - ``pulse_id``: パルス番号。
        - ``pulse_type``: パルス種別。
        - ``cycle``: パルス開始時のサイクル番号。
        - ``step``: パルス開始時のステップ番号。
        - ``t0``: 推定したパルス開始時刻 (s)。
        - ``V0``: パルス開始前の基準電圧 (V)。
        - ``I0``: パルス開始前の基準電流 (mA/[amount])。
        - ``Q0``: パルス開始前の基準容量 (mAh/[amount])。
        - ``Δt``: パルス開始からの経過時間 (s)。
        - ``ΔI``: 基準電流からの変化量 (mA/[amount])。
        - ``ΔV``: 基準電圧からの変化量 (V)。
        - ``DCR``: 正規化されたDCR (Ω・[amount])。
        - ``DCR_raw``: 正規化前のDCR (Ω)。
        - ``Δt_nearest``: 補間点に最も近い実測点の経過時間 (s)。
        ``t_extract`` に時刻を指定した場合のみ含まれる。

    Notes
    -----
    DCRの計算には ``time``, ``current``, ``voltage`` が必要である。
    ``capacity`` はDCR計算には使用せず、パルス開始前の値を
    ``Q0`` として結果に付加する。

    パルス開始候補は、直前のデータ点からの電流変化が
    ``threshold`` を超えた点とする。パルス開始後
    ``current_transient_time`` 以内の追加の電流変化は同一パルスの
    過渡応答として扱い、新しいパルスとして数えない。

    ``step``, ``cycle``, ``step_time`` が利用可能な場合は、
    測定プログラムの切替情報からパルス開始時刻 ``t0`` を求める。
    利用できない場合は、最初のパルス点から ``first_point_elapsed`` を
    差し引いて ``t0`` を推定する。

    過渡区間では、電流が定電流値に向かってほぼ単調に変化することを
    要求する。``current_transient_time`` 経過後の最初の実測電流値を
    定電流域の基準とし、そこからの差が ``current_eps`` 未満である
    連続区間をパルスとして使用する。``current_transient_time`` より短い
    パルスは除外する。

    ``ΔI`` は定電流域の基準値ではなく、各データ点の実測電流と
    パルス開始前の電流 ``I0`` との差として計算する。
    DCRは各データ点について ``ΔV / ΔI`` として計算する。

    ``t_extract`` に時刻を指定した場合、``ΔI`` と ``ΔV`` は
    時間に対して線形補間する。指定時刻がパルス終端を超える場合でも、
    その差が ``extract_time_tolerance`` 以下であれば最終データ点を使用する。
    """

    cls = ChargeDischargeData

    # 引数と計算に必須の列を検証する
    if first_point_elapsed < 0:
        raise ValueError("first_point_elapsed must be non-negative.")
    if current_transient_time < 0:
        raise ValueError("current_transient_time must be non-negative.")
    if extract_time_tolerance < 0:
        raise ValueError("extract_time_tolerance must be non-negative.")

    if isinstance(t_extract, list):
        if not t_extract:
            raise ValueError("t_extract is empty.")
        if any(not math.isfinite(t) or t < 0 for t in t_extract):
            raise ValueError("t_extract must contain finite non-negative values.")

    required_columns = [
        cls.time,
        cls.current,
        cls.voltage,
    ]
    missing_columns = [column.name for column in required_columns if not data.is_col_ready(column)]
    if missing_columns:
        raise ValueError("Required columns are not ready: " + ", ".join(missing_columns))

    output_columns = [
        "pulse_id",
        "pulse_type",
        "cycle",
        "step",
        "t0",
        "V0",
        "I0",
        "Q0",
        "Δt",
        "ΔI",
        "ΔV",
        "DCR",
        "DCR_raw",
    ]
    if isinstance(t_extract, list):
        output_columns.append("Δt_nearest")

    # 矩形電流パルスを検出する
    df = _detect_dcr_pulses(
        data,
        threshold=threshold,
        current_eps=current_eps,
        first_point_elapsed=first_point_elapsed,
        current_transient_time=current_transient_time,
    )

    # パルス開始点を基準とした変化量を求める
    df = df.with_columns(
        (cls.time.expr - pl.col("t0")).alias("Δt"),
        (cls.current.expr - pl.col("I0")).alias("ΔI"),
        (cls.voltage.expr - pl.col("V0")).alias("ΔV"),
    )

    # 指定時刻のデータを抽出する
    if t_extract == "last":
        df = df.group_by(
            "pulse_id",
            maintain_order=True,
        ).last()

    elif isinstance(t_extract, list):
        target_times = pl.DataFrame({"t_star": sorted(set(t_extract))})

        df = (
            df.sort("pulse_id", "Δt")
            .join(target_times, how="cross")
            .filter(
                pl.col("Δt").max().over("pulse_id") >= pl.col("t_star") - extract_time_tolerance
            )
        )

        if df.height == 0:
            df = df.with_columns(
                pl.lit(
                    None,
                    dtype=df.schema["Δt"],
                ).alias("Δt_nearest")
            )
        else:
            df = (
                df.group_by(
                    "pulse_id",
                    "t_star",
                    maintain_order=True,
                )
                .map_groups(_interpolate_dcr)
                .sort("pulse_id", "t_star")
            )

    # DCRを計算し、出力を整える
    if data.norm.amount is None or not math.isfinite(data.norm.amount):
        norm_factor = 1.0
    else:
        norm_factor = data.norm.amount

    return (
        df.with_columns(
            (pl.col("ΔV") / pl.col("ΔI") * 1000).alias("DCR"),
        )
        .with_columns(
            (pl.col("DCR") / norm_factor).alias("DCR_raw"),
            pl.coalesce(
                pl.when(cls.current.expr.abs() < current_eps).then(pl.lit("relax")),
                pl.when(cls.current.expr > pl.col("I0")).then(pl.lit("pulse(+)")),
                pl.when(cls.current.expr < pl.col("I0")).then(pl.lit("pulse(-)")),
            )
            .alias("pulse_type")
            .cast(pl.Enum(["relax", "pulse(+)", "pulse(-)"])),
        )
        .select(output_columns)
    )


def _detect_dcr_pulses(
    data: ChargeDischargeData,
    threshold: float,
    current_eps: float,
    first_point_elapsed: float,
    current_transient_time: float,
) -> pl.DataFrame:
    """
    電流波形から矩形パルス区間を検出する。

    ``threshold`` を超える電流変化をパルス開始候補とする。
    パルス開始後 ``current_transient_time`` 以内の追加の電流変化は
    同一パルスの過渡応答として扱い、新しいパルスとして数えない。

    過渡区間では、電流が定電流値に向かってほぼ単調に変化することを
    要求する。``current_transient_time`` 経過後の最初の電流値を基準とし、
    そこからの差が ``current_eps`` 未満である連続区間をパルスとする。

    付与列: pulse_id, cycle, step, t0, V0, I0, Q0

    Notes
    -----
    - ``current_transient_time`` は推定された ``t0`` を基準とする。
    - ``current_transient_time`` より短いパルスは除外する。
    - 過渡区間では ``current_eps`` までの逆方向変動を許容する。

                t0                           t0 + current_transient_time
                |                                   |
                v                                   v
        I       0 ---- 0.2 ---- 0.7 ---- 0.95 ---- 1.0 ---- 1.0
                |<----- current_transient_time ---->|
                |        電流は変化していてよい       | ここから定電流と仮定
    """
    cls = ChargeDischargeData
    df = data.table

    # パルス開始候補と、その候補に対するt0を求める
    step_ready = data.is_col_ready(cls.step)
    cycle_ready = data.is_col_ready(cls.cycle)
    step_time_ready = data.is_col_ready(cls.step_time)

    if step_ready and cycle_ready:
        is_program_step_start = (cls.step.expr != cls.step.expr.shift()) | (
            cls.cycle.expr != cls.cycle.expr.shift()
        )
    elif step_ready:
        is_program_step_start = cls.step.expr != cls.step.expr.shift()
    elif cycle_ready:
        is_program_step_start = cls.cycle.expr != cls.cycle.expr.shift()
    else:
        is_program_step_start = pl.lit(False)

    if step_time_ready and (step_ready or cycle_ready):
        start_offset = (
            pl.when(is_program_step_start & cls.step_time.expr.is_not_null())
            .then(cls.step_time.expr)
            .otherwise(first_point_elapsed)
        )
    else:
        start_offset = pl.lit(first_point_elapsed)

    df = (
        df.with_row_index("_row_index")
        .with_columns(
            cls.current.expr.shift().alias("_I_prev"),
            cls.voltage.expr.shift().alias("_V_prev"),
            cls.capacity.expr.shift().alias("_Q_prev"),
        )
        .with_columns(
            ((cls.current.expr - pl.col("_I_prev")).abs() > threshold).alias("_is_candidate"),
            (cls.time.expr - start_offset).alias("_candidate_t0"),
        )
    )

    candidates = df.filter(pl.col("_is_candidate")).select(
        "_row_index",
        cls.time.name,
        "_candidate_t0",
    )

    # transient中の候補を除外する。
    # Pythonで走査するのは全データではなく、電流変化候補だけ。
    accepted_indices: list[int] = []
    accepted_t0: list[float] = []

    last_t0: float | None = None

    for index, time, t0 in candidates.iter_rows():
        if last_t0 is not None and time <= last_t0 + current_transient_time:
            continue

        accepted_indices.append(index)
        accepted_t0.append(t0)
        last_t0 = t0

    # 採用した開始点を元データへ戻す
    is_pulse_start = np.zeros(df.height, dtype=bool)
    t0_start = np.full(df.height, np.nan)

    if accepted_indices:
        is_pulse_start[accepted_indices] = True
        t0_start[accepted_indices] = accepted_t0

    df = (
        df.with_columns(
            pl.Series("_is_pulse_start", is_pulse_start),
            pl.Series("_t0_start", t0_start),
        )
        .with_columns(
            pl.col("_is_pulse_start").cum_sum().cast(pl.UInt32).alias("pulse_id"),
            pl.when("_is_pulse_start").then("_t0_start").forward_fill().alias("t0"),
            pl.when("_is_pulse_start").then("_V_prev").forward_fill().alias("V0"),
            pl.when("_is_pulse_start").then("_I_prev").forward_fill().alias("I0"),
            pl.when("_is_pulse_start").then("_Q_prev").forward_fill().alias("Q0"),
            pl.when(pl.col("_is_pulse_start")).then(cls.cycle.expr).forward_fill().alias("cycle"),
            pl.when(pl.col("_is_pulse_start")).then(cls.step.expr).forward_fill().alias("step"),
        )
        .filter(pl.col("pulse_id") > 0)
        .with_columns(
            (cls.time.expr - pl.col("t0")).alias("_Δt"),
        )
    )

    # transient終了後の最初の点を定電流域の基準とする
    df = df.with_columns(
        pl.when(pl.col("_Δt") >= current_transient_time)
        .then(cls.current.expr)
        .alias("_reference_current_candidate"),
        pl.when(pl.col("_Δt") >= current_transient_time)
        .then(pl.col("_Δt"))
        .alias("_reference_time_candidate"),
    ).with_columns(
        pl.col("_reference_current_candidate")
        .drop_nulls()
        .first()
        .over("pulse_id")
        .alias("_reference_current"),
        pl.col("_reference_time_candidate")
        .drop_nulls()
        .first()
        .over("pulse_id")
        .alias("_reference_time"),
    )

    # pulse開始から定電流域まで、ほぼ単調に変化することを確認する
    current_direction = (cls.current.expr.first().over("pulse_id") - pl.col("I0")).sign()

    current_diff = cls.current.expr.diff().over("pulse_id").fill_null(0.0)

    is_monotonic = (
        pl.when(pl.col("_Δt") <= pl.col("_reference_time"))
        .then(current_direction * current_diff >= -current_eps)
        .otherwise(True)
    )

    df = df.with_columns(is_monotonic.all().over("pulse_id").alias("_valid_transient"))

    # transient後は、基準電流から最初に外れるまでをパルス区間とする
    is_in_pulse = (pl.col("_Δt") < current_transient_time) | (
        (cls.current.expr - pl.col("_reference_current")).abs() < current_eps
    )

    df = df.with_columns(
        is_in_pulse.cast(pl.UInt8)
        .cum_prod()
        .over("pulse_id")
        .cast(pl.Boolean)
        .alias("_is_in_pulse")
    )

    return df.filter(
        pl.col("_reference_current").is_not_null(),
        pl.col("_valid_transient"),
        pl.col("_is_in_pulse"),
    ).drop(
        "_row_index",
        "_I_prev",
        "_V_prev",
        "_Q_prev",
        "_is_candidate",
        "_candidate_t0",
        "_is_pulse_start",
        "_t0_start",
        "_Δt",
        "_reference_current_candidate",
        "_reference_time_candidate",
        "_reference_current",
        "_reference_time",
        "_valid_transient",
        "_is_in_pulse",
    )


def _interpolate_dcr(g: pl.DataFrame) -> pl.DataFrame:
    """1つのパルスから指定時刻の値を線形補間する。"""
    t = float(g["t_star"][0])
    x = g["Δt"].to_numpy()

    nearest_idx = int(np.abs(x - t).argmin())

    return g.slice(nearest_idx, 1).with_columns(
        pl.lit(np.interp(t, x, x)).alias("Δt"),
        pl.lit(np.interp(t, x, g["ΔI"])).alias("ΔI"),
        pl.lit(np.interp(t, x, g["ΔV"])).alias("ΔV"),
        pl.lit(x[nearest_idx]).alias("Δt_nearest"),
    )


def calc_z_theta(data: EISData):
    data.abs_Z = (data.re_Z**2 + data.im_Z**2).sqrt()
    data.theta = data.table.select(
        pl.arctan2(EISData.im_Z.expr, EISData.re_Z.expr).alias(EISData.theta.name)
    ).to_series()


_default_q_grid = np.linspace(0.0, 1.0, 101)


def featurize_charge_discharge(
    data: ChargeDischargeData,
    *,
    cycle: int = 1,
    q_grid: Sequence[float] | npt.NDArray[np.floating] = _default_q_grid,
    first_point_elapsed: float = 0.01,
    smooth_window: int = 7,
    smooth_polyorder: int = 2,
) -> dict[str, float]:
    """1サイクルの充放電データを機械学習用の特徴量に変換する。

    Rest → Charge → Rest → Discharge → Rest の測定を対象とし、
    容量・エネルギー、V(q)、dV/dq、およびRest時の電圧変化を返す。

    qは各充放電stepの最終容量で規格化した容量 Q / Q_end とする。
    """
    cls = ChargeDischargeData

    q_grid = np.asarray(q_grid, dtype=float, copy=True)

    if (
        q_grid.ndim != 1
        or len(q_grid) < 2
        or np.any(np.diff(q_grid) <= 0)
        or q_grid[0] < 0.0
        or q_grid[-1] > 1.0
    ):
        raise ValueError("q_grid must be a strictly increasing 1-D sequence within [0, 1].")

    dq = np.diff(q_grid)
    if not np.allclose(dq, dq[0]):
        raise ValueError("q_grid must be equally spaced.")

    if first_point_elapsed < 0:
        raise ValueError("first_point_elapsed must be >= 0.")

    if smooth_window < 3 or smooth_window % 2 == 0:
        raise ValueError("smooth_window must be an odd integer >= 3.")

    if smooth_polyorder >= smooth_window:
        raise ValueError("smooth_polyorder must be smaller than smooth_window.")

    if smooth_window > len(q_grid):
        raise ValueError("smooth_window must not exceed the length of q_grid.")

    if not data.is_col_ready(cls.state):
        raise ValueError("state must already be assigned before feature extraction.")

    # 元データは変更しない。
    data = data.with_table(data.table.clone())

    # stateの連続区間を論理stepとして再構成する。
    # first_point_elapsed > 0 の場合、最初の測定点はstep開始時刻より後にある。
    detect_steps(
        data,
        first_point_elapsed=first_point_elapsed,
    )

    # stepを作り直したため、step単位の容量・エネルギーも再計算する。
    integrate_capacity(
        data,
        skip_cycle=True,
        skip_total=True,
    )
    integrate_energy(
        data,
        skip_cycle=True,
        skip_total=True,
    )

    df = data.table.filter(cls.cycle.expr == cycle)

    if df.is_empty():
        raise ValueError(f"Cycle {cycle} does not exist.")

    steps = df.group_by(cls.step.name, maintain_order=True).agg(
        cls.state.expr.first().alias(cls.state.name)
    )

    states = steps[cls.state.name].to_list()
    expected = [
        State.REST,
        State.CHARGE,
        State.REST,
        State.DISCHARGE,
        State.REST,
    ]

    if states != expected:
        raise ValueError(
            f"Expected Rest -> Charge -> Rest -> Discharge -> Rest, but got {' -> '.join(states)}."
        )

    step_ids = steps[cls.step.name].to_list()

    charge = df.filter(cls.step.expr == step_ids[1])
    rest_after_charge = df.filter(cls.step.expr == step_ids[2])
    discharge = df.filter(cls.step.expr == step_ids[3])
    rest_after_discharge = df.filter(cls.step.expr == step_ids[4])

    features: dict[str, float] = {}

    # 容量・エネルギー
    features["charge_capacity"] = float(charge[cls.step_capacity.name][-1])
    features["discharge_capacity"] = float(discharge[cls.step_capacity.name][-1])
    features["charge_energy"] = float(charge[cls.step_energy.name][-1])
    features["discharge_energy"] = float(discharge[cls.step_energy.name][-1])

    # V(q), dV/dq(q)
    for prefix, step in (
        ("charge", charge),
        ("discharge", discharge),
    ):
        q = step[cls.step_capacity.name].to_numpy()
        v = step[cls.voltage.name].to_numpy()

        valid = np.isfinite(q) & np.isfinite(v)
        q = q[valid]
        v = v[valid]

        if len(q) < 2:
            raise ValueError(f"Not enough valid points in {prefix} step.")

        q_end = q[-1]
        if not np.isfinite(q_end) or q_end <= 0:
            raise ValueError(f"Invalid final capacity in {prefix} step.")

        q = q / q_end

        # 同じ容量値が複数ある場合は最後の電圧を採用する。
        _, reverse_index = np.unique(
            q[::-1],
            return_index=True,
        )
        keep = len(q) - 1 - reverse_index
        keep.sort()

        q = q[keep]
        v = v[keep]

        if len(q) < 2:
            raise ValueError(f"Not enough unique capacity points in {prefix} step.")

        # q=0は通常未観測なので、最初の実測電圧で補う。
        v_raw = np.interp(
            q_grid,
            q,
            v,
            left=v[0],
            right=v[-1],
        )

        v_smooth = savgol_filter_np(
            v_raw,
            window_length=smooth_window,
            polyorder=smooth_polyorder,
        )

        dvdq = savgol_filter_np(
            v_raw,
            window_length=smooth_window,
            polyorder=smooth_polyorder,
            deriv=1,
            delta=float(dq[0]),
        )

        for q_value, value in zip(q_grid, v_smooth):
            features[f"{prefix}_v_q{q_value:.3f}"] = float(value)

        for q_value, value in zip(q_grid, dvdq):
            features[f"{prefix}_dvdq_q{q_value:.3f}"] = float(value)

    # Rest時の電圧変化
    for prefix, previous, rest in (
        (
            "post_charge_rest",
            charge,
            rest_after_charge,
        ),
        (
            "post_discharge_rest",
            discharge,
            rest_after_discharge,
        ),
    ):
        previous_v = previous[cls.voltage.name].drop_nulls().to_numpy()
        rest_v = rest[cls.voltage.name].drop_nulls().to_numpy()

        if len(previous_v) == 0 or len(rest_v) == 0:
            raise ValueError(f"Not enough voltage data for {prefix}.")

        v_ref = float(previous_v[-1])

        features[f"{prefix}_delta_v_first"] = float(rest_v[0]) - v_ref
        features[f"{prefix}_delta_v_end"] = float(rest_v[-1]) - v_ref

    return features


if TYPE_CHECKING:
    _detect_states: Modifier[ChargeDischargeData] = detect_states
    _detect_steps: Modifier[ChargeDischargeData] = detect_steps
    _integrate_capacity: Modifier[ChargeDischargeData] = integrate_capacity
    _integrate_energy: Modifier[ChargeDischargeData] = integrate_energy
    _differentiate: Modifier[ChargeDischargeData] = differentiate
    _chargedischarge_to_cycle: Converter[ChargeDischargeData, CycleSummaryData] = (
        chargedischarge_to_cycle
    )
    _calc_z_theta: Modifier[EISData] = calc_z_theta
