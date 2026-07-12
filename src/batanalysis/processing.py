from __future__ import annotations
import math
from typing import cast, Literal, TYPE_CHECKING
from logging import getLogger
import polars as pl
import numpy as np

from exptoolkit.processing import Modifier, Converter
from batanalysis.data import ChargeDischargeData, State, CycleSummaryData, EISData
from batanalysis._savgol import savgol_filter_np

logger = getLogger()

def _cumulative_trapezoid(y: pl.Expr, x: pl.Expr, x0: None | float=None) -> pl.Expr:
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
            pl.lit(State.REST, dtype=dtype)
        ).cast(dtype).alias(cls.state.name)
    ).to_series()

def detect_steps(data: ChargeDischargeData, recalc_time=True) -> None:
    cls = ChargeDischargeData
    if not data.is_col_ready('state'):
        detect_states(data)
    st = cls.state.expr.shift(fill_value=cls.state.expr.first()) != cls.state.expr
    cy = cls.cycle.expr.shift(fill_value=cls.cycle.expr.first()) != cls.cycle.expr
    data.step = data.table.select(
        (st | cy).cum_sum().alias(cls.step.name)
    ).to_series()
    if recalc_time:
        data.table = data.table.with_columns(
            (cls.time.expr - cls.time.expr.first().over(cls.step.expr)).alias(cls.step_time.name),
            (cls.time.expr - cls.time.expr.first().over(cls.cycle.expr)).alias(cls.cycle_time.name),
        )

def integrate_capacity(data: ChargeDischargeData) -> None:
    cls = ChargeDischargeData
    if not data.is_col_ready('step'):
        detect_steps(data)
    data.table = data.table.with_columns(
        # step capacity
        _cumulative_trapezoid(cls.current.expr, cls.step_time.expr/3600, x0=0.0)
            .over(cls.step.expr, cls.cycle.expr)
            .cast(cls.step_capacity.dtype)
            .alias(cls.step_capacity.name)
        * pl.when(cls.state.expr == 'discharge').then(pl.lit(-1.)).otherwise(pl.lit(1.)),
        # cycle capacity
        _cumulative_trapezoid(cls.current.expr, cls.cycle_time.expr/3600, x0=0.0)
            .over(cls.cycle.expr)
            .cast(cls.cycle_capacity.dtype)
            .alias(cls.cycle_capacity.name),
        # total capacity
        _cumulative_trapezoid(cls.current.expr, cls.time.expr/3600, x0=0.0)
            .cast(cls.capacity.dtype)
            .alias(cls.capacity.name)
    )

def integrate_energy(data: ChargeDischargeData) -> None:
    cls = ChargeDischargeData
    if not data.is_col_ready('step_capacity'):
        integrate_capacity(data)

    data.table = data.table.with_columns(
        # step capacity
        _cumulative_trapezoid(cls.voltage.expr, cls.step_capacity.expr, x0=0.0)
            .over(cls.step.expr, cls.cycle.expr)
            .cast(cls.step_energy.dtype)
            .alias(cls.step_energy.name),
        # cycle capacity
        _cumulative_trapezoid(cls.voltage.expr, cls.cycle_capacity.expr, x0=0.0)
            .over(cls.cycle.expr)
            .cast(cls.cycle_energy.dtype)
            .alias(cls.cycle_energy.name),
        # total capacity
        _cumulative_trapezoid(cls.voltage.expr, cls.capacity.expr, x0=0.0)
            .cast(cls.energy.dtype)
            .alias(cls.energy.name)
    )

def differentiate(
    data: ChargeDischargeData,
    window_in_volt: float = 0.02,
    polyorder: int = 2
    ) -> None:
    """calculates dq/dv and dv/dq using Savitzky-Golay algorithm.

    Args:
        window_in_volt: window width for smoothing. recommended value is between 0.01-0.02.
        polyorder: order of the polynomial"""
    cls = ChargeDischargeData
    if not data.is_col_ready(cls.step_capacity.name):
        integrate_capacity(data)
    data.table = (
        data.table
        .group_by(cls.step.name, cls.cycle.name, maintain_order=True)
        .map_groups(lambda g: _differentiate_step(g, window_in_volt, polyorder))
    )

def _differentiate_step(g: pl.DataFrame, window_in_volt, polyorder) -> pl.DataFrame:
    """Called from `polars.DataFrame.group_by(...).map_groups()`."""
    cls = ChargeDischargeData

    def _none():
        return g.with_columns(
            pl.lit(None, dtype=cls.dqdv.dtype).alias(cls.dqdv.name),
            pl.lit(None, dtype=cls.dvdq.dtype).alias(cls.dvdq.name),
        )

    cycle = g[cls.cycle.name].first()
    step = g[cls.step.name].first()

    try:
        if not g[cls.state.name].is_in(['charge', 'discharge']).all():
            return _none()

        q = g[cls.step_capacity.name].to_numpy()
        v = g[cls.voltage.name].to_numpy()
        dqdv_full = np.full_like(v, np.nan)
        dvdq_full = np.full_like(v, np.nan)

        # exclude constant voltage region
        mask = np.abs(v - v[-1]) >= 0.005
        v_ = v[mask]
        q_ = q[mask]

        # should be placed before np.nanmax(v_) because if v_ is empty, np.nanmax(v_) will raise ValueError.
        if len(v_) < 2:
            logger.warning(
                'Skipping differentiation for (cycle, step) = (%s, %s) because there are not enough data points (%s)',
                cycle, step, len(v_)
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
            logger.info('window_length of (cycle, step) = (%s, %s): %s',
                        cycle, step, wl)
            dq = savgol_filter_np(q_, window_length=wl, polyorder=polyorder, deriv=1)
            dv = savgol_filter_np(v_, window_length=wl, polyorder=polyorder, deriv=1)
        else:
            logger.warning(
                'Disable smoothing for (cycle, step) = (%s, %s) because window_length is too large/small (%s)',
                cycle, step, wl
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
            'Failed to differentiate for (cycle, step) = (%s, %s)',
            cycle, step
        )
        return _none()

def chargedischarge_to_cycle(
    data: ChargeDischargeData,
    base: Literal['first', 'max'] = 'first',
    copy_metadata: bool = True
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
        if base == 'first':
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
            cdd.step_capacity.expr.sum().alias('capacity'),
            cdd.step_energy.expr.sum().alias('energy'),
        )
        # ------------------------------------------------------------
        # cycle内の charge / discharge の値を横持ち列として取得
        # ------------------------------------------------------------
        .pivot(
            on = cdd.state.name,
            on_columns= [State.CHARGE, State.DISCHARGE],
            index = cdd.cycle.name,
            values = ['capacity', 'energy']
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
            (100.0 * _ret(csd.capacity_discharge.expr)).alias(csd.capacity_discharge_retention.name),
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
            (100.0 * (csd.capacity_discharge.expr / csd.capacity_charge.expr))
                .alias(csd.coulomb_efficiency.name),
            (100.0 * (csd.energy_discharge.expr / csd.energy_charge.expr))
                .alias(csd.energy_efficiency.name),
        )
    )
    return CycleSummaryData(
        new_table,
        normalization=data.norm,
        metadata=data.metadata.copy() if copy_metadata else None
    )

def calc_dcr(
    data: ChargeDischargeData,
    t_extract: list[float] | Literal['last'] | None = None,
    threshold: float = 0.1,
    current_eps: float = 1e-4,
    ) -> pl.DataFrame:
    """
    電流ステップを検出し、DCRを計算する。

    Parameters
    ----------
    data
        充放電データ。
    t_extract
        DCRを抽出する時刻。
        - list[float]:  指定した経過時間におけるDCRを返す
        - 'last':       検出された各電流ステップの最終点におけるDCRを返す
        - None:         電流ステップ内の全データ点についてDCRを返す
    threshold
        電流ステップ検出に用いる電流変化量の閾値 (mA/[amount])。
    current_eps
        電流値を0とみなす閾値 (mA/[amount])。

    Notes
    -----
    - 電流ステップ開始点は、以下の両方の条件を満たす最初のデータ点とする。

        1. 直前のデータ点からの電流変化量が ``threshold`` より大きい
        2. 直後のデータ点との電流変化量が ``current_eps`` 未満である（定電流区間の開始点）

    - 電流ステップの終了点は、電流値の変化が ``current_eps`` 以上となる最初のデータ点とする（定電流区間の終了点）。

    - 検出された電流ステップは次のように分類される。
        - ``relax``:    ステップ後の電流絶対値が ``current_eps`` 未満
        - ``pulse(+)``: ステップ後の電流値がステップ開始直前の電流値(I0)より大きい（充電パルス）
        - ``pulse(-)``: ステップ後の電流値がステップ開始直前の電流値(I0)より小さい（放電パルス）

    - t_extract に list[float] を指定した場合、ΔV は時間に対して線形補間して求められる。DCR は補間後の ΔV を用いて計算される。
      ステップの幅より長い時間を指定した場合、その差が10%以下であれば端点を用い、それ以上ならデータが破棄される。
    Returns
    -------
    pl.DataFrame

    常に含まれる列:
        pulse_id    電流パルスの通し番号
        pulse_type  電流パルスの分類
        cycle       ステップ開始時のサイクル番号
        step        ステップ開始時のステップ番号
        t0          推定された電流ステップ開始時刻 (s)
                    電流ステップがstep切替と同時に発生した場合はstep開始時刻を使用し、
                    それ以外の場合はステップ検出点の時刻を使用する
        V0          ステップ開始直前の電圧 (V)
        I0          ステップ開始直前の電流 (mA/[amount])
        Q0          ステップ開始直前の積算容量 (mAh/[amount])
        Δt          ステップ開始からの経過時間 (s)
        ΔV          V0からの電圧変化 (V)
        ΔI          I0からの電流変化 (mA/[amount])
        DCR         DCR (Ω・[amount])
        DCR_raw     DCR (Ω)
                    normalizeされていない抵抗値。dataがnormalizeされている場合、DCRとDCR_rawの値は異なる。

    t_extract が list[float] の場合のみ追加される列:
        Δt_nearest  補間に使用した前後データ点のうち、最も近いデータ点におけるΔt (s)。
                    補間結果の信頼性の目安として利用可能。
    """
    cls = ChargeDischargeData
    df = data.table

    cols = ['pulse_id', 'pulse_type', 'cycle', 'step', 't0', 'V0', 'I0', 'Q0', 'Δt', 'ΔI', 'ΔV', 'DCR', 'DCR_raw']
    if (t_extract is not None) and (t_extract != 'last'):
        cols.append('Δt_nearest')

    df = df.with_columns([
        # 計算用の列を準備（直前、直後のI, V)
        cls.current.expr.shift(1).alias("I_prev"),
        cls.current.expr.shift(-1).alias("I_next"),
        cls.voltage.expr.shift(1).alias("V_prev"),
        cls.capacity.expr.shift(1).alias("Q_prev"),
    ]).with_columns([
        # 条件1, 2をチェックし、ステップ開始点をマーキング
        (((cls.current.expr - pl.col("I_prev")).abs() > threshold)
            & ((cls.current.expr - pl.col("I_next")).abs() < current_eps)
        ).alias('is_step_start'),
    ]).with_columns([
        # t0 ~ I0 の計算
        # t0 は電流パルス開始の瞬間の時刻。
        pl.col('is_step_start').cum_sum().alias('pulse_id'),
        pl.when(pl.col('is_step_start')).then(cls.cycle.expr).forward_fill().alias('cycle'),
        pl.when(pl.col('is_step_start')).then(cls.step.expr).forward_fill().alias('step'),
        pl.when(pl.col('is_step_start')).then(
            # 電流ステップ開始の瞬間の累計時刻を求める。
            # 電流ステップ開始点が step (プログラム上の区切り) の切り替わり点でもあるなら、累計時刻から step_time を引けば良い。
            # そうでない場合、開始点は推測不能なので累計時刻をそのまま使う。
            cls.time.expr
            - pl.when(
                (cls.step.expr.shift() != cls.step.expr)
                |(cls.cycle.expr.shift() != cls.cycle.expr)
                ).then(cls.step_time.expr).fill_null(0.0)
        ).forward_fill().alias('t0'),
        pl.when(pl.col('is_step_start')).then(pl.col("V_prev")).forward_fill().alias('V0'),
        pl.when(pl.col('is_step_start')).then(pl.col("Q_prev")).forward_fill().alias('Q0'),
        pl.when(pl.col('is_step_start')).then(pl.col("I_prev")).forward_fill().alias('I0'),
    ]).with_columns([
        # Δt ～ ΔVの計算。
        # Rはt_extractの処理をしてから計算する。
        (cls.time.expr - pl.col('t0')).alias('Δt'),
        (cls.current.expr - pl.col('I0')).alias('ΔI'),
        (cls.voltage.expr - pl.col('V0')).alias('ΔV'),
    ]).filter(
        pl.col('pulse_id') > 0,
        # 電流ステップ開始時点から一定電流の領域だけを切り出す。
        ((cls.current.expr - cls.current.expr.first().over('pulse_id')).abs() < current_eps)
            .cum_prod()
            .over('pulse_id'),
    )

    if t_extract == 'last':
        df = df.group_by('pulse_id', maintain_order=True).last()
    elif t_extract is not None:
        if not t_extract:
            raise ValueError("t_extract is empty.")
        # t_extractの秒数のデータを取り出す（線形補間）
        TIME_TOLERANCE = 0.9
        t_df = pl.DataFrame({"t_star": t_extract})
        df = (
            df
            .join(t_df, how='cross')
            .filter(pl.col('Δt').max().over('pulse_id') >= pl.col('t_star') * TIME_TOLERANCE)
            .group_by('pulse_id', 't_star')
            .map_groups(_interpolate_dcr)
            .group_by(['pulse_id', 't_star'])
            .agg(
                pl.all().first()
            )
            .sort(['pulse_id', 't_star'])
        )

    if (data.norm.amount is None) or (not math.isfinite(data.norm.amount)):
        _norm_factor = 1.0
    else:
        _norm_factor = data.norm.amount

    df = df.with_columns([
        (pl.col('ΔV') / pl.col('ΔI') * 1000).alias('DCR'),
    ]).with_columns(
        (pl.col('DCR') / _norm_factor).alias('DCR_raw'),
        pl.coalesce(
            pl.when(cls.current.expr.abs() < current_eps).then(pl.lit('relax')),
            pl.when(cls.current.expr > pl.col('I0')).then(pl.lit('pulse(+)')),
            pl.when(cls.current.expr < pl.col('I0')).then(pl.lit('pulse(-)')),
        ).alias('pulse_type')
        .cast(dtype=pl.Enum(['relax', 'pulse(+)', 'pulse(-)'])),
    ).select(*cols)

    return df

def _interpolate_dcr(g: pl.DataFrame):
    x = g['Δt']
    x_on = g['t_star']
    exprs: list[pl.Series | pl.Expr] = [
        pl.lit(g.sort((pl.col('Δt') - pl.col('t_star')).abs())['Δt'].first()).alias('Δt_nearest')
    ]
    for col in ['Δt', 'ΔI', 'ΔV']:
        exprs.append(pl.Series(col, np.interp(x_on, x, g[col]), dtype=g[col].dtype))
    return g.with_columns(exprs)


def calc_z_theta(data: EISData):
    data.abs_Z = (data.re_Z**2 + data.im_Z**2).sqrt()
    data.theta = data.table.select(
        pl.arctan2(EISData.im_Z.expr, EISData.re_Z.expr).alias(EISData.theta.name)
    ).to_series()


if TYPE_CHECKING:
    _detect_states: Modifier[ChargeDischargeData] = detect_states
    _detect_steps: Modifier[ChargeDischargeData] = detect_steps
    _integrate_capacity: Modifier[ChargeDischargeData] = integrate_capacity
    _integrate_energy: Modifier[ChargeDischargeData] = integrate_energy
    _differentiate: Modifier[ChargeDischargeData] = differentiate
    _chargedischarge_to_cycle: Converter[ChargeDischargeData, CycleSummaryData] \
        = chargedischarge_to_cycle
    _calc_z_theta: Modifier[EISData] = calc_z_theta
