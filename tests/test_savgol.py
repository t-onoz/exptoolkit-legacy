import numpy as np
import pytest

from batanalysis._savgol import savgol_filter_np


def test_shape():
    x = np.linspace(0, 1, 100)
    y = savgol_filter_np(x, 11, 3)
    assert y.shape == x.shape


def test_constant_signal():
    x = np.ones(100)
    y = savgol_filter_np(x, 11, 3)
    assert np.allclose(y, 1.0, atol=1e-10)


def test_polynomial_recovery():
    x = np.arange(100, dtype=float)
    true = 0.1 * x**3 - 0.5 * x**2 + 2 * x + 1

    y = savgol_filter_np(true, 21, 3)

    center = slice(30, 70)
    assert np.allclose(y[center], true[center], rtol=1e-6, atol=1e-6)


def test_derivative():
    x = np.linspace(0, 10, 200)
    y_true = x**2

    y = savgol_filter_np(y_true, 11, 3, deriv=1, delta=x[1] - x[0])

    expected = 2 * x
    center = slice(10, 90)
    assert np.allclose(y[center], expected[center], rtol=1e-3, atol=1e-3)


def test_edge_no_nan():
    x = np.random.randn(100)
    y = savgol_filter_np(x, 11, 3)
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_noisy_smoothing():
    np.random.seed(0)

    x = np.linspace(0, 10, 200)
    true = np.sin(x)

    noise = np.random.normal(0, 0.3, size=x.shape)
    noisy = true + noise

    y = savgol_filter_np(noisy, window_length=11, polyorder=3)

    err_before = np.var(noisy - true)
    err_after = np.var(y - true)

    assert err_after < err_before


def test_signal_shape_preservation():
    np.random.seed(0)

    x = np.linspace(0, 4 * np.pi, 200)
    true = np.sin(x)

    noise = np.random.normal(0, 0.2, size=x.shape)
    noisy = true + noise

    y = savgol_filter_np(noisy, 21, 3)

    corr_before = np.corrcoef(noisy, true)[0, 1]
    corr_after = np.corrcoef(y, true)[0, 1]

    assert corr_after > corr_before


def test_scipy_equivalence():
    scipy_signal = pytest.importorskip("scipy.signal")
    savgol_filter = scipy_signal.savgol_filter

    np.random.seed(0)

    x = np.linspace(0, 4 * np.pi, 200)
    true = np.sin(x)

    noise = np.random.normal(0, 0.033, size=x.shape)
    noisy = true + noise

    y_np = savgol_filter_np(noisy, 21, 3)
    y_sp = savgol_filter(noisy, 21, 3)
    assert np.allclose(y_np, y_sp)

    deriv_y_np = savgol_filter_np(noisy, 21, 3, deriv=1)
    deriv_y_sp = savgol_filter(noisy, 21, 3, deriv=1)
    assert np.allclose(deriv_y_np, deriv_y_sp)
