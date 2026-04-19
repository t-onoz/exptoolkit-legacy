from __future__ import annotations
import typing as t
from math import factorial
import numpy as np

def savgol_coeffs(window_length: int, polyorder:int , deriv: int=0, delta: float=1.0):
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    half = window_length // 2

    # Vandermonde matrix
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, polyorder + 1, increasing=True)

    A_pinv = np.linalg.pinv(A)

    coeffs = A_pinv[deriv] * factorial(deriv) / (delta ** deriv)

    return coeffs

def _edge_coeffs(window_length: int, polyorder: int, deriv: int, delta: float):
    half = window_length // 2
    coeffs = []

    for i in range(half):
        xi = np.arange(-i, window_length - i, dtype=float)
        A = np.vander(xi, polyorder + 1, increasing=True)
        A_pinv = np.linalg.pinv(A)
        c = A_pinv[deriv] * factorial(deriv) / (delta ** deriv)
        coeffs.append(c)

    return np.array(coeffs)  # shape: (half, window_length)

def savgol_filter_np(
    x,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    mode: t.Literal["interp"] = "interp"
    ):
    coeffs = savgol_coeffs(window_length, polyorder, deriv, delta)

    y = np.convolve(x, coeffs[::-1], mode="same")

    if mode == "interp":
        edge = _edge_coeffs(window_length, polyorder, deriv, delta)

        half = window_length // 2

        for i in range(half):
            y[i] = edge[i] @ x[:window_length]
            y[-i-1] = edge[i] @ x[-window_length:][::-1] * (-1)**deriv

        return y
    raise ValueError('only mode="interp" is supported.')

def _test_shape():
    x = np.linspace(0, 1, 100)
    y = savgol_filter_np(x, 11, 3)
    assert y.shape == x.shape
    print("[OK] shape test")

def _test_constant_signal():
    x = np.ones(100)
    y = savgol_filter_np(x, 11, 3)

    assert np.allclose(y, 1.0, atol=1e-10)
    print("[OK] constant signal")

def _test_polynomial_recovery():
    x = np.arange(100, dtype=float)
    true = 0.1 * x**3 - 0.5 * x**2 + 2 * x + 1

    y = savgol_filter_np(true, 21, 3)

    center = slice(30, 70)
    assert np.allclose(y[center], true[center], rtol=1e-6, atol=1e-6)
    print("[OK] polynomial recovery")

def _test_derivative():
    x = np.linspace(0, 10, 200)
    y_true = x ** 2  # f(x)=x^2 → f'(x)=2x

    y = savgol_filter_np(y_true, 11, 3, deriv=1, delta=x[1]-x[0])

    expected = 2 * x

    center = slice(10, 90)
    assert np.allclose(y[center], expected[center], rtol=1e-3, atol=1e-3)
    print("[OK] derivative test")

def _test_edge_no_nan():
    x = np.random.randn(100)
    y = savgol_filter_np(x, 11, 3)
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))
    print("[OK] no NaN/Inf")

def _test_noisy_smoothing():
    np.random.seed(0)

    x = np.linspace(0, 10, 200)
    true = np.sin(x)

    noise = np.random.normal(0, 0.3, size=x.shape)
    noisy = true + noise

    y = savgol_filter_np(noisy, window_length=11, polyorder=3)

    err_before = np.var(noisy - true)
    err_after = np.var(y - true)

    assert err_after < err_before
    print("[OK] noisy smoothing (variance reduced)")

def _test_signal_shape_preservation():
    np.random.seed(0)

    x = np.linspace(0, 4*np.pi, 200)
    true = np.sin(x)

    noise = np.random.normal(0, 0.2, size=x.shape)
    noisy = true + noise

    y = savgol_filter_np(noisy, 21, 3)

    corr_before = np.corrcoef(noisy, true)[0, 1]
    corr_after = np.corrcoef(y, true)[0, 1]

    assert corr_after > corr_before
    print("[OK] shape preservation (correlation improved)")

def _plot_smoothing_demo():
    import matplotlib.pyplot as plt

    np.random.seed(0)

    x = np.linspace(0, 10, 300)
    true = np.sin(x)
    true_deriv = np.cos(x)

    noise = np.random.normal(0, 0.05, size=x.shape)
    noisy = true + noise
    noisy_deriv = np.gradient(noisy) / (x[1] - x[0])

    filtered = savgol_filter_np(noisy, 21, 3, delta=x[1]-x[0])
    filtered_deriv = savgol_filter_np(noisy, 31, 3, deriv=1, delta=x[1]-x[0])

    plt.figure(figsize=(10, 4))
    plt.plot(x, true, label="true")
    plt.plot(x, noisy, label="noisy", alpha=0.4)
    plt.plot(x, filtered, label="filtered", linewidth=2)
    plt.title("Savitzky-Golay visual test")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 3))
    plt.plot(x, noisy - true, label="noise", alpha=0.6)
    plt.plot(x, filtered - true, label="filtered error")
    plt.title("Error comparison")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_deriv, label="true derivative")
    plt.plot(x, noisy_deriv, label="noise", alpha=0.4)
    plt.plot(x, filtered_deriv, label="filtered", linewidth=2)
    plt.title("Savitzky-Golay visual test (dy/dx)")
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    _plot_smoothing_demo()

    _test_shape()
    _test_constant_signal()
    _test_polynomial_recovery()
    _test_derivative()
    _test_edge_no_nan()

    _test_noisy_smoothing()
    _test_signal_shape_preservation()


    print("ALL TESTS PASSED")
