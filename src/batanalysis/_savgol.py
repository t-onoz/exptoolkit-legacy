from __future__ import annotations

import typing as t
from math import factorial

import numpy as np


def savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0):
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    half = window_length // 2

    # Vandermonde matrix
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, polyorder + 1, increasing=True)

    A_pinv = np.linalg.pinv(A)

    coeffs = A_pinv[deriv] * factorial(deriv) / (delta**deriv)

    return coeffs


def _edge_coeffs(window_length: int, polyorder: int, deriv: int, delta: float):
    half = window_length // 2
    coeffs = []

    for i in range(half):
        xi = np.arange(-i, window_length - i, dtype=float)
        A = np.vander(xi, polyorder + 1, increasing=True)
        A_pinv = np.linalg.pinv(A)
        c = A_pinv[deriv] * factorial(deriv) / (delta**deriv)
        coeffs.append(c)

    return np.array(coeffs)  # shape: (half, window_length)


def savgol_filter_np(
    x,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    mode: t.Literal["interp"] = "interp",
):
    coeffs = savgol_coeffs(window_length, polyorder, deriv, delta)

    y = np.convolve(x, coeffs[::-1], mode="same")

    if mode == "interp":
        edge = _edge_coeffs(window_length, polyorder, deriv, delta)

        half = window_length // 2

        for i in range(half):
            y[i] = edge[i] @ x[:window_length]
            y[-i - 1] = edge[i] @ x[-window_length:][::-1] * (-1) ** deriv

        return y
    raise ValueError('only mode="interp" is supported.')
