import numpy as np
from scipy_dae.integrate._dae.benchmarks.common import benchmark


"""Index 1 DAE found in Chapter 4 of Brenan1996.

References:
-----------
Brenan1996: https://doi.org/10.1137/1.9781611971224.ch4
"""
def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - t * y2p + y1 - (1 + t) * y2
    F[1] = y2 - np.sin(t)

    return F


if __name__ == "__main__":
    # exponents
    m_max = 45
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10**(-(1 + ms / 4))
    atols = rtols
    h0s = 1e-2 * rtols

    # time span
    t0 = 0
    t1 = 10

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)

    # reference solution
    y_ref = np.array([
        np.exp(-t1) + t1 * np.sin(t1),
        np.sin(t1),
    ])

    benchmark(t0, t1, y0, yp0, F, rtols, atols, h0s, "Brenan", y_ref)
