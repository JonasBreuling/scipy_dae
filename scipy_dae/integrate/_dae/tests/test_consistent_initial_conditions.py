# see https://www.mathworks.com/help/matlab/ref/decic.html
import pytest
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy_dae.integrate import consistent_initial_conditions


def fun_implicit(t, y, yp):
    return np.array([
        2 * yp[0] - y[1],
        y[0] + y[1],
    ])

def jac_implicit(t, y, yp):
    Jy = np.array([
        [0, -1],
        [1,  1],
    ])
    Jyp = np.array([
        [2, 0],
        [0, 0],
    ])
    return Jy, Jyp

parameters_linear = product(
    [([], []), ([0], []), ([], [0])], # fixed_y0, fixed_yp0
    [jac_implicit] # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_linear)
def test_implicit(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, 0]

    f0 = fun_implicit(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0))

    y0, yp0, f0 = consistent_initial_conditions(
        fun_implicit, jac, t0, y0, yp0, 
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0)
    assert np.allclose(f0, np.zeros_like(f0))

if __name__ == "__main__":
    # python -m pytest scipy_dae/integrate/_dae/tests/test_consistent_initial_conditions.py 
    for p in parameters_linear:
        test_implicit(*p)