# see https://www.mathworks.com/help/matlab/ref/decic.html
import pytest
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy_dae.integrate import consistent_initial_conditions

# TODO: Add possiblity to use finite differences in consistent initial conditions


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

parameters_implicit = product(
    [([], []), ([0], []), ([], [0])], # fixed_y0, fixed_yp0
    # [None, jac_implicit] # jac
    [jac_implicit] # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_implicit)
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


def fun_algebraic(t, y, yp):
    return np.array([
        2 * y[0] * y[1] - y[0] + 1,
        y[0] + y[1] * y[1] + 2,
    ])

def jac_algebraic(t, y, yp):
    Jy = np.array([
        [2 * y[1] - 1,  2 * y[0]],
        [           1,  2 * y[1]],
    ])
    Jyp = np.zeros((2, 2))
    return Jy, Jyp

parameters_algebraic = product(
    [([], []), ([], [0]), ([], [1]), ([], [0, 1])], # fixed_y0, fixed_yp0
    # [None, jac_implicit] # jac
    [jac_algebraic] # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_algebraic)
def test_algebraic(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = [-2, 0.5]
    yp0 = np.random.rand(2)

    f0 = fun_algebraic(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0))

    y0, yp0, f0 = consistent_initial_conditions(
        fun_algebraic, jac, t0, y0, yp0, 
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0)
    assert np.allclose(f0, np.zeros_like(f0))


def fun_differential(t, y, yp):
    return np.array([
        2 * yp[0] * yp[1] - yp[0] + 1,
        yp[0] + yp[1] * yp[1] + 2,
    ])

def jac_differential(t, y, yp):
    Jy = np.zeros((2, 2))
    Jyp = np.array([
        [2 * yp[1] - 1,  2 * yp[0]],
        [            1,  2 * yp[1]],
    ])
    return Jy, Jyp

parameters_differential = product(
    [([], []), ([0], []), ([1], []), ([0, 1], [])], # fixed_y0, fixed_yp0
    # [None, jac_implicit] # jac
    [jac_differential] # jac
)
@pytest.mark.parametrize("fixed_y0_and_fixed_yp0, jac", parameters_differential)
def test_differential(fixed_y0_and_fixed_yp0, jac):
    fixed_y0, fixed_yp0 = fixed_y0_and_fixed_yp0
    t0 = 0
    y0 = np.random.rand(2)
    yp0 = [-2, 0.5]

    f0 = fun_differential(t0, y0, yp0)
    assert not np.allclose(f0, np.zeros_like(f0))

    y0, yp0, f0 = consistent_initial_conditions(
        fun_differential, jac, t0, y0, yp0, 
        fixed_y0=fixed_y0, fixed_yp0=fixed_yp0)
    assert np.allclose(f0, np.zeros_like(f0))

if __name__ == "__main__":
    # python -m pytest scipy_dae/integrate/_dae/tests/test_consistent_initial_conditions.py 
    for p in parameters_implicit:
        test_implicit(*p)

    for p in parameters_algebraic:
        test_algebraic(*p)

    for p in parameters_differential:
        test_differential(*p)