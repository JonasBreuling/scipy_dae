from itertools import product
import numpy as np
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
import pytest
from scipy.sparse import coo_matrix, csc_matrix, diags, identity
from scipy_dae.integrate import solve_dae

from scipy.integrate._ivp.tests.test_ivp import fun_linear, jac_linear, sol_linear
from scipy.integrate._ivp.tests.test_ivp import fun_rational, fun_rational_vectorized, jac_rational, jac_rational_sparse, sol_rational
from scipy.integrate._ivp.tests.test_ivp import fun_complex, jac_complex, jac_complex_sparse, sol_complex
from scipy.integrate._ivp.tests.test_ivp import compute_error


def F_linear(t, y, yp):
    return yp - fun_linear(t, y)


J_linear = (
    -jac_linear(),
    np.eye(2),
)


J_linear_sparse = (
    -csc_matrix(jac_linear()),
    identity(2, format="csc"),
)


def F_rational(t, y, yp):
    return yp - fun_rational(t, y)


def F_rational_vectorized(t, y, yp):
    return yp - fun_rational_vectorized(t, y)


def J_rational(t, y, yp):
    Jy = -jac_rational(t, y)
    Jyp = np.eye(2)
    return Jy, Jyp


def J_rational_sparse(t, y, yp):
    Jy = -jac_rational_sparse(t, y)
    Jyp = identity(2)
    return Jy, Jyp


def F_complex(t, y, yp):
    return yp - fun_complex(t, y)


def J_complex(t, y, yp):
    Jy = -jac_complex(t, y)
    Jyp = np.eye(1)
    return Jy, Jyp


def J_complex_sparse(t, y, yp):
    Jy = -jac_complex_sparse(t, y)
    Jyp = identity(1, dtype=np.common_type(y, yp))
    return Jy, Jyp


parameters_linear = product(
    ["BDF", "Radau"], # method
    [None, J_linear, J_linear_sparse] # jac
)
@pytest.mark.parametrize("method, jac", parameters_linear)
def test_integration_const_jac(method, jac):
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    yp0 = fun_linear(0, y0)
    t_span = [0, 2]

    res = solve_dae(F_linear, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


parameters_complex = product(
    ["BDF"], # method
    [None, J_complex, J_complex_sparse] # jac
)
@pytest.mark.parametrize("method, jac", parameters_complex)
def test_integration_complex(method, jac):
    rtol = 1e-3
    atol = 1e-6
    y0 = np.array([0.5 + 1j])
    yp0 = fun_complex(0, y0)
    t_span = [0, 1]
    tc = np.linspace(t_span[0], t_span[1])

    res = solve_dae(F_complex, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    # assert res.nfev < 25
    assert res.nfev < 29

    if method == 'BDF':
        assert_equal(res.njev, 1)
        assert res.nlu < 6
    else:
        assert res.njev == 0
        assert res.nlu == 0

    y_true = sol_complex(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert np.all(e < 5)

    yc_true = sol_complex(tc)
    yc = res.sol(tc)[0]
    e = compute_error(yc, yc_true, rtol, atol)

    assert np.all(e < 5)


parameters_rational = product(
    [False], # vectorized
    ["BDF", "Radau"], # method
    [[5, 9], [5, 1]], # t_span
    [None, J_rational, J_rational_sparse] # jac
)
@pytest.mark.parametrize("vectorized, method, t_span, jac", parameters_rational)
def test_integration_rational(vectorized, method, t_span, jac):
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    yp0 = fun_rational(5, y0)

    if vectorized:
        fun = F_rational_vectorized
    else:
        fun = F_rational

    res = solve_dae(fun, t_span, y0, yp0, rtol=rtol, atol=atol, 
                    method=method, dense_output=True, jac=jac, 
                    vectorized=vectorized)
    
    assert_equal(res.t[0], t_span[0])
    assert_(res.t_events is None)
    assert_(res.y_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    if method == "BDF":
        assert_(0 < res.njev < 3)
        assert_(0 < res.nlu < 10)
    else: # Radau
        assert_(0 < res.njev < 4)
        assert_(0 < res.nlu < 11)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 6))

    tc = np.linspace(*t_span)
    yc_true = sol_rational(tc)
    yc = res.sol(tc)

    e = compute_error(yc, yc_true, rtol, atol)
    assert_(np.all(e < 6))

    tc = (t_span[0] + t_span[-1]) / 2
    yc_true = sol_rational(tc)
    yc = res.sol(tc)

    e = compute_error(yc, yc_true, rtol, atol)
    assert_(np.all(e < 5))

    assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)


parameters_stiff = ["BDF", "Radau"]
@pytest.mark.slow
@pytest.mark.parametrize("method", parameters_stiff)
def test_integration_robertson(method):
    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]
    
    def F_robertson(t, state, statep):
        return statep - fun_robertson(t, state)
    
    rtol = 1e-6
    atol = 1e-6
    y0 = [1e4, 0, 0]
    yp0 = fun_robertson(0, y0)
    tspan = [0, 1e8]

    if method == "BDF":
        for NDF_strategy, max_order in product(
            ["stability", "efficiency", None], # NDF_strategy
            [1, 2, 3, 4, 5, 6], # max_order
        ):
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                        "Choosing `max_order = 6` is not recomended due to its "
                        "poor stability properties.")
                res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                                atol=atol, method=method, max_order=max_order,
                                NDF_strategy=NDF_strategy)
                
                # If the stiff mode is not activated correctly, these numbers will be much 
                # bigger (see max_order=1 case)
                if max_order == 1:
                    assert res.nfev < 21000
                else:
                    assert res.nfev < 5000
                assert res.njev < 50

    else: # Radau
        for stages, continuous_error_weight in product(
            [3, 5], # stages
            [0.0, 0.5, 1.0], # continuous_error_weight
        ):
            res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                            atol=atol, method=method, stages=stages,
                            continuous_error_weight=continuous_error_weight)

            # If the stiff mode is not activated correctly, these numbers will be much bigger
            assert res.nfev < 3000
            assert res.njev < 100


parameters_stiff = ["BDF", "Radau"]
@pytest.mark.slow
@pytest.mark.parametrize("method", parameters_stiff)
def test_integration_robertson_dae(method):
    def F_robertson(t, y, yp):
        y1, y2, y3 = y
        y1p, y2p, y3p = yp

        return [
            y1p - (-0.04 * y1 + 1e4 * y2 * y3),
            y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2),
            y1 + y2 + y3 - 1,
        ]
    
    rtol = 1e-6
    atol = 1e-6
    y0 = [1, 0, 0]
    yp0 = [-0.04, 0.04, 0]
    tspan = [0, 1e8]

    if method == "BDF":
        for NDF_strategy, max_order in product(
            ["stability", "efficiency", None], # NDF_strategy
            [1, 2, 3, 4, 5, 6], # max_order
        ):
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                        "Choosing `max_order = 6` is not recomended due to its "
                        "poor stability properties.")
                res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                                atol=atol, method=method, max_order=max_order,
                                NDF_strategy=NDF_strategy)
                
                # If the stiff mode is not activated correctly, these numbers will be much 
                # bigger (see max_order=1 case)
                if method == "BDF" and max_order == 1:
                    assert res.nfev < 21000
                else:
                    assert res.nfev < 3000
                assert res.njev < 100

    else: # Radau
        for stages, continuous_error_weight in product(
            [3, 5], # stages
            [0.0, 0.5, 1.0], # continuous_error_weight
        ):
            res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                            atol=atol, method=method, stages=stages,
                            continuous_error_weight=continuous_error_weight)

            # If the stiff mode is not activated correctly, these numbers will be much bigger
            assert res.nfev < 3000
            assert res.njev < 100


if __name__ == "__main__":
    for params in parameters_linear:
        test_integration_const_jac(*params)

    for params in parameters_complex:
        test_integration_complex(*params)

    for params in parameters_rational:
        test_integration_rational(*params)

    for params in parameters_stiff:
        test_integration_robertson(params)

    for params in parameters_stiff:
        test_integration_robertson_dae(params)
