from itertools import product
from numpy.testing import suppress_warnings
import pytest
from scipy_dae.integrate import solve_dae


parameters_stiff = ["BDF", "Radau"]
@pytest.mark.slow
@pytest.mark.parametrize("method", parameters_stiff)
def test_integration_robertson_ode(method):
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
            [3, 5, 7], # stages
            [0.0, 0.5, 1.0], # continuous_error_weight
        ):
            res = solve_dae(F_robertson, tspan, y0, yp0, rtol=rtol,
                            atol=atol, method=method, stages=stages,
                            continuous_error_weight=continuous_error_weight)

            # If the stiff mode is not activated correctly, these numbers will be much bigger
            assert res.nfev < 3100
            assert res.njev < 150


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
