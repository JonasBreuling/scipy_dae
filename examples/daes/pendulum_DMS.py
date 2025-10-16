import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize._numdiff import approx_derivative
from scipy._lib._util import _RichResult


def newton(
    fun,
    x0,
    jac="2-point",
    atol=1e-6,
    rtol=1e-6,
    max_iter=20,
):
    nfev = 0
    njev = 0

    # wrap function
    def fun(x, f=fun):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(f(x))

    # wrap jacobian or use a finite difference approximation
    if callable(jac):

        def jacobian(x):
            nonlocal njev
            njev += 1
            return jac(x)

    elif jac in ["2-point", "3-point", "cs"]:

        def jacobian(x):
            nonlocal njev
            njev += 1
            return approx_derivative(
                lambda y: fun(y),
                x,
                method=jac,
            )

    else:
        raise RuntimeError

    # eliminate round-off errors
    Delta_x = np.zeros_like(x0)
    x = x0 + Delta_x

    # initial function value
    f = fun(x)

    # scaling with relative and absolute tolerances
    scale = atol + np.abs(f) * rtol

    # error of initial guess
    error = np.linalg.norm(f / scale) / scale.size**0.5
    converged = error < 1

    # Newton loop
    i = 0
    if not converged:
        for i in range(max_iter):
            # evaluate Jacobian
            J = np.atleast_2d(jacobian(x))

            # Newton update
            dx = np.linalg.solve(J, f)

            # perform Newton step
            Delta_x -= dx
            x = x0 + Delta_x

            # new function value, error and convergence check
            f = np.atleast_1d(fun(x))
            error = np.linalg.norm(f / scale) / scale.size**0.5
            converged = error < 1
            if converged:
                break

        # if not converged:
        #     raise RuntimeError(
        #         f"newton is not converged after {i + 1} iterations with error {error:.2e}"
        #     )

    return _RichResult(
        x=x,
        success=converged,
        error=error,
        fun=f,
        nit=i + 1,
        nfev=nfev,
        njev=njev,
        rate=1.0,  # if converged else 10.0,
    )


def radau_tableau(s):
    # compute quadrature nodes from right Radau polynomial
    Poly = np.polynomial.Polynomial
    poly = Poly([0, 1]) ** (s - 1) * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # compute coefficent matrix
    V = np.vander(c, increasing=True)
    R = np.diag(1 / np.arange(1, s + 1))
    A = np.diag(c) @ V @ R @ np.linalg.inv(V)

    # extract quadrature weights
    b = A[-1, :]

    # quadrature and stage order
    p = 2 * s - 1
    q = s

    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)


def solve_dae_IRK(f, y0, yp0, t_span, h, tableau, atol=1e-6, rtol=1e-6):
    """
    Solves a system of DAEs using an implicit Runge-Kutta method.

    Parameters:
        f: Function defining the DAE system, f(t, y, yp) = 0.
        y0: Initial condition for y.
        yp0: Initial condition for y'.
        t_span: Tuple (t0, t1) defining the time span.
        h: Step size.
        A, b, c: Butcher tableau coefficients for the IRK method.
        atol: Absolute tolerance for the Newton solver.
        rtol: Relative tolerance for the Newton solver.

    Returns:
        _RichResult containing time points, solutions y, and derivatives yp.
    """
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    A, b, c, p, q, s = tableau.A, tableau.b, tableau.c, tableau.p, tableau.q, tableau.s

    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
    m = len(y0)
    s = len(c)

    # Initial guess for stage derivatives
    Yp = np.tile(yp0, s).reshape(s, -1)
    Y = y0 + h * A.dot(Yp)

    # Initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]
    Ys = [Y]
    Yps = [Yp]

    frac = (t1 - t0) / 100
    pbar = tqdm(total=100, leave=True)
    i = 0

    if True:
        while t0 < t1:
            i1 = int((t0 + h) // frac)
            pbar.update(i1 - i)
            pbar.set_description(f"t: {t0 + h:0.2e}s < {t1:0.2e}s")
            i = i1

            # Precompute stage times
            T = t0 + c * h

            def residual(Yp_flat):
                # Reshape flat input to stage derivatives
                Yp = Yp_flat.reshape(s, -1)

                # Compute stage solutions
                Y = y0 + h * A.dot(Yp)

                # Residuals for all stages
                F = np.zeros((s, m))
                for i in range(s):
                    F[i] = f(T[i], Y[i], Yp[i])
                return F.flatten()

            # Solve the nonlinear system
            sol = newton(residual, Yp.flatten(), atol=atol, rtol=rtol)
            if not sol.success:
                raise RuntimeError(
                    f"Newton solver failed at t={t0 + h} with error={sol.error:.2e}"
                )

            # Extract the solution for stages
            Yp = sol.x.reshape(s, -1)
            Y = y0 + h * A.dot(Yp)

            # Update y and y'
            y1 = y0 + h * b.dot(Yp)
            yp1 = Yp[-1]  # only correct for stiffly accurate methods

            # Append to solution arrays
            t.append(t0 + h)
            y.append(y1)
            yp.append(yp1)
            Ys.append(Y)
            Yps.append(Yp)

            # Advance time and update initial values
            t0 += h
            y0 = y1

    return _RichResult(
        t=np.array(t),
        y=np.array(y),
        yp=np.array(yp),
        Y=np.array(Ys),
        Yp=np.array(Yps),
    )


m = 1
l = 1
g = 10

def F1(t, vy, vyp):
    """Acceleration formulation (index 1)."""
    x, y, u, v, la = vy
    x_dot, y_dot, u_dot, v_dot, _ = vyp

    R = np.zeros(5, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u
    R[1] = y_dot - v
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g
    R[4] = 2 * u**2 + 2 * v**2 + 2 * x * u_dot + 2 * y * v_dot

    return R

def F2(t, vy, vyp):
    """Velocity formulation (index 2)."""
    x, y, u, v, la = vy
    x_dot, y_dot, u_dot, v_dot, _ = vyp

    R = np.zeros(5, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u
    R[1] = y_dot - v
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g
    R[4] = 2 * x * u + 2 * y * v

    return R

def F3(t, vy, vyp):
    # """Position formulation (index 3)."""
    # x, y, u, v, la = vy
    # x_dot, y_dot, u_dot, v_dot, _ = vyp

    # R = np.zeros(5, dtype=np.common_type(vy, vyp))
    # R[0] = x_dot - u
    # R[1] = y_dot - v
    # R[2] = m * u_dot - 2 * x * la
    # R[3] = m * v_dot - 2 * y * la + m * g
    # R[4] = x**2 + y**2 - l**2

    """Position and Velocitiy formulation (stabilized index 2)."""
    x, y, u, v, la, mu = vy
    x_dot, y_dot, u_dot, v_dot, _, _ = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u - 2 * x * mu
    R[1] = y_dot - v - 2 * y * mu
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g
    R[4] = 2 * x * u + 2 * y * v
    R[5] = x * x + y * y - l * l

    return R


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([l, 0, 0, 0, 0, 0], dtype=float)
    yp0 = np.array([0, 0, 0, -g, 0, 0], dtype=float)

    # solver options
    s = 2
    h = 5e-2

    def solve_DAE(F, y0, yp0):
        return solve_dae_IRK(F, y0, yp0, t_span=t_span, h=h, tableau=radau_tableau(s))

    ##############
    # dae solution
    ##############
    sol1 = solve_DAE(F1, y0[:-1], yp0[:-1])
    t1 = sol1.t
    y1 = sol1.y.T
    yp1 = sol1.yp.T

    sol2 = solve_DAE(F2, y0[:-1], yp0[:-1])
    t2 = sol2.t
    y2 = sol2.y.T
    yp2 = sol2.yp.T

    sol3 = solve_DAE(F3, y0, yp0)
    # sol3 = solve_DAE(F3, y0[:-1], yp0[:-1])
    t3 = sol3.t
    y3 = sol3.y.T
    yp3 = sol3.yp.T

    # export solution
    import sys
    from pathlib import Path

    header = "t_pos, g_pos, t_vel, g_vel, t_acc, g_acc"

    data = np.vstack((
        t3[None, :],
        (y3[0]**2 + y3[1]**2 - l**2)[None, :],
        t2[None, :],
        (y2[0]**2 + y2[1]**2 - l**2)[None, :],
        t1[None, :],
        (y1[0]**2 + y1[1]**2 - l**2)[None, :],
    )).T

    path = Path(sys.modules["__main__"].__file__)

    np.savetxt(
        path.parent / (path.stem + ".txt"),
        data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t1, y1[0], "-", label="x acc.")
    ax[0].plot(t2, y2[0], "--", label="x vel.")
    ax[0].plot(t3, y3[0], ":", label="x pos.")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t1, y1[1], "-", label="y acc.")
    ax[1].plot(t2, y2[1], "--", label="y vel.")
    ax[1].plot(t3, y3[1], ":", label="y pos.")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t1, y1[0]**2 + y1[1]**2 - l**2, "-", label="g acc.")
    ax[2].plot(t2, y2[0]**2 + y2[1]**2 - l**2, "--", label="g vel.")
    ax[2].plot(t3, y3[0]**2 + y3[1]**2 - l**2, ":", label="g pos.")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t1, 2 * y1[0] * y1[2] + 2 * y1[1] * y1[3], "-", label="g_dot acc.")
    ax[3].plot(t2, 2 * y2[0] * y2[2] + 2 * y2[1] * y2[3], "--", label="g_dot vel.")
    ax[3].plot(t3, 2 * y3[0] * y3[2] + 2 * y3[1] * y3[3], ":", label="g_dot pos.")
    ax[3].legend()
    ax[3].grid()

    plt.show()
