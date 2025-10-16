import numpy as np
from tqdm import tqdm
from scipy.optimize._numdiff import approx_derivative
from scipy._lib._util import _RichResult
import matplotlib.pyplot as plt


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


def half_explicit_Euler(system, q0, u0, t_span, h, atol=1e-8, rtol=1e-8):
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    W0 = system.W(t0, q0)
    nla = W0.shape[1]

    # initialize solution arrays
    t = [t0]
    q = [q0]
    u = [u0]
    la0 = np.zeros(nla) # initial guess
    la = [la0]

    frac = (t1 - t0) / 100
    pbar = tqdm(total=100, leave=True)
    i = 0
    
    def f(t, q, u):
        return u
    
    def k(t, q, u, la):
        M = system.M(t, q)
        h = system.h(t, q, u)
        W = system.W(t, q)
        return np.linalg.solve(M, h + W @ la)
    
    while t0 < t1:
        i1 = int((t0 + h) // frac)
        pbar.update(i1 - i)
        pbar.set_description(f"t: {t0 + h:0.2e}s < {t1:0.2e}s")
        i = i1
        
        def residual(la):
            t1 = t0 + h
            return system.g(t1, q0 + h * f(t1, q0, u0 + h * k(t1, q0, u0, la)))

        # Solve the nonlinear system
        sol = newton(residual, la0, atol=atol, rtol=rtol)
        if not sol.success:
            raise RuntimeError(
                f"Newton solver failed at t={t0 + h} with error={sol.error:.2e}"
            )
        
        # compute solution
        la1 = sol.x
        u1 = u0 + h * k(t0 + h, q0, u0, la1)
        q1 = q0 + h * f(t0 + h, q0, u1)

        # Append to solution arrays
        t.append(t0 + h)
        q.append(q1)
        u.append(u1)
        la.append(la1)

        # Advance time and update initial values
        t0 += h
        q0 = q1.copy()
        u0 = u1.copy()

    return _RichResult(
        t=np.array(t),
        q=np.array(q).T,
        u=np.array(u).T,
    )


m = 1
l = 1
g = 10

class CartesianPendulum:
    def __init__(self, m=1, l=1, g=10):
        self.m = m
        self.l = l
        self.grav = g

    def M(self, t, q):
        return np.diag([self.m, self.m])
    
    def h(self, t, q, u):
        return np.array([0, -self.m * self.grav])
    
    def g(self, t, q):
        x, y = q
        return np.array([x**2 + y**2 - self.l**2])
    
    def W(self, t, q):
        x, y = q
        return np.array([[2 * x], [2 * y]])


if __name__ == "__main__":
    # time span
    t0 = 0
    t = 10
    t_span = (t0, t)

    # mechanical system
    system = CartesianPendulum()

    # initial conditions
    q0 = np.array([1, 0])
    u0 = np.array([0, 0])

    ##############
    # dae solution
    ##############
    # h = 5e-2
    h = 1e-2
    sol1 = half_explicit_Euler(system, q0, u0, t_span, h)
    t = sol1.t
    q = sol1.q
    u = sol1.u

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, q[0], "-", label="x")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, q[1], "-", label="y")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, q[0]**2 + q[1]**2 - l**2, "-", label="g")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, 2 * q[0] * u[0] + 2 * q[1] * u[1], "-", label="g_dot")
    ax[3].legend()
    ax[3].grid()

    plt.show()
