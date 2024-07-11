import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative

def F(t, y, yp, u, A, system_data):
    """Define implicit system of differential algebraic equations."""
    R, L, C, splits = system_data
    # AC, AR, AL, AI, AV = A # <= this was the error!
    AC, AR, AL, AV, AI = A

    y = np.nan_to_num(y)
    yp = np.nan_to_num(yp)
    qc, phi, e, jv = np.split(y, splits)
    q_t, phi_t, e_t, jv_t = np.split(yp, splits)
    i, v = u(t)

    F0 = AC @ q_t + AL @ (phi / L) + AV @ jv + AR @ (AR.T @ e) / R + AI @ i
    F1 = phi_t - AL.T @ e
    F2 = AC.T @ e - qc / C                # algebraic equation
    F3 = AV.T @ e - v                     # algebraic equation
    return np.concatenate((F0, F1, F2, F3))


def jac(t, y, yp, u, A, system_data, f=None):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp, u, A, system_data)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                          z, method="3-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp


def get_microgrid_params():
    AC = np.array([[1, 0, 0, -1]]).T
    AR = np.array([[0, -1, 1, 0]]).T
    AL = np.array([[0, 0, -1, 1]]).T
    AV = np.array([[-1, 1, 0, 0]]).T
    AI = np.array([[0, 0, 0, 0]]).T
    splits = np.array([len(AC.T), 
                       len(AC.T) + len(AL.T), 
                       len(AC.T) + len(AL.T) + len(AC)])
    R = np.array([1.0])
    L = np.array([1.0])
    C = np.array([1.0])
    A = (AC, AR, AL, AV, AI)
    system_data = (R, L, C, splits)
    return (A, system_data)


def get_microgrid_policy(t):
    return np.array([[0, np.sin(t)]]).T # [i, v]

# time span
t0 = 0
t1 = 2e1
t_span = (t0, t1)
t_eval = np.linspace(t0, t1, num=int(1e3))

# solver options
method = "Radau"
# method = "BDF" # alternative solver
atol = rtol = 1e-4

# system parameters
A, system_data = get_microgrid_params()
AC, AR, AL, AV, AI = A
len_y = len(AC.T) + len(AL.T) + len(AC) + len(AV.T)

# initial conditions
u = get_microgrid_policy
func = partial(F, A=A, system_data=system_data, u=u)
jac = partial(jac, A=A, system_data=system_data, u=u)
y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
yp0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
f0 = func(t0, y0, yp0)
print(f"f0: {f0}")
Jy0, Jyp0 = jac(t0, y0, yp0)
J0 = Jy0 + Jyp0
print(f"Jy0:\n{Jy0}")
print(f"Jyp0:\n{Jyp0}")
print(f"J0:\n{Jyp0}")
print(f"rank(Jy0):  {np.linalg.matrix_rank(Jy0)}")
print(f"rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")
print(f"rank(J0):   {np.linalg.matrix_rank(J0)}")
print(f"J0.shape: {J0.shape}")
# y0, yp0, fnorm = consistent_initial_conditions(func, lambda t, y, yp, f: jac(t, y, yp), t0, y0, yp0)

# solve DAE
start = time.time()
sol = solve_dae(func, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, jac=jac, t_eval=t_eval)
end = time.time()
print(f'elapsed time: {end - start}')
t = sol.t
y = sol.y

# visualization
fig, ax = plt.subplots(2, 2)

ax[0, 0].set_xlabel("t")
ax[0, 0].plot(t, y[0], label="q")
ax[0, 0].legend()
ax[0, 0].grid()

ax[0, 1].set_xlabel("t")
ax[0, 1].plot(t, y[1], label="phi")
ax[0, 1].legend()
ax[0, 1].grid()

ax[1, 0].set_xlabel("t")
ax[1, 0].plot(t, y[2], label="e")
ax[1, 0].legend()
ax[1, 0].grid()

ax[1, 1].set_xlabel("t")
ax[1, 1].plot(t, y[3], label="jv")
ax[1, 1].legend()
ax[1, 1].grid()

plt.show()