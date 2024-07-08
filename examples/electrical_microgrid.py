import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative

def F(t, y, yp, u, A, system_data):
    """Define implicit system of differential algebraic equations."""
    R, L, C, splits = system_data
    AC, AR, AL, AI, AV = A

    y = np.nan_to_num(y)
    yp = np.nan_to_num(yp)
    qc, phi, e, jv = np.split(y, splits)
    qp, phip, ep, jvp = np.split(yp, splits)
    i, v = u(t)
    g = lambda e : (AR.T @ e) / R 

    dH1 = phi / L
    dH0 = qc / C

    F0 = AC @ qp + AL @ dH1 + AV @ jv + AR @ g(e) + AI @ i
    F1 = phip - AL.T @ e
    F2 = AC.T @ e - dH0                   # algebraic equation
    F3 = AV.T @ e - v                     # algebraic equation
    # F0 = AC @ qp + AL @ dH1 + AV @ jvp + AR @ g(ep) + AI @ i
    # F1 = phip - AL.T @ ep
    # F2 = AC.T @ ep - dH0                   # algebraic equation
    # F3 = AV.T @ ep - v                     # algebraic equation
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
t1 = 1e4
t_span = (t0, t1)
t_eval = np.linspace(0, 1, num=100)

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
# yp0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print(func(0.0, y0, yp0))
# y0, yp0, fnorm = consistent_initial_conditions(func, lambda t, y, yp, f: jac(t, y, yp), t0, y0, yp0)
# exit()
# solve DAE
start = time.time()
sol = solve_dae(func, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, jac=jac, stages=3)
end = time.time()
print(f'elapsed time: {end - start}')
t = sol.t
y = sol.y

# visualization
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.plot(t, y[0], label="q")
ax.plot(t, y[1], label="phi")
ax.plot(t, y[2], label="e")
ax.plot(t, y[3], label="jv")
ax.legend()
ax.grid()
plt.show()