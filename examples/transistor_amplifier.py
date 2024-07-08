import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""Stiff differential-algebraic equation (DAE) from electrical circuit.
   AMP1DAE runs a demo of the solution of a stiff differential-algebraic
   equation (DAE) system expressed as a problem with a singular mass matrix,
   M*u' = f(t,u).

   This is the one transistor amplifier problem displayed on p. 377 of
   E. Hairer and G. Wanner, Solving Ordinary Differential Equations II
   Stiff and Differential-Algebraic Problems, 2nd ed., Springer, Berlin,
   1996. This problem can easily be written in semi-explicit form, but it
   arises in the form M*u' = f(t,u) with a mass matrix that is not
   diagonal.  It is solved here in its original, non-diagonal form.
   Fig. 1.4 of Hairer and Wanner shows the solution on [0 0.2], but here it
   is computed on [0 0.05] because the computation is less expensive and
   the nature of the solution is clearly visible on the shorter interval.

   References:
   -----------
   [1]_ mathworks: https://fr.mathworks.com/help/matlab/math/solve-stiff-transistor-dae.html \\
   [2]_ Hairer1996: ...
"""


# Problem parameters
Ub = 6
R0 = 1000
R = 9000
alpha = 0.99
beta = 1e-6
Uf = 0.026
Ue = lambda t: 0.4 * np.sin(200 * np.pi * t)  

# The constant, singular mass matrix
c = 1e-6 * np.arange(1, 4)
M = np.zeros((5, 5))
M[0, 0] = -c[0]
M[0, 1] =  c[0]
M[1, 0] =  c[0]
M[1, 1] = -c[0]
M[2, 2] = -c[1]
M[3, 3] = -c[2]
M[3, 4] =  c[2]
M[4, 3] =  c[2]
M[4, 4] = -c[2]

# Hairer and Wanner's RADAU5 requires consistent initial conditions
# which they take to be
u0 = np.zeros(5)
u0[1] = Ub / 2
u0[2] = Ub / 2
u0[3] = Ub

# Perturb the algebraic variables to test initialization.
u0[3] = u0[3] + 0.1
u0[4] = u0[4] + 0.1


def F(t, u, up):
    f23 = beta * (np.exp((u[1] - u[2]) / Uf) - 1)
    dudt = np.array([
        -(Ue(t) - u[0]) / R0,
        -(Ub / R - u[1] * 2 / R - (1 - alpha) * f23),
        -(f23 - u[2] / R),
        -((Ub - u[3]) / R - alpha * f23),
        u[4] / R,
    ])
    return M @ up - dudt


def jac(t, y, yp, f):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                          z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp


# time span
t0 = 0
t1 = 0.05
t_span = (t0, t1)

method = "Radau"
# method = "BDF" # alternative solver

# consistent initial conditions
up0 = np.zeros_like(u0)
print(f"u0: {u0}")
print(f"up0: {up0}")
u0, up0, fnorm = consistent_initial_conditions(F, jac, t0, u0, up0)
print(f"u0: {u0}")
print(f"up0: {up0}")
print(f"fnorm: {fnorm}")

# f0 = F(t0, u0, up0)
# Jy0, Jyp0 = jac(t0, u0, up0, f0)
# J0 = Jy0 + Jyp0
# print(f"Jy0:\n{Jy0}")
# print(f"Jyp0:\n{Jyp0}")
# print(f"J0:\n{Jyp0}")
# print(f"rank(Jy0):  {np.linalg.matrix_rank(Jy0)}")
# print(f"rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")
# print(f"rank(J0):   {np.linalg.matrix_rank(J0)}")
# print(f"J0.shape: {J0.shape}")
# exit()

# solver options
atol = rtol = 1e-6

# dae solution
start = time.time()
sol = solve_dae(F, t_span, u0, up0, atol=atol, rtol=rtol, method=method)
end = time.time()
print(f"elapsed time: {end - start}")
t = sol.t
u = sol.y
success = sol.success
status = sol.status
message = sol.message
print(f"success: {success}")
print(f"status: {status}")
print(f"message: {message}")
print(f"nfev: {sol.nfev}")
print(f"njev: {sol.njev}")
print(f"nlu: {sol.nlu}")

# visualization
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_title("One transistor amplifier problem")
ax.plot(t, Ue(t), label="input voltage Ue(t)")
ax.plot(t, u[4], label="output voltage U5(t)")
ax.grid()
ax.legend()
plt.show()
