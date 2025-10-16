import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""This example shows how to solve a stiff differential algebraic equation (DAE) 
that describes an electrical circuit, see [1]_ and [2]_. The parameters and the 
exponential model for the transistor are discussed in [3]_. 

References
----------
.. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems", p. 377.
.. [2] https://www.mathworks.com/help/matlab/math/solve-stiff-transistor-dae.html
.. [3] https://doi.org/10.1007/BF01398915
"""

# Problem parameters
Ub = 6
R0 = 1000
Ri = 9000
R1, R2, R3, R4, R5 = 9000 * np.ones(5)
alpha_F = 0.99 # 0.98 - 0.99
alpha_R = 0.5 # 0.1 - 0.5
beta = 1e-6
U_T = 0.026
Ue = lambda t: 0.4 * np.sin(200 * np.pi * t)
f = lambda U: beta * (np.exp(U / U_T) - 1)
C1, C2, C3 = 1e-6 * np.arange(1, 4)

# initial states
y0 = np.zeros(5)
y0[1] = Ub * R1 / (R1 + R2)
y0[2] = Ub * R1 / (R1 + R2)
y0[3] = Ub


def F(t, y, yp):
    U1, U2, U3, U4, U5 = y
    Up1, Up2, Up3, Up4, Up5 = yp

    # f23 = f(U2 - U3)

    # return np.array([
    #     (Ue(t) - U1) / R0 + C1 * (Up2 - Up1),
    #     (Ub - U2) / R2 - U2 / R1 + C1 * (Up1 - Up2) - (1 - alpha_F) * f23,
    #     f23 - U3 / R3 - C2 * Up3,
    #     (Ub - U4) / R4 + C3 * (Up5 - Up4) - alpha_F * f23,
    #     -U5 / R5 + C3 * (Up4 - Up5),
    # ])

    U_B = U2
    U_E = U3
    U_C = U4

    U_BE = U_B - U_E
    U_BC = U_B - U_C

    I_F = f(U_BE)
    # I_R = f(U_BC)
    I_R = 0.0

    I_E = I_F - alpha_R * I_R
    I_C = alpha_F * I_F - I_R
    I_B = I_E - I_C
    # I_B = (1 - alpha_F) * I_F + (1 - alpha_R) * I_R

    I_E *= -1
    I_C *= -1
    I_B *= -1

    return np.array([
        (Ue(t) - U1) / R0 + C1 * (Up2 - Up1),
        (Ub - U2) / R2 - U2 / R1 + C1 * (Up1 - Up2) + I_B,
        -I_E - U3 / R3 - C2 * Up3,
        (Ub - U4) / R4 + C3 * (Up5 - Up4) + I_C,
        -U5 / R5 + C3 * (Up4 - Up5),
    ])


# time span
t0 = 0
t1 = 0.2
# t1 = 0.3
t_span = (t0, t1)
t_eval = np.linspace(t0, t1, num=int(1e4))
# t_eval = None

method = "Radau"
# method = "BDF" # alternative solver

# consistent initial conditions
up0 = np.zeros_like(y0)
print(f"u0: {y0}")
print(f"up0: {up0}")
print(f"fnorm: {np.linalg.norm(F(t0, y0, up0))}")
y0, up0, fnorm = consistent_initial_conditions(F, t0, y0, up0)
print(f"u0: {y0}")
print(f"up0: {up0}")
print(f"fnorm: {fnorm}")

# solver options
# atol = rtol = 1e-2
# atol = rtol = 1e-4
atol = 1e-4
rtol = 1e-8

# dae solution
start = time.time()
sol = solve_dae(F, t_span, y0, up0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, stages=3)
end = time.time()
print(f"elapsed time: {end - start}")
t = sol.t
y = sol.y
success = sol.success
status = sol.status
message = sol.message
print(f"success: {success}")
print(f"status: {status}")
print(f"message: {message}")
print(f"nfev: {sol.nfev}")
print(f"njev: {sol.njev}")
print(f"nlu: {sol.nlu}")

import numpy as np
import sys
from pathlib import Path

header = "t, Ue, U1, U2, U3, U4, U5"

data = np.vstack((
    t[None, :],
    Ue(t)[None, :],
    y,
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
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_title("One transistor amplifier problem")
ax.plot(t, Ue(t), "-", label="input voltage Ue(t)")
ax.plot(t, y[4], "-", label="output voltage U5(t)")
ax.grid()
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.plot(t, y[1] - y[2], label="U2 - U3")
ax.plot(t, y[1] - y[3], label="U2 - U4")
# ax.plot(t, Ue(t), label="input voltage Ue(t)")
# ax.plot(t, y[0], label="output voltage U1(t)")
# ax.plot(t, y[1], label="output voltage U2(t)")
# ax.plot(t, y[2], label="output voltage U3(t)")
# ax.plot(t, y[3], label="output voltage U4(t)")
# ax.plot(t, y[4], label="output voltage U5(t)")
ax.grid()
ax.legend()

plt.show()