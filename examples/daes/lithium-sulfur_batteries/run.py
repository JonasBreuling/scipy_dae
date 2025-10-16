import numpy as np
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
import matplotlib.pyplot as plt

"""
References:
-----------
https://pubs.rsc.org/en/content/articlelanding/2016/cp/c5cp05755h#!divAbstract
"""

# Define constants
T = 298.0
Iapp = 1.7
Iapp = 10
F = 9.64853399e4
R = 8.314462175
NA = 6.0221412927e23
e = 1.60217656535e-19
ne = 4.0
ns8, ns4, ns2, ns = 8.0, 4.0, 2.0, 1.0
MS8 = 32.0
# Eh0_ref, El0_ref = 2.195, 2.35
Eh0_ref, El0_ref = 2.35, 2.195
# ih0, il0 = 0.96, 0.48
ih0, il0 = 10, 5
Ar = 0.96
ms = 2.7
velyt = 0.0114
fh = ns4**2 * MS8 * velyt / ns8
# print(f"fh: {fh}")
# fh = 0.7296
fl = ns**2 * ns2 * MS8**2 * velyt**2 / ns4
# print(f"fl: {fl}")
# fl = 0.0665
# exit()
# vol = 0.0114e-3
vol = 0.0114
# rhoS = 2.0e6
rhoS = 2.0e3
Ksp = 0.0001
kp = 100.0
ks = 0.0002

# initial conditions
S80, S40, S220, S20, Sp0 = 2.673, 0.0128002860955374, 4.33210229104915e-06, 1.63210229104915e-06, 2.7e-06
Ih0, Il0, V0 = 1.7, 0.0, 2.4
ETAh0, ETAl0 = -0.0102460242980059, 0.0
Eh0, El0 = 2.4102460242980059, 2.4

V0 = 0
Ih0 = 0

# # - charge
# V0 = 0
# Sp0 = 0
# S20 = 0

# # - discharge
# V0 = 0
# S80 = 0
# Sp0 = 0

# initial split between ih and il must be known

ETAh0 = np.arcsinh(ih0 / (2 * ih0 * Ar)) * 2 * R * T / (ne * F)
ETAl0 = np.arcsinh(il0 / (2 * il0 * Ar)) * 2 * R * T / (ne * F)
S40 = np.sqrt(fh * S80)

# assume S22 = S2 + Sp

# initial conditions
y0 = [S80, S40, S220, S20, Sp0, Ih0, Il0, V0, ETAh0, ETAl0, Eh0, El0]

# define the DAE system
def dae_system(t, y, yp):
    S8, S4, S22, S2, Sp, ih, il, V, ETAh, ETAl, Eh, El = y
    dS8_dt, dS4_dt, dS22_dt, dS2_dt, dSp_dt, *_ = yp

    # differential equations
    eq1 = dS8_dt - (-ih * (ns8 * MS8) / (ne * F) - ks * S8)
    eq2 = dS4_dt - (ih * (ns8 * MS8) / (ne * F) + ks * S8 - il * (ns4 * MS8) / (ne * F))
    eq3 = dS22_dt - (il * (ns2 * MS8) / (ne * F))
    eq4 = dS2_dt - (2.0 * il * (ns * MS8) / (ne * F) - Sp * (kp / (vol * rhoS)) * (S2 - Ksp))
    eq5 = dSp_dt - (Sp * (kp / (vol * rhoS)) * (S2 - Ksp))
    
    # algebraic equations
    eq6 = ih + il - Iapp
    eq7 = ih - 2.0 * ih0 * Ar * np.sinh(ne * F * ETAh / (2.0 * R * T))
    eq8 = il - 2.0 * il0 * Ar * np.sinh(ne * F * ETAl / (2.0 * R * T))
    eq9 = ETAh - (V - Eh)
    eq10 = ETAl - (V - El)
    eq11 = Eh - (Eh0 + (R * T / (4.0 * F)) * np.log(fh * S8 / S4**2.0))
    eq12 = El - (El0 + (R * T / (4.0 * F)) * np.log(fl * S4 / (S22 * S2**2.0)))
    
    return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12])

# initial guess for derivatives
yp0 = np.zeros_like(y0)

# time
t0 = 0
t1 = 200
# t1 = 3600
t_span = (t0, t1)
t_eval = np.linspace(t0, t1)
t_eval = None

# compute consistent initial conditions
F0 = dae_system(t0, y0, yp0)
print(f"y0: {y0}")
print(f"yp0: {yp0}")
print(f"||F0|| = {np.linalg.norm(F0)}")
y0, yp0, fnorm = consistent_initial_conditions(dae_system, t0, y0, yp0)
print(f"y0: {y0}")
print(f"yp0: {yp0}")
print(f"fnorm: {fnorm}")

# Solve the system
atol = 1e-6
rtol = 1e-6
# method = "BDF"
method = "Radau"
sol = solve_dae(
    dae_system, 
    t_span, 
    y0, 
    yp0, 
    method=method, 
    atol=atol, 
    rtol=rtol, 
    t_eval=t_eval, 
    stages=3, 
    # newton_max_iter=None, 
    # # jac_recompute_rate=1e-3, 
    # jac_recompute_rate=5e-1,
    # newton_iter_embedded=3,
)

# extract solution
t, y = sol.t, sol.y
success = sol.success
status = sol.status
message = sol.message
print(f"success: {success}")
print(f"status: {status}")
print(f"message: {message}")
print(f"nfev: {sol.nfev}")
print(f"njev: {sol.njev}")
print(f"nlu: {sol.nlu}")

# visualize solution
variables = ["S8", "S4", "S22", "S2", "Sp", "ih", "il", "V", "ETAh", "ETAl", "Eh", "El"]
rows = 4
cols = 3
fig, ax = plt.subplots(rows, cols)

for i in range(rows * cols):
    ii = i // cols
    jj = i % cols

    ax[ii, jj].plot(t, y[i], label=variables[i])
    ax[ii, jj].grid()
    ax[ii, jj].legend()

plt.show()