import numpy as np
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
import matplotlib.pyplot as plt

from make_params import make_params
from make_discharge import make_ic_disch_12
from model import gen_F
from plot_data import plot_constant_current_simulation

# Initialize problem
ppt = True  # Include precipitation model
BV = True   # Include Butler-Volmer model
# ppt = False  # Include precipitation model
# BV = True   # Include Butler-Volmer model
# ppt = False  # Include precipitation model
# BV = False   # Include Butler-Volmer model

# Load parameters
params = make_params(temp=298, R_area=0.96, ppt=ppt, BV=BV)

# Constant current value (negative = charge) and system sulfur mass
I = 0.17  # A
I = 1.7
I = 6.8
ms = 2.7  # g S

# Time settings
t0 = 0  # s
t1 = 56500  # s
t1 = 5.625e4
t_span = (t0, t1)

# # Current axis for constant current
# I_vec = np.full_like(t_eval, I)  # Constant current; alternatively, this could be a varying vector

# Construct initial conditions
Vin = 2.4  # V
S8in = 0.99 * ms
Spin = 0.000001 * ms

# Calculate the initial conditions for all other variables
# y0 = make_ic_disch_12(params, I_vec[0], Vin, S8in, Spin)
y0 = make_ic_disch_12(params, I, Vin, S8in, Spin)
yp0 = np.zeros_like(y0)

F = gen_F(params, I)

# compute consistent initial conditions
y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
print(f"y0: {y0}")
print(f"yp0: {yp0}")
print(f"fnorm: {fnorm}")

# solve the system
atol = 1e-6
rtol = 1e-6
# method = "BDF"
method = "Radau"
sol = solve_dae(
    F, 
    t_span, 
    y0, 
    yp0, 
    method=method, 
    atol=atol, 
    rtol=rtol, 
    # t_eval=t_eval, 
    stages=3, 
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

# # visualize solution
# variables = ["S8", "S4", "S22", "S2", "Sp", "ih", "il", "V", "ETAh", "ETAl", "Eh", "El"]
# rows = 4
# cols = 3
# fig, ax = plt.subplots(rows, cols)

# for i in range(rows * cols):
#     ii = i // cols
#     jj = i % cols

#     ax[ii, jj].plot(t, y[i], label=variables[i])
#     ax[ii, jj].grid()
#     ax[ii, jj].legend()

# plt.show()

plot_constant_current_simulation(t, y)
