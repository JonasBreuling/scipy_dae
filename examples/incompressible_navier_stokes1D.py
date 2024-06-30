import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative


# see https://www.ams.org/journals/bull/1967-73-06/S0002-9904-1967-11853-6/S0002-9904-1967-11853-6.pdf
# see https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650071008 for boundary conditions
# see https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver/blob/master/docs_part1/vanilaCavityFlow_EN.md
# see https://chatgpt.com/c/285a219d-4d5a-4415-aea2-1dd991287617


def initialize_grid(Nx, Ny, Lx, Ly):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    return x, y, dx, dy


def initial_conditions(Nx, Ny):
    # Initial velocity field (u, v) and pressure field (p)
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))
    u[:, 0] = 1.0
    y = np.concatenate((u.flatten(), v.flatten(), p.flatten()))
    return y


def compute_rhs(y, Nx, Ny, rho, nu, dx, dy, u_left, u_right, v_top, v_bottom):
    """
    Compute the right-hand side of the combined system of ODEs for 2D Navier-Stokes equations.
    
    Parameters:
    y       - state vector containing velocities (u, v) and pressures (p) (array)
    Nx, Ny  - number of spatial points in x and y directions
    rho     - density
    nu      - kinematic viscosity
    dx, dy  - spatial step sizes in x and y directions
    u_left, u_right - boundary velocities in x-direction
    v_top, v_bottom - boundary velocities in y-direction
    
    Returns:
    dydt    - right-hand side of the ODE system (array)
    """
    u = y[:Nx*Ny].reshape((Nx, Ny))
    v = y[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    p = y[2*Nx*Ny:].reshape((Nx, Ny))
    
    dudt = np.zeros((Nx, Ny))
    dvdt = np.zeros((Nx, Ny))
    dpdt = np.zeros((Nx, Ny))
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Convective terms
            convective_u = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dx) + v[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dy)
            convective_v = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dx) + v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dy)
            
            # Pressure gradient terms
            pressure_gradient_u = (p[i+1, j] - p[i-1, j]) / (2 * rho * dx)
            pressure_gradient_v = (p[i, j+1] - p[i, j-1]) / (2 * rho * dy)
            
            # Diffusive terms
            diffusive_u = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
            diffusive_v = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
            
            dudt[i, j] = -convective_u - pressure_gradient_u + diffusive_u
            dvdt[i, j] = -convective_v - pressure_gradient_v + diffusive_v
            
            # Incompressibility constraint
            dpdt[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dx) + (v[i, j+1] - v[i, j-1]) / (2 * dy)
    
    # Enforcing boundary conditions using Lagrange multipliers
    dudt[0, :] = u_left - u[0, :]
    dudt[-1, :] = u_right - u[-1, :]
    dvdt[:, 0] = v_bottom - v[:, 0]
    dvdt[:, -1] = v_top - v[:, -1]
    
    # Combine dudt, dvdt, and dpdt into a single vector dydt
    dydt = np.concatenate((dudt.flatten(), dvdt.flatten(), dpdt.flatten()))
    
    return dydt


def M(Nx, Ny):
    diags = np.concatenate((np.ones(2 * Nx * Ny), np.zeros(Nx * Ny)))
    return np.diag(diags)


def F(t, y, yp, Nx, Ny, rho, nu, dx, dy, u_left, u_right, v_top, v_bottom):
    return M(Nx, Ny) @ yp - compute_rhs(y, Nx, Ny, rho, nu, dx, dy, u_left, u_right, v_top, v_bottom)


def F(t, y, yp, Nx, Ny, rho, nu, dx, dy, u_left, u_right, u_bottom, u_top, v_left, v_right, v_top, v_bottom):
    """
    Compute the right-hand side of the combined system of ODEs for 2D Navier-Stokes equations.
    
    Parameters:
    y       - state vector containing velocities (u, v) and pressures (p) (array)
    Nx, Ny  - number of spatial points in x and y directions
    rho     - density
    nu      - kinematic viscosity
    dx, dy  - spatial step sizes in x and y directions
    u_left, u_right - boundary velocities in x-direction
    v_top, v_bottom - boundary velocities in y-direction
    
    Returns:
    dydt    - right-hand side of the ODE system (array)
    """
    u = y[:Nx*Ny].reshape((Nx, Ny))
    v = y[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    # p = y[2*Nx*Ny:].reshape((Nx, Ny))
    du = yp[:Nx*Ny].reshape((Nx, Ny))
    dv = yp[Nx*Ny:2*Nx*Ny].reshape((Nx, Ny))
    p = yp[2*Nx*Ny:].reshape((Nx, Ny))
    
    # dudt = np.zeros((Nx, Ny))
    # dvdt = np.zeros((Nx, Ny))
    # dpdt = np.zeros((Nx, Ny))
    Fu = np.zeros((Nx, Ny))
    Fv = np.zeros((Nx, Ny))
    Fp = np.zeros((Nx, Ny))
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Convective terms
            convective_u = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dx) + v[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dy)
            convective_v = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dx) + v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dy)
            
            # Pressure gradient terms
            pressure_gradient_u = (p[i+1, j] - p[i-1, j]) / (2 * rho * dx)
            pressure_gradient_v = (p[i, j+1] - p[i, j-1]) / (2 * rho * dy)
            
            # Diffusive terms
            diffusive_u = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
            diffusive_v = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
            
            # dudt[i, j] = -convective_u - pressure_gradient_u + diffusive_u
            # dvdt[i, j] = -convective_v - pressure_gradient_v + diffusive_v
            Fu[i, j] = du[i, j] - (-convective_u - pressure_gradient_u + diffusive_u)
            Fv[i, j] = dv[i, j] - (-convective_v - pressure_gradient_v + diffusive_v)
            
            # Incompressibility constraint
            # dpdt[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dx) + (v[i, j+1] - v[i, j-1]) / (2 * dy)
            Fp[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dx) + (v[i, j+1] - v[i, j-1]) / (2 * dy)
    
    # Enforcing boundary conditions using Lagrange multipliers
    # dudt[0, :] = u_left - u[0, :]
    # dudt[-1, :] = u_right - u[-1, :]
    # dvdt[:, 0] = v_bottom - v[:, 0]
    # dvdt[:, -1] = v_top - v[:, -1]

    Fu[0, :] = u_top - u[0, :]
    Fu[-1, :] = u_bottom - u[-1, :]
    Fu[:, 0] = u_left - u[:, 0]
    Fu[:, -1] = u_right - u[:, -1]

    Fv[0, :] = v_top - v[0, :]
    Fv[-1, :] = v_bottom - v[-1, :]
    Fv[:, 0] = v_left - v[:, 0]
    Fv[:, -1] = v_right - v[:, -1]
    
    # pressure boundary conditions
    # TODO: This seems to be wrong!
    Fp[0, :] = p[0, :]
    Fp[-1, :] = p[-1, :]
    Fp[:, 0] = p[:, 0]
    Fp[:, -1] = p[:, -1]
    
    # Combine dudt, dvdt, and dpdt into a single vector dydt
    # dydt = np.concatenate((dudt.flatten(), dvdt.flatten(), dpdt.flatten()))
    F = np.concatenate((Fu.flatten(), Fv.flatten(), Fp.flatten()))
    
    return F

if __name__ == "__main__":
    # Parameters
    Lx, Ly = 1.0, 1.0  # Lengths of the domain in x and y directions
    # Nx, Ny = 50, 50  # Number of spatial points in x and y directions
    Nx, Ny = 4, 4  # Number of spatial points in x and y directions
    rho = 1.0  # Density
    nu = 0.1  # Kinematic viscosity

    # Boundary velocities
    u_left = 0.0
    u_right = 0.0
    u_bottom = 0.0
    u_top = 0.0

    v_left = 0.0
    v_right = 0.0
    v_bottom = 0.0
    # v_top = 1.0
    v_top = 0.0

    # Initialize the grid and build implicit function
    x, y, dx, dy = initialize_grid(Nx, Ny, Lx, Ly)
    f = lambda t, y, yp: F(t, y, yp, Nx, Ny, rho, nu, dx, dy, u_left, u_right, u_bottom, u_top, v_left, v_right, v_top, v_bottom)

    # time span
    t0 = 0
    t1 = 0.1
    t_span = (t0, t1)

    # initial conditions
    y0 = initial_conditions(Nx, Ny)
    # yp0 = compute_rhs(y0, Nx, Ny, rho, nu, dx, dy, u_left, u_right, v_top, v_bottom)
    yp0 = -f(t0, y0, np.zeros_like(y0))
    yp0 = np.zeros_like(y0)
    # yp0 = np.ones_like(y0)

    Jy0 = approx_derivative(lambda y: f(t0, y, yp0), y0)
    print(f"Jy0.shape: {Jy0.shape}")
    print(f"np.linalg.matrix_rank(J0): {np.linalg.matrix_rank(Jy0)}")

    Jyp0 = approx_derivative(lambda yp: f(t0, y0, yp), yp0)
    print(f"Jyp0.shape: {Jyp0.shape}")
    print(f"np.linalg.matrix_rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")

    J = Jy0 + Jyp0
    print(f"J.shape: {J.shape}")
    print(f"np.linalg.matrix_rank(J): {np.linalg.matrix_rank(J)}")
    
    assert J.shape[0] == np.linalg.matrix_rank(J)

    method = "BDF"
    # method = "Radau"

    # yp0 = np.zeros_like(y0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # y0, yp0, fnorm = consistent_initial_conditions(F, jac, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-3

    # solve the system
    start = time.time()
    sol = solve_dae(f, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, dense_output=True)
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

#     # export solution
#     import sys
#     from pathlib import Path
#     path = Path(sys.modules["__main__"].__file__)
#     header = "t, y1, y2, y3"
#     export_data = np.concatenate((t[None, :], y), axis=0)
#     np.savetxt(
#         path.parent / f"{path.stem}.txt",
#         export_data.T,
#         delimiter=", ",
#         header=header,
#         comments="",
#     )

#     # visualization
#     fig, ax = plt.subplots()

#     ax.plot(t, y[0], "-ok", label="y1 DAE" + f" ({method})", mfc="none")
#     ax.plot(t, y[1] * 1e4, "-ob", label="y2 DAE" + f" ({method})", mfc="none")
#     ax.plot(t, y[2], "-og", label="y3 DAE" + f" ({method})", mfc="none")
#     ax.plot(t_scipy, y_scipy[0], "xr", label="y1 scipy Radau", markersize=7)
#     ax.plot(t_scipy, y_scipy[1] * 1e4, "xy", label="y2 scipy Radau", markersize=7)
#     ax.plot(t_scipy, y_scipy[2], "xm", label="y3 scipy Radau", markersize=7)
#     ax.set_xscale("log")
#     ax.legend()
#     ax.grid()

#     plt.show()
