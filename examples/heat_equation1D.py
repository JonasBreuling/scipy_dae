import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae
from scipy.optimize._numdiff import approx_derivative


def generate_system(Lx, Nx, kappa, u0, u1):
    """Munz2019 - Section 7.2."""
    dx = Lx / (Nx - 1)
    x = np.linspace(0, Lx, Nx)
    u_init = -1.5 * x + 2 + np.sin(np.pi * x)
    up_init =  -1.5 + np.pi * np.cos(np.pi * x)
    y0 = u_init[1:-1]
    yp0 = up_init[1:-1]
    
    def u_exact(t, x):
        return -1.5 * x + 2 + np.exp(-1.14 * np.pi**2 * t) * np.sin(np.pi * x)

    def redundant_coordinates(y, yp):
        u = np.empty(Nx)
        u[0] = u0
        u[-1] = u1
        u[1:-1] = y

        up = np.empty(Nx)
        up[0] = u0
        up[-1] = u1
        up[1:-1] = yp

        return u, up
        
    def F(t, y, yp):
        # set boundary conditions
        u, up = redundant_coordinates(y, yp)

        # residual
        F = np.zeros_like(u)
        
        # all interior points
        for i in range(1, Nx-1):
            F[i] = up[i] - kappa * (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx**2)
        
        # do not solve for unknowns on the boundaries
        return F[1:-1]
    
    return x, dx, y0, yp0, redundant_coordinates, u_exact, F

if __name__ == "__main__":
    # Parameters
    Lx = 1.0 # Lengths of the domain
    Nx = int(1e2) # Number of spatial points
    kappa = 1.14
    u0 = 2
    u1 = 0.5

    x, dx, y0, yp0, redundant_coordinates, u_exact, F = generate_system(Lx, Nx, kappa, u0, u1)

    # time span
    t0 = 0
    # t1 = 0.1
    t1 = 1
    t_span = (t0, t1)

    Jy0 = approx_derivative(lambda y: F(t0, y, yp0), y0)
    print(f"Jy0.shape: {Jy0.shape}")
    print(f"np.linalg.matrix_rank(J0): {np.linalg.matrix_rank(Jy0)}")

    Jyp0 = approx_derivative(lambda yp: F(t0, y0, yp), yp0)
    print(f"Jyp0.shape: {Jyp0.shape}")
    print(f"np.linalg.matrix_rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")

    J = Jy0 + Jyp0
    print(f"J.shape: {J.shape}")
    print(f"np.linalg.matrix_rank(J): {np.linalg.matrix_rank(J)}")
    
    assert J.shape[0] == np.linalg.matrix_rank(J)

    # method = "BDF"
    method = "Radau"

    # solver options
    atol = rtol = 1e-6

    # solve the system
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, dense_output=True, max_step=1e-0)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    yp = sol.yp
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # reconstruct solution
    nt = len(t)
    u = np.zeros((Nx, nt))
    up = np.zeros((Nx, nt))
    for i, (yi, ypi) in enumerate(zip(y.T, yp.T)):
        u[:, i], up[:, i] = redundant_coordinates(yi, ypi)

    # visualize solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T, X = np.meshgrid(t, x)
    u_true = np.zeros_like(u)
    for i in range(Nx):
        for j in range(nt):
            u_true[i, j] = u_exact(T[i, j], X[i, j])
    error = np.linalg.norm(u - u_true)
    print(f"error: {error}")

    ax.plot_surface(T, X, u)
    ax.plot_surface(T, X, u_true, alpha=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')

    plt.show()