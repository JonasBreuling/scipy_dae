import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy_dae.integrate import solve_dae
from scipy.optimize._numdiff import approx_derivative


def generate_system(L, N, kappa, u_left, u_right, u_top, u_bottom):
    dx = dy = L / (N - 1)
    x = y = np.linspace(0, L, N)

    u_init = np.zeros((N, N))
    N4 = int(N / 4)
    u_init[N4:-N4, N4:-N4] = 1
    # TODO:
    up_init = np.zeros_like(u_init)

    y0 = u_init[1:-1, 1:-1]
    yp0 = up_init[1:-1, 1:-1]
    y0 = y0.reshape(-1)
    yp0 = yp0.reshape(-1)

    def redundant_coordinates(y, yp):
        y = y.reshape((N - 2, N - 2))
        yp = yp.reshape((N - 2, N - 2))

        u = np.empty((N, N))
        up = np.empty((N, N))

        u[:, 0] = u_left
        u[:, -1] = u_right
        u[0, :] = u_top
        u[-1, :] = u_bottom

        u[1:-1, 1:-1] = y
        up[1:-1, 1:-1] = yp

        # u = u.reshape(-1)
        # up = up.reshape(-1)

        return u, up
        
    def F(t, y, yp):
        # set boundary conditions
        u, up = redundant_coordinates(y, yp)

        # residual
        F = np.zeros_like(u)
        
        # all interior points
        for i in range(1, N-1):
            for j in range(1, N-1):
                F[i, j] = up[i, j] - kappa * (
                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (dx**2)
                    + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / (dy**2)
                )
        
        # do not solve for unknowns on the boundaries
        return F[1:-1, 1:-1].reshape(-1)
    
    return x, y, dx, dy, y0, yp0, redundant_coordinates, F

if __name__ == "__main__":
    # Parameters
    L = 1.0 # Lengths of the domain
    N = int(2e1) # Number of spatial points
    kappa = 1.14
    u_left = u_top = 1
    u_right = u_bottom = 0

    x, y, dx, dy, y0, yp0, redundant_coordinates, F = generate_system(L, N, kappa, u_left, u_right, u_top, u_bottom)

    # time span
    t0 = 0
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

    method = "BDF"
    # method = "Radau"

    # solver options
    atol = rtol = 1e-6

    # solve the system
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, dense_output=True, max_step=1e-0)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    sol_y = sol.y
    sol_yp = sol.yp
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
    u = np.zeros((N, N, nt))
    up = np.zeros((N, N, nt))
    for i in range(nt):
        u[:, :, i], up[:, :, i] = redundant_coordinates(sol_y[:, i], sol_yp[:, i])

    # visualize solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    plot = ax.plot_surface(X, Y, u[:, :, 0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim(0, L)
    
    def data_gen(framenumber, u, plot):
        #change soln variable for the next frame
        ax.clear()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim(0, L)
        plot = ax.plot_surface(X, Y, u[:, :, framenumber])
        return plot,

    pam_ani = animation.FuncAnimation(fig, data_gen, frames=nt, fargs=(u, plot),
                                      interval=30, blit=False)

    plt.show()