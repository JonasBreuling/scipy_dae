import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy_dae.integrate import solve_dae
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csc_matrix


def generate_system(Lx, Ly, nx, ny, nu, BC):
    # number of nodes
    Nx = nx + 1
    Ny = ny + 1

    # grid size (Equispaced)
    dx = Lx / nx
    dy = Ly / ny

    # coordinate of each grid (cell center)
    xij = (np.arange(nx) + 0.5) * dx
    yij = (np.arange(ny) + 0.5) * dy

    # coordinate of each grid (cell corner)
    xi2j2 = (np.arange(Nx)) * dx
    yi2j2 = (np.arange(Ny)) * dy

    # # visualize mesh for debugging purpose
    # Xij, Yij = np.meshgrid(xij, yij)
    # Xi2j2, Yi2j2 = np.meshgrid(xi2j2, yi2j2)
    # fig, ax = plt.subplots()
    # ax.plot(Xij, Yij, "ok", label="cell centers")
    # ax.plot(Xi2j2, Yi2j2, "xk", label="cell corners")
    # ax.legend()
    # ax.grid()
    # ax.set_aspect("equal")
    # plt.show()
    # exit()

    # boundary conditions
    u_left = BC["u_left"]
    u_right = BC["u_right"]
    u_bot = BC["u_bot"]
    u_top = BC["u_top"]

    v_left = BC["v_left"]
    v_right = BC["v_right"]
    v_bot = BC["v_bot"]
    v_top = BC["v_top"]

    # initial conditions
    u_init = np.zeros((nx + 1, ny + 2))
    v_init = np.zeros((nx + 2, ny + 1))
    p_init = np.zeros((nx, ny))

    ut_init = np.zeros((nx + 1, ny + 2))
    vt_init = np.zeros((nx + 2, ny + 1))
    pt_init = np.zeros((nx, ny))

    y0 = np.concatenate((
        u_init[1:nx, 1:ny + 1].flatten(), 
        v_init[1:nx + 1, 1:ny].flatten(), 
        p_init.flatten(), 
    ))
    yp0 = np.concatenate((
        ut_init[1:nx, 1:ny + 1].flatten(), 
        vt_init[1:nx + 1, 1:ny].flatten(), 
        pt_init.flatten(), 
    ))

    def redundant_coordinates(y, yp):
        # unpack state vector and derivatives
        nu = (nx - 1) * ny
        nv = nx * (ny - 1)
        split = np.cumsum([nu, nv])
        u, v, p = np.array_split(y, split)
        ut, vt, pt = np.array_split(yp, split)

        # reshape 2D
        u = u.reshape((nx - 1, ny))
        v = v.reshape((nx, ny - 1))
        p = p.reshape((nx, ny))
        ut = ut.reshape((nx - 1, ny))
        vt = vt.reshape((nx, ny - 1))
        pt = pt.reshape((nx, ny))

        # build redundant coordinates
        u_red = np.zeros((nx + 1, ny + 2))
        v_red = np.zeros((nx + 2, ny + 1))
        p_red = np.zeros((nx, ny))
        ut_red = np.zeros((nx + 1, ny + 2))
        vt_red = np.zeros((nx + 2, ny + 1))
        pt_red = np.zeros((nx, ny))

        # interior velocites are the unknowns; all pressures are unknown
        u_red[1:-1, 1:-1] = u
        v_red[1:-1, 1:-1] = v
        p_red = p

        ut_red[1:-1, 1:-1] = ut
        vt_red[1:-1, 1:-1] = vt
        pt_red = pt

        # Dirichlet boundary conditions for velocities
        if u_left is None:
            u_red[0, :] = u_red[1, :]
        else:
            u_red[0, :] = 2 * u_left - u_red[1, :]
        if u_right is None:
            u_red[-1, :] = u_red[-2, :]
        else:
            u_red[-1, :] = 2 * u_right - u_red[-2, :]
        if u_bot is None:
            u_red[:, 0] = u_red[:, 1]
        else:
            u_red[:, 0] = 2 * u_bot - u_red[:, 1]
        if u_top is None:
            u_red[:, -1] = u_red[:, -2]
        else:
            u_red[:, -1] = 2 * u_top - u_red[:, -2]

        if v_left is None:
            v_red[0, :] = v_red[1, :]
        else:
            v_red[0, :] = 2 * v_left - v_red[1, :]
        if v_right is None:
            v_red[-1, :] = v_red[-2, :]
        else:
            v_red[-1, :] = 2 * v_right - v_red[-2, :]
        if v_bot is None:
            v_red[:, 0] = v_red[:, 1]
        else:
            v_red[:, 0] = 2 * v_bot - v_red[:, 1]
        if v_top is None:
            v_red[:, -1] = v_red[:, -2]
        else:
            v_red[:, -1] = 2 * v_top - v_red[:, -2]

        return u_red, v_red, p_red, ut_red, vt_red, pt_red

    def F(t, y, yp):
        # set boundary conditions
        u, v, p, ut, vt, pt = redundant_coordinates(y, yp)
        p = pt # note: Index reduction!

        # interpolate velocities
        uij = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
        u2ij = uij**2
        vij = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
        v2ij = vij**2
        ui2j2 = 0.5 * (u[:, :-1] + u[:, 1:])
        vi2j2 = 0.5 * (v[:-1] + v[1:])

        # momentum equation for u
        Fu = (
            ut[1:-1, 1:-1]
            + (u2ij[1:] - u2ij[:-1]) / dx
            + (ui2j2[1:-1, 1:] * vi2j2[1:-1, 1:] - ui2j2[1:-1, :-1] * vi2j2[1:-1, :-1]) / dy
            + (p[1:] - p[:-1]) / dx
            - nu * (
                (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
                + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
            )
        )

        # momentum equation for v
        Fv = (
            vt[1:-1, 1:-1]
            + (ui2j2[1:, 1:-1] * vi2j2[1:, 1:-1] - ui2j2[:-1, 1:-1] * vi2j2[:-1, 1:-1]) / dx
            + (v2ij[:, 1:] - v2ij[:, :-1]) / dy
            + (p[:, 1:] - p[:, :-1]) / dy
            - nu * (
                (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
                + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
            )
        )

        # continuity equation
        Fp = (
            (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
            + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
        )

        return np.concatenate((
            Fu.flatten(), 
            Fv.flatten(), 
            Fp.flatten(),
        ))
    
    def animate(x, y, u, v, p, interval=1):
        fig, ax = plt.subplots()

        def update(num):
            ax.clear()
            ax.set_xlim(-0.25 * Lx, 1.25 * Lx)
            ax.set_ylim(-0.25 * Ly, 1.25 * Ly)
            ax.set_aspect("equal")
            # ax.plot(x, y, "ok")

            # # with np.errstate(divide='ignore'):
            # quiver = ax.quiver(x, y, u[:, :, num], v[:, :, num])

            contourf = ax.contourf(x, y, np.sqrt(u[:, :, num]**2 + v[:, :, num]**2), alpha=0.5)
            streamplot = ax.streamplot(x, y, u[:, :, num], v[:, :, num], density=1.5)
            return contourf, streamplot

        anim = animation.FuncAnimation(fig, update, frames=u.shape[-1], interval=interval, blit=False)
        plt.show()
    
    return xij, yij, xi2j2, yi2j2, dx, dy, y0, yp0, redundant_coordinates, F, animate


if __name__ == "__main__":
    ############
    # parameters
    ############
    # lengths of the domain
    Lx = 1
    Ly = 1
    # Lx = 4
    # Ly = 1

    # number of cell centers
    # nx = 3
    # ny = 2
    # nx = 10
    # ny = 10
    # nx = 20
    # ny = 20
    nx = 50
    ny = 50

    # nx = 40
    # ny = 10

    # kinematic viscosity
    # nu = 1e-0
    # nu = 1e-1
    nu = 1e-2
    # nu = 1e-3
    # nu = 1e-4

    # boundary conditions:
    # - lid-driven cavity flow
    BC = {
        "u_top": 1,
        "u_bot": 0,
        "u_left": 0,
        "u_right": 0,
        "v_top": 0,
        "v_bot": 0,
        "v_left": 0,
        "v_right": 0,
    }

    # # - channel flow  
    # u_max = 1
    # y_channel = np.linspace(0, Ly, ny + 2)
    # u_left_profile = u_max * (1 - ((y_channel / Ly - 0.5) * 2)**2)
    # BC = {
    #     "u_top": 0,
    #     "u_bot": 0,
    #     "u_left": u_left_profile,
    #     "u_right": None,
    #     "v_top": 0,
    #     "v_bot": 0,
    #     "v_left": 0,
    #     "v_right": 0,
    # }

    xij, yij, xi2j2, yi2j2, dx, dy, y0, yp0, redundant_coordinates, F, animate = generate_system(Lx, Ly, nx, ny, nu, BC)

    # time span
    t0 = 0
    t1 = 20
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e2))

    Jy0 = approx_derivative(lambda y: F(t0, y, yp0), y0, method="2-point")
    sparsity_Jy = csc_matrix(Jy0)
    print(f"Jy0.shape: {Jy0.shape}")
    # print(f"np.linalg.matrix_rank(J0): {np.linalg.matrix_rank(Jy0)}")

    Jyp0 = approx_derivative(lambda yp: F(t0, y0, yp), yp0, method="2-point")
    sparsity_Jyp = csc_matrix(Jyp0)
    print(f"Jyp0.shape: {Jyp0.shape}")
    # print(f"np.linalg.matrix_rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")

    jac_sparsity = (sparsity_Jy, sparsity_Jyp)
    # jac_sparsity = None
    
    # method = "BDF"
    method = "Radau"

    # solver options
    atol = rtol = 1e-3

    # solve the system
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, jac_sparsity=jac_sparsity, first_step=1e-5)
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
    u = np.zeros((nx + 1, ny + 2, nt))
    v = np.zeros((nx + 2, ny + 1, nt))
    p = np.zeros((nx, ny, nt))
    ut = np.zeros((nx + 1, ny + 2, nt))
    vt = np.zeros((nx + 2, ny + 1, nt))
    pt = np.zeros((nx, ny, nt))
    for i in range(nt):
        u[:, :, i], v[:, :, i] , p[:, :, i], ut[:, :, i], vt[:, :, i], pt[:, :, i] = redundant_coordinates(sol_y[:, i], sol_yp[:, i])

    # interpolate velocity at cell centers and cell corners
    uij = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
    ui2j2 = 0.5 * (u[:, :-1] + u[:, 1:])
    vij = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
    vi2j2 = 0.5 * (v[:-1, :] + v[1:, :])

    # transpose data for "xy" meshgrid and streamplot
    ui2j2 = ui2j2.transpose(1, 0, 2)
    vi2j2 = vi2j2.transpose(1, 0, 2)
    uij = uij.transpose(1, 0, 2)
    vij = vij.transpose(1, 0, 2)
    p = p.transpose(1, 0, 2)

    Xij, Yij = np.meshgrid(xij, yij, indexing="xy")
    animate(Xij, Yij, uij, vij, p)
