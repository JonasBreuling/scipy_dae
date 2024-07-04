import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy_dae.integrate import solve_dae
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import diags, block_diag, lil_matrix, csc_matrix


# see https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver/blob/master/docs_part1/vanilaCavityFlow_EN.md
# and https://barbagroup.github.io/essential_skills_RRC/numba/4/


# def generate_system(Lx, Ly, Nx, Ny, rho, nu, u_top):
def generate_system(Lx, Ly, Nx, Ny, Re):
    """Lid-Driven Cavity Flow."""
    # Grid size (Equispaced)
    dx = Lx / Nx
    dy = Ly / Ny

    # Coordinate of each grid (cell center)
    xce = (np.arange(Nx) + 0.5) * dx
    yce = (np.arange(Ny) + 0.5) * dy

    # Coordinate of each grid (cell corner)
    xco = (np.arange(Nx + 1)) * dx
    yco = (np.arange(Ny + 1)) * dy

    # Xce, Yce = np.meshgrid(xce, yce)
    # Xco, Yco = np.meshgrid(xco, yco)
    # fig, ax = plt.subplots()
    # ax.plot(Xce, Yce, "ok", label="cell centers")
    # ax.plot(Xco, Yco, "xk", label="cell corners")
    # ax.legend()
    # ax.grid()
    # ax.set_aspect("equal")
    # plt.show()
    # exit()

    # boundary conditions
    u_top = u_left = u_right = u_bottom = 0
    v_left = v_right = v_top = v_bottom = 0
    u_top = 1

    # initial conditions
    u_init = np.zeros((Nx + 1, Ny + 2))
    v_init = np.zeros((Nx + 2, Ny + 1))
    p_init = np.zeros((Nx, Ny))

    # u_int = np.random.rand(Nx + 1, Ny + 2)
    # v_int = np.random.rand(Nx + 2, Ny + 2)
    # p_int = np.random.rand(Nx, Ny)

    ut_init = np.zeros((Nx + 1, Ny + 2))
    vt_init = np.zeros((Nx + 2, Ny + 1))
    pt_init = np.zeros((Nx, Ny))

    y0 = np.concatenate((
        u_init[1:-1, 1:-1].flatten(), 
        v_init[1:-1, 1:-1].flatten(), 
        p_init.flatten(), 
    ))
    yp0 = np.concatenate((
        ut_init[1:-1, 1:-1].flatten(), 
        vt_init[1:-1, 1:-1].flatten(), 
        pt_init.flatten(), 
    ))

    def redundant_coordinates(y, yp):
        Nu = (Nx - 1) * Ny
        Nv = Nx * (Ny - 1)
        split = np.cumsum([Nu, Nv])
        u, v, p = np.array_split(y, split)
        ut, vt, pt = np.array_split(yp, split)

        u = u.reshape((Nx - 1, Ny))
        v = v.reshape((Nx, Ny - 1))
        p = p.reshape((Nx, Ny))
        ut = ut.reshape((Nx - 1, Ny))
        vt = vt.reshape((Nx, Ny - 1))
        pt = pt.reshape((Nx, Ny))

        # p = pt

        # build redundant coordinates
        u_red = np.zeros((Nx + 1, Ny + 2))
        v_red = np.zeros((Nx + 2, Ny + 1))
        p_red = np.zeros((Nx, Ny))
        ut_red = np.zeros((Nx + 1, Ny + 2))
        vt_red = np.zeros((Nx + 2, Ny + 1))
        pt_red = np.zeros((Nx, Ny))

        # interior velocites are the unknowns; all pressures are unknown
        u_red[1:-1, 1:-1] = u
        v_red[1:-1, 1:-1] = v
        p_red = p

        ut_red[1:-1, 1:-1] = ut
        vt_red[1:-1, 1:-1] = vt
        pt_red = pt

        # Dirichlet boundary conditions for velocities
        u_red[0, :] = 2 * u_left - u_red[1, :]
        u_red[-1, :] = 2 * u_right - u_red[-2, :]
        u_red[:, 0] = 2 * u_bottom - u_red[:, 1]
        u_red[:, -1] = 2 * u_top - u_red[:, -2]

        v_red[0, :] = 2 * v_left - v_red[1, :]
        v_red[-1, :] = 2 * v_right - v_red[-2, :]
        v_red[:, 0] = 2 * v_bottom - v_red[:, 1]
        v_red[:, -1] = 2 * v_top - v_red[:, -2]

        return u_red, v_red, p_red, ut_red, vt_red, pt_red


    def create_diff_operators():
        # u = u.reshape((Nx - 1, Ny))
        # v = v.reshape((Nx, Ny - 1))

        # Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
        # Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
        # D2x = block_diag([Dxx for _ in range(Ny)])
        # D2y = block_diag([Dyy for _ in range(Nx)]).T

        Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(Nx + 2, Nx + 2)) / dx**2
        Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(Nx + 2, Nx + 2)) / dy**2
        Du2x = block_diag([Dxx for _ in range(Ny + 1)])
        Du2y = block_diag([Dyy for _ in range(Ny + 1)]).T
        Dv2x = block_diag([Dxx for _ in range(Nx + 1)])
        Dv2y = block_diag([Dyy for _ in range(Nx + 1)]).T

        # Dx = diags([-1, 1], [0, 1], shape=(Nx, Nx)) / dx
        # Dy = diags([-1, 1], [0, 1], shape=(Ny, Ny)) / dy
        # Dx = block_diag([Dx for _ in range(Ny)])
        # Dy = block_diag([Dy for _ in range(Nx)]).T

        Dx = diags([-1, 1], [0, 1], shape=(Nx + 2, Nx + 2)) / dx
        Dy = diags([-1, 1], [0, 1], shape=(Ny + 2, Ny + 2)) / dy
        Dpx = diags([-1, 1], [0, 1], shape=(Nx, Nx)) / dx
        Dpy = diags([-1, 1], [0, 1], shape=(Ny, Ny)) / dy
        Dux = block_diag([Dx for _ in range(Ny + 1)])
        Duy = block_diag([Dy for _ in range(Ny + 1)]).T
        Dvx = block_diag([Dx for _ in range(Nx + 1)])
        Dvy = block_diag([Dy for _ in range(Nx + 1)]).T
        Dpx = block_diag([Dpx for _ in range(Ny)])
        Dpy = block_diag([Dpy for _ in range(Nx)]).T

        return Du2x, Du2y, Dv2x, Dv2y, Dux, Duy, Dvx, Dvy, Dpx, Dpy

    Du2x, Du2y, Dv2x, Dv2y, Dux, Duy, Dvx, Dvy, Dpx, Dpy = create_diff_operators()
    # Dxx, Dyy, Dx, Dy = create_diff_operators()

    def F(t, y, yp):
        # set boundary conditions
        u, v, p, ut, vt, pt = redundant_coordinates(y, yp)
        p = pt # TODO: Index reduction!

        # Fu = ut.flatten() - (Du2x + Du2y) @ u.flatten() / Re
        # Fv = vt.flatten() - (Dv2x + Dv2y) @ v.flatten() / Re

        # # Non-linear terms
        # Fu += (
        #     u.flatten() * (Dux @ u.flatten())
        #     + v.flatten() * (Duy @ u.flatten())
        # )
        # Fv += (
        #     u.flatten() * (Dvx @ v.flatten())
        #     + v.flatten() * (Dvy * v.flatten())
        # )

        # # Pressure terms
        # # TODO: Understand pressure gradient term; currently has wrong shape.
        # Fu.reshape((Nx + 1, Ny + 2))[:-1, 1:-1] += (Dpx @ p.flatten()).reshape((Nx, Ny))
        # Fv.reshape((Nx + 2, Ny + 1))[1:-1, :-1] += (Dpy @ p.flatten()).reshape((Nx, Ny))

        # # Continuity equation
        # Fp = (
        #     Dpx @ u.reshape((Nx + 1, Ny + 2))[:-1, 1:-1].flatten()
        #     + Dpy @ v.reshape((Nx + 2, Ny + 1))[1:-1, :-1].flatten()
        # )

        # # do not solve for unknowns on the boundaries
        # F = np.concatenate((
        #     Fu.reshape((Nx + 1, Ny + 2))[1:-1, 1:-1].flatten(), 
        #     Fv.reshape((Nx + 2, Ny + 1))[1:-1, 1:-1].flatten(), 
        #     Fp,
        # ))
        # return F

        # residual
        Fu = np.zeros_like(u)
        Fv = np.zeros_like(v)
        Fp = np.zeros_like(p)
    
        # all interior points of the u-velocity
        for i in range(1, Nx):
            for j in range(1, Ny + 1):
                Fu[i, j] = (
                    ut[i, j]
                    + u[i, j] * (u[i + 1, j] - u[i - 1, j]) / (2 * dx)
                    + v[i, j] * (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
                    # + (1 / rho) * (p[i, j - 1] - p[i - 1, j - 1]) / dx # note index shift in p!
                    # - nu * (
                    #     (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2
                    #     + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
                    # )
                    + (p[i, j - 1] - p[i - 1, j - 1]) / dx # note index shift in p!
                    - (
                        (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2
                        + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
                    ) / Re
                )

        # # add divergence part using operator
        # Fu[1:-1, 1:-1] += (
        #     ut.flatten() - (Du2x + Du2y) @ u.flatten() / Re
        # ).reshape((Nx + 1, Ny + 2))[1:-1, 1:-1]
    
        # all interior points of the v-velocity
        for i in range(1, Nx + 1):
            for j in range(1, Ny):
                Fv[i, j] = (
                    vt[i, j]
                    + u[i, j] * (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
                    + v[i, j] * (v[i, j + 1] - v[i, j - 1]) / (2 * dy)
                    # + (1 / rho) * (p[i - 1, j] - p[i - 1, j - 1]) / dy # note index shift in p!
                    # - nu * (
                    #     (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2
                    #     + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2
                    # )
                    + (p[i - 1, j] - p[i - 1, j - 1]) / dy # note index shift in p!
                    - (
                        (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2
                        + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2
                    ) / Re
                )

        # TODO: This leads to the rank deficiency!
        # continuity equation
        for i in range(Nx):
            for j in range(Ny):
                # Fp[i, j] = (
                #     (u[i + 1, j] - u[i, j]) / dx
                #     + (v[i, j + 1] - v[i, j]) / dy
                # )

                # if 1 <= i <= Nx - 2:
                #     # central differences in the interior
                #     u_x = (u[i + 1, j] - u[i - 1, j]) / dx
                # else:
                #     u_x = (u[i + 1, j] - u[i, j]) / dx

                # if 1 <= j <= Ny - 2:
                #     # central differences in the interior
                #     v_y = (u[i, j + 1] - v[i, j - 1]) / dy
                # else:
                #     v_y = (v[i, j + 1] - v[i, j]) / dy

                # TODO: Understand the index shift!
                u_x = (u[i + 1, j + 1] - u[i, j + 1]) / dx
                v_y = (v[i + 1, j + 1] - v[i + 1, j]) / dy
                Fp[i, j] = u_x + v_y
                # Fp[i, j] = p[i, j]

        # # pressure poission equation PPSE
        # # https://barbagroup.github.io/essential_skills_RRC/numba/4/
        # for i in range(Nx):
        #     for j in range(Ny):
        #         if i == 0:
        #             # forward differences on left boundary
        #             p_xx = (p[i + 2, j] - 2 * p[i + 1, j] + p[i, j]) / dx**2
        #         elif i == Nx - 1:
        #             # backward differences on left boundary
        #             p_xx = (p[i, j] - 2 * p[i - 1, j] + p[i - 2, j]) / dx**2
        #         else:
        #             # central differences in the interior
        #             p_xx = (p[i + 1, j] - 2 * p[i, j] + p[i - 1, j]) / dx**2

        #         if j == 0:
        #             # forward differences on left boundary
        #             p_yy = (p[i, j + 2] - 2 * p[i, j + 1] + p[i, j]) / dy**2
        #         elif j == Ny - 1:
        #             # backward differences on left boundary
        #             p_yy = (p[i, j] - 2 * p[i, j - 1] + p[i, j - 2]) / dy**2
        #         else:
        #             # central differences in the interior
        #             p_yy = (p[i, j + 1] - 2 * p[i, j] - p[i, j - 1]) / dy**2

        #         u_x = (u[i + 1, j] - u[i, j]) / dx
        #         u_y = (u[i, j + 1] - u[i, j]) / dy
        #         v_x = (v[i + 1, j] - v[i, j]) / dx
        #         v_y = (v[i, j + 1] - v[i, j]) / dy
        #         Fp[i, j] = p_xx + p_yy + rho * (u_x**2 + 2 * u_y * v_x + v_y**2)



        # # for i in range(Nx):
        # #     for j in range(Nx):
        #         # # Convective terms
        #         # convective_u = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dx) + v[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dy)
        #         # convective_v = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dx) + v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dy)
                
        #         # Pressure gradient terms
        #         # pressure_gradient_u = (p[i+1, j] - p[i-1, j]) / (2 * rho * dx)
        #         # pressure_gradient_v = (p[i, j+1] - p[i, j-1]) / (2 * rho * dy)
        #         # pressure_gradient_u = (p[i, j] - p[i-1, j]) / (2 * rho * dx)
        #         # pressure_gradient_v = (p[i, j] - p[i, j-1]) / (2 * rho * dy)
        #         pressure_gradient_u = (p[i, j] - p[i-1, j]) / (rho * dx)
        #         pressure_gradient_v = (p[i, j] - p[i, j-1]) / (rho * dy)
        #         # pressure_gradient_u = (p[i+1, j] - p[i, j]) / (rho * dx)
        #         # pressure_gradient_v = (p[i, j+1] - p[i, j]) / (rho * dy)
                
        #         # Diffusive terms
        #         # diffusive_u = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
        #         # diffusive_v = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
        #         diffusive_u = nu * ((u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy**2)
        #         diffusive_v = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)

        #         Fu[i, j] = ut[i, j] - (-convective_u - pressure_gradient_u + diffusive_u)
        #         Fv[i, j] = vt[i, j] - (-convective_v - pressure_gradient_v + diffusive_v)
                
        #         # # Incompressibility constraint
        #         # Fp[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dx) + (v[i, j+1] - v[i, j-1]) / (2 * dy)

        #         # PPE (Poisson Equation for Pressure)
        #         Fp[i, j] = (
        #             (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 
        #             + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2 
        #             - rho * (
        #                 (u[i+1, j] - u[i-1, j])**2 / (2 * dx)**2
        #                 + 2 * (u[i, j+1] - u[i, j-1]) / (2 * dy) * (v[i+1, j] - v[i-1, j]) / (2 * dx)
        #                 + (v[i, j+1] - v[i, j-1])**2 / (2 * dy)**2
        #             )
        #         )
            
        # do not solve for unknowns on the boundaries
        F = np.concatenate((
            Fu[1:-1, 1:-1].flatten(), 
            Fv[1:-1, 1:-1].flatten(), 
            Fp.flatten(),
        ))
        return F
    
    def animate_velocity_field(x, y, u, v, interval=50):
        fig, ax = plt.subplots()
        
        def update_quiver(num):
            ax.clear()
            ax.set_xlim(-0.25 * Lx, 1.25 * Lx)
            ax.set_ylim(-0.25 * Ly, 1.25 * Ly)
            ax.set_aspect("equal")
            ax.plot(x, y, "ok")
            contour = ax.contourf(x, y, np.sqrt(u[:, :, num]**2 + v[:, :, num]**2), alpha=0.5)
            # with np.errstate(divide='ignore'):
            quiver = ax.quiver(x, y, u[:, :, num], v[:, :, num])
            return quiver, contour
            # streamplot, = ax.streamplot(x, y, u[:, :, num], v[:, :, num])
            # return streamplot,

        anim = animation.FuncAnimation(fig, update_quiver, frames=u.shape[-1], interval=interval, blit=True)
        plt.show()
    
    return xce, yce, xco, yco, dx, dy, y0, yp0, redundant_coordinates, F, animate_velocity_field

if __name__ == "__main__":
    # Parameters
    # Lengths of the domain
    Lx = 1
    Ly = 1
    # Number of spatial points
    Nx = 5
    Ny = 5
    # Re = 100
    Re = 30

    xce, yce, xco, yco, dx, dy, y0, yp0, redundant_coordinates, F, animate_velocity_field = generate_system(Lx, Ly, Nx, Ny, Re)

    # time span
    t0 = 0
    t1 = 20
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))

    Jy0 = approx_derivative(lambda y: F(t0, y, yp0), y0, method="2-point")
    sparsity_Jy = csc_matrix(Jy0)
    print(f"Jy0.shape: {Jy0.shape}")
    print(f"np.linalg.matrix_rank(J0): {np.linalg.matrix_rank(Jy0)}")

    Jyp0 = approx_derivative(lambda yp: F(t0, y0, yp), yp0, method="2-point")
    sparsity_Jyp = csc_matrix(Jyp0)
    print(f"Jyp0.shape: {Jyp0.shape}")
    print(f"np.linalg.matrix_rank(Jyp0): {np.linalg.matrix_rank(Jyp0)}")

    jac_sparsity = (sparsity_Jy, sparsity_Jyp)
    # jac_sparsity = None

    # J = Jy0 + Jyp0
    # print(f"J.shape: {J.shape}")
    # print(f"np.linalg.matrix_rank(J): {np.linalg.matrix_rank(J)}")
    
    # # assert J.shape[0] == np.linalg.matrix_rank(J)
    # # exit()

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

    # Xce, Yce = np.meshgrid(xce, yce)
    # Xco, Yco = np.meshgrid(xco, yco)
    # fig, ax = plt.subplots()
    # ax.plot(Xce, Yce, "ok", label="cell centers")
    # ax.plot(Xco, Yco, "xk", label="cell corners")
    # ax.legend()
    # ax.grid()
    # ax.set_aspect("equal")
    # plt.show()
    # # exit()

    # reconstruct solution
    nt = len(t)
    u = np.zeros((Nx + 1, Ny + 2, nt))
    v = np.zeros((Nx + 2, Ny + 1, nt))
    p = np.zeros((Nx, Ny, nt))
    ut = np.zeros((Nx + 1, Ny + 2, nt))
    vt = np.zeros((Nx + 2, Ny + 1, nt))
    pt = np.zeros((Nx, Ny, nt))
    for i in range(nt):
        u[:, :, i], v[:, :, i] , p[:, :, i], ut[:, :, i], vt[:, :, i], pt[:, :, i] = redundant_coordinates(sol_y[:, i], sol_yp[:, i])


    # 1. interpolate velocity at cell center/cell corner
    # uce = (u(1:end-1,2:end-1)+u(2:end,2:end-1))/2;
    uce = (
        # u(1:end-1,2:end-1)+u(2:end,2:end-1)
        u[:-1, 1:-1] + u[1:, 1:-1]
    ) / 2
    uco = (
        # u(:,1:end-1) + u(:,2:end)
        u[:, :-1] + u[:, 1:]
    ) / 2
    vco = (
        # v(1:end-1,:)+v(2:end,:)
        v[:-1, :] + v[1:, :]
    ) / 2
    vce = (
        # v(2:end-1,1:end-1)+v(2:end-1,2:end)
        v[1:-1, :-1] + v[1:-1, 1:]
    ) / 2

    uco = uco.transpose(1, 0, 2)
    vco = vco.transpose(1, 0, 2)
    uce = uce.transpose(1, 0, 2)
    vce = vce.transpose(1, 0, 2)

    Xce, Yce = np.meshgrid(xce, yce)
    animate_velocity_field(Xce, Yce, uce, vce)

    # Xco, Yco = np.meshgrid(xco, yco)
    # animate_velocity_field(Xco, Yco, uco, vco)
