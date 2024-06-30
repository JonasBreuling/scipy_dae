import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy_dae.integrate import solve_dae
from scipy.optimize._numdiff import approx_derivative


# see https://www.ams.org/journals/bull/1967-73-06/S0002-9904-1967-11853-6/S0002-9904-1967-11853-6.pdf
# see https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650071008 for boundary conditions
# see https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver/blob/master/docs_part1/vanilaCavityFlow_EN.md
# see https://chatgpt.com/c/285a219d-4d5a-4415-aea2-1dd991287617


def generate_system(L, N, rho, nu, u0):
    """Lid-Driven Cavity Flow."""
    # spacial discretization
    dx = dy = L / (N - 1)
    x = y = np.linspace(0, L, N)
    y = -y

    # boundary conditions
    u_top = u0
    u_left = u_right = u_bottom = 0
    v_left = v_right = v_top = v_bottom = 0
    dpdx_left = dpdx_right = dpdy_top = dpdy_bottom = 0

    # initial conditions
    u_init = np.zeros((N, N))
    v_init = np.zeros((N, N))
    p_init = np.ones((N, N))

    ut_init = np.zeros((N, N))
    vt_init = np.zeros((N, N))
    pt_init = np.ones((N, N))

    y0 = np.concatenate((
        u_init[1:-1, 1:-1].flatten(), 
        v_init[1:-1, 1:-1].flatten(), 
        p_init[1:-1, 1:-1].flatten(), 
    ))
    yp0 = np.concatenate((
        ut_init[1:-1, 1:-1].flatten(), 
        vt_init[1:-1, 1:-1].flatten(), 
        pt_init[1:-1, 1:-1].flatten(), 
    ))

    def redundant_coordinates(y, yp):
        N2N2 = (N - 2) * (N - 2)
        split = np.cumsum([N2N2, N2N2, N2N2])[:-1]
        u, v, p = np.array_split(y, split)
        ut, vt, pt = np.array_split(yp, split)

        u = u.reshape((N - 2, N - 2))
        v = v.reshape((N - 2, N - 2))
        p = p.reshape((N - 2, N - 2))
        ut = ut.reshape((N - 2, N - 2))
        vt = vt.reshape((N - 2, N - 2))
        pt = pt.reshape((N - 2, N - 2))

        u_red = np.zeros((N, N))
        v_red = np.zeros((N, N))
        p_red = np.zeros((N, N))
        ut_red = np.zeros((N, N))
        vt_red = np.zeros((N, N))
        pt_red = np.zeros((N, N))

        u_red[1:-1, 1:-1] = u
        v_red[1:-1, 1:-1] = v
        p_red[1:-1, 1:-1] = p

        ut_red[1:-1, 1:-1] = ut
        vt_red[1:-1, 1:-1] = vt
        pt_red[1:-1, 1:-1] = pt

        # Dirichlet boundary conditions for velocities
        u_red[:, 0] = u_left
        u_red[:, -1] = u_right
        u_red[0, :] = u_top
        u_red[-1, :] = u_bottom

        v_red[:, 0] = v_left
        v_red[:, -1] = v_right
        v_red[0, :] = v_top
        v_red[-1, :] = v_bottom

        # Neumann boundary conditions for pressure
        p_red[:, 0] = p_red[:, 1]
        p_red[:, -1] = p_red[:, -2]
        p_red[0, :] = p_red[1, :]
        p_red[-1, :] = p_red[-2, :]

        # pt_red[:, 0] = pt_red[:, 1]
        # pt_red[:, -1] = pt_red[:, -2]
        # pt_red[0, :] = pt_red[1, :]
        # pt_red[-1, :] = pt_red[-2, :]

        return u_red, v_red, p_red, ut_red, vt_red, pt_red
        
    def F(t, y, yp):
        # set boundary conditions
        u, v, p, ut, vt, pt = redundant_coordinates(y, yp)
        p = pt # TODO: Index reduction!

        # residual
        Fu = np.zeros_like(u)
        Fv = np.zeros_like(v)
        Fp = np.zeros_like(p)
        
        # all interior points
        for i in range(1, N-1):
            for j in range(1, N-1):
                # Convective terms
                convective_u = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dx) + v[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dy)
                convective_v = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dx) + v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dy)
                
                # Pressure gradient terms
                pressure_gradient_u = (p[i+1, j] - p[i-1, j]) / (2 * rho * dx)
                pressure_gradient_v = (p[i, j+1] - p[i, j-1]) / (2 * rho * dy)
                
                # Diffusive terms
                diffusive_u = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                diffusive_v = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)

                Fu[i, j] = ut[i, j] - (-convective_u - pressure_gradient_u + diffusive_u)
                Fv[i, j] = vt[i, j] - (-convective_v - pressure_gradient_v + diffusive_v)
                
                # # Incompressibility constraint
                # Fp[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dx) + (v[i, j+1] - v[i, j-1]) / (2 * dy)

                # PPE (Poisson Equation for Pressure)
                Fp[i, j] = (
                    (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 
                    + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2 
                    - rho * (
                        (u[i+1, j] - u[i-1, j])**2 / (2 * dx)**2
                        + 2 * (u[i, j+1] - u[i, j-1]) / (2 * dy) * (v[i+1, j] - v[i-1, j]) / (2 * dx)
                        + (v[i, j+1] - v[i, j-1])**2 / (2 * dy)**2
                    )
                )
            
        # do not solve for unknowns on the boundaries
        F = np.concatenate((
            Fu[1:-1, 1:-1].flatten(), 
            Fv[1:-1, 1:-1].flatten(), 
            Fp[1:-1, 1:-1].flatten(),
        ))
        return F
    
    def animate_velocity_field(x, y, u, v, interval=50):
        fig, ax = plt.subplots()
        quiver = ax.quiver(x, y, u[:, :, 0], v[:, :, 0])

        def update_quiver(num):
            quiver.set_UVC(u[:, :, num], v[:, :, num])
            return quiver,

        anim = animation.FuncAnimation(fig, update_quiver, frames=u.shape[-1], interval=interval, blit=True)
        plt.show()
    
    return x, y, dx, dy, y0, yp0, redundant_coordinates, F, animate_velocity_field

if __name__ == "__main__":
    # Parameters
    L = 1.0 # Lengths of the domain
    # N = int(1e1) # Number of spatial points
    N = 12
    nu = 0.1
    rho = 1
    u0 = 1 # horizontal velocity at the top surface

    x, y, dx, dy, y0, yp0, redundant_coordinates, F, animate_velocity_field = generate_system(L, N, rho, nu, u0)

    # time span
    t0 = 0
    t1 = 10
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

    # exit()

    # method = "BDF"
    method = "Radau"

    # solver options
    atol = rtol = 1e-3

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
    v = np.zeros((N, N, nt))
    p = np.zeros((N, N, nt))
    ut = np.zeros((N, N, nt))
    vt = np.zeros((N, N, nt))
    pt = np.zeros((N, N, nt))
    for i in range(nt):
        u[:, :, i], v[:, :, i] , p[:, :, i], ut[:, :, i], vt[:, :, i], pt[:, :, i] = redundant_coordinates(sol_y[:, i], sol_yp[:, i])

    X, Y = np.meshgrid(x, y)
    animate_velocity_field(X, Y, u, v)
