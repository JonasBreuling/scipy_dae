import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags, eye, kron
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions

# SCENARIO = "channel"
SCENARIO = "lid cavity"

RESOLUTION = 1


def D_forward(N, h):
    D = np.ones(N) / h
    return spdiags([-D, D], [0, 1], format="coo")

def D_backward(N, h):
    D = np.ones(N) / h
    return spdiags([D, -D], [1, 0], format="coo")

def D_central(N, h):
    D = np.ones(N) / (2 * h)
    return spdiags([D, -D], [1, -1], format="coo")

def DD_central(N, h):
    D = np.ones(N) / h**2
    return spdiags([D, -2 * D, D], [-1, 0, 1], format="coo")


class IncompressibleFluid:
    """The implementation follows mathworks.

    The staggered grid with ghost cells

    •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |   
    ↑ - ❖---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |    
    ↑ - 0---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •

    References:
    -----------
    mathworks: https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver/blob/master/docs_part1/vanilaCavityFlow_EN.md
    """
    def __init__(self, Nx, Ny, Lx, Ly, rho=1, nu=1e-4):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # # Coordinate of each grid (cell center)
        # self.xce = (np.arange(Nx) + 0.5) * self.dx
        # self.yce = (np.arange(Ny) + 0.5) * self.dy
        # self.Xce, self.Yce = np.meshgrid(self.xce, self.yce)

        # Coordinate of each grid (cell corner)
        self.row_range = (np.arange(Ny + 1)) * self.dy
        self.column_range = (np.arange(Nx + 1)) * self.dx
        # self.Xco, self.Yco = np.meshgrid(self.xco, self.yco, indexing="ij")
        self.Row_range, self.Column_range = np.meshgrid(self.row_range, self.column_range, indexing="ij")
        # self.Row_range, self.Column_range = np.meshgrid(self.row_range, self.column_range, indexing="xy")

        self.rho = rho
        self.nu = nu
        self.mu = nu / rho

        # shapes
        self.shape_u = (Ny + 2, Nx + 1)
        self.shape_v = (Ny + 1, Nx + 2)
        self.shape_p = (Ny, Nx)
        self.shape_u_interior = (Ny    , Nx - 1)
        self.shape_v_interior = (Ny - 1, Nx    )
        self.shape_p_interior = (Ny, Nx)
        
        # number of unknowns
        self.Nu = np.prod(self.shape_u)
        self.Nv = np.prod(self.shape_v)
        self.Np = np.prod(self.shape_p)
        self.N = self.Nu + self.Nv + self.Np
        self.Nu_interior = np.prod(self.shape_u_interior)
        self.Nv_interior = np.prod(self.shape_v_interior)
        self.Np_interior = np.prod(self.shape_p_interior)
        self.N_interior = self.Nu_interior + self.Nv_interior + self.Np_interior
        self.split = np.cumsum([self.Nu_interior, self.Nv_interior])

        # boundary DOF mappings
        # TODO: Extend this to arbitrary boundary conditions later
        # self.inner_DOFs_u = np.s_[1:Ny + 1, 1:Nx    ]
        # self.inner_DOFs_v = np.s_[1:Ny    , 1:Nx + 1]
        # self.inner_DOFs_p = np.s_[:, :]
        self.inner_DOFs_u = np.s_[1:-1, 1:-1]
        self.inner_DOFs_v = np.s_[1:-1, 1:-1]
        self.inner_DOFs_p = np.s_[:, :]

        self.u = np.zeros(self.shape_u)
        self.v = np.zeros(self.shape_v)
        self.p = np.zeros(self.shape_p)
        self.U = np.zeros(self.Nu)
        self.V = np.zeros(self.Nv)
        self.P = np.zeros(self.Np)
        self.Ut = np.zeros(self.Nu)
        self.Vt = np.zeros(self.Nv)
        self.Pt = np.zeros(self.Np)

        # operators
        # - first derivatives
        self._Du_x = kron(
            eye(Ny + 2), D_central(Nx + 1, self.dx), format="csr"
        )
        self._Du_y = kron(
            D_central(Ny + 2, self.dy), eye(Nx + 1), format="csr"
        )

        self._Dv_x = kron(
            eye(Ny + 1), D_central(Nx + 2, self.dx), format="csr"
        )
        self._Dv_y = kron(
            D_central(Ny + 1, self.dy), eye(Nx + 2), format="csr"
        )

        self._Dp_x = kron(
            eye(Ny), D_forward(Nx, self.dx), format="csr"
        )
        self._Dp_y = kron(
            D_forward(Ny, self.dy), eye(Nx), format="csr"
        )

        # - second derivatives
        self._DDu_x = kron(
            eye(Ny + 2), DD_central(Nx + 1, self.dx), format="csr"
        )
        self._DDu_y = kron(
            DD_central(Ny + 2, self.dy), eye(Nx + 1), format="csr"
        )

        self._DDv_x = kron(
            eye(Ny + 1), DD_central(Nx + 2, self.dx), format="csr"
        )
        self._DDv_y = kron(
            DD_central(Ny + 1, self.dy), eye(Nx + 2), format="csr"
        )

    def create_redundant_coordinates(self, y, yp):
        # extract interior points
        u_interior, v_interior, p_interior = np.array_split(y, self.split)
        ut_interior, vt_interior, pt_interior = np.array_split(yp, self.split)

        # reshape interior points
        u_interior = u_interior.reshape(self.shape_u_interior)
        v_interior = v_interior.reshape(self.shape_v_interior)
        p_interior = p_interior.reshape(self.shape_p_interior)
        ut_interior = ut_interior.reshape(self.shape_u_interior)
        vt_interior = vt_interior.reshape(self.shape_v_interior)
        pt_interior = pt_interior.reshape(self.shape_p_interior)

        # build redundant coordinates
        u = self.U.reshape(self.shape_u)
        v = self.V.reshape(self.shape_v)
        p = self.P.reshape(self.shape_p)
        ut = self.Ut.reshape(self.shape_u)
        vt = self.Vt.reshape(self.shape_v)
        pt = self.Pt.reshape(self.shape_p)
        assert np.shares_memory(self.U, u)
        assert np.shares_memory(self.V, v)
        assert np.shares_memory(self.P, p)
        assert np.shares_memory(self.Ut, ut)
        assert np.shares_memory(self.Vt, vt)
        assert np.shares_memory(self.Pt, pt)

        # interior velocites are the unknowns; all pressures are unknown
        u[1:-1, 1:-1] = u_interior
        u[self.inner_DOFs_u] = u_interior
        v[self.inner_DOFs_v] = v_interior
        p[self.inner_DOFs_p] = p_interior
        assert np.shares_memory(u[self.inner_DOFs_u], u)
        assert np.shares_memory(v[self.inner_DOFs_v], v)
        assert np.shares_memory(p[self.inner_DOFs_p], p)

        ut[self.inner_DOFs_u] = ut_interior
        vt[self.inner_DOFs_v] = vt_interior
        pt[self.inner_DOFs_p] = pt_interior
        assert np.shares_memory(ut[self.inner_DOFs_u], ut)
        assert np.shares_memory(vt[self.inner_DOFs_v], vt)
        assert np.shares_memory(pt[self.inner_DOFs_p], pt)

        # # TODO: Add these functions as arguments to fluid
        # u_left = v_left = 0
        # u_right = v_right = 0
        # u_bottom = v_bottom = 0
        # u_top = v_top = 0
        # # u_top = 1
        # # u_bottom = 1
        # u_left = 1
        # # u_right = 1
        # du_right = 0
        # dv_right = 0

        # # u_left = u_right = +1
        # # u_bottom = u_top = +1
        # # v_left = v_right = -1
        # # v_bottom = v_top = -1

        match SCENARIO:
            case "channel":
                # Dirichlet boundary conditions
                u_left = 1
                u_top = u_bottom = 0
                v_top = v_bottom = v_left = 0

                u[ 0, :] = 2 * u_top - u[1, :]
                u[-1, :] = 2 * u_bottom - u[-2, :]
                u[1:-1,  0] = u_left

                v[ 0, 1:-1] = v_top
                v[-1, 1:-1] = v_bottom
                v[:,  0] = 2 * v_left - v[:, 1]

                # Neumann boundary conditions (du/dx = du_right, dv/dx = dv_right)
                du_right = dv_right = 0
                u[1:-1, -1] = 2 * self.dx * du_right + u[1:-1, -3]
                v[:, -1] = self.dx * dv_right + v[:, -2]
            case "lid cavity":
                # Dirichlet boundary conditions
                u_top = 1
                u_bottom = u_left = u_right = 0
                v_top = v_bottom = v_left = v_right = 0

                u[ 0, :] = 2 * u_top - u[1, :]
                u[-1, :] = 2 * u_bottom - u[-2, :]
                u[1:-1,  0] = u_left
                u[1:-1, -1] = u_right

                v[ 0, 1:-1] = v_top
                v[-1, 1:-1] = v_bottom
                v[:,  0] = 2 * v_left - v[:, 1]
                v[:, -1] = 2 * v_right - v[:, -2]
            case _:
                raise NotImplementedError

        # # Dirichlet boundary conditions for velocities
        # u[ 0, :] = 2 * u_top - u[1, :]
        # u[-1, :] = 2 * u_bottom - u[-2, :]
        # u[1:-1,  0] = u_left
        # # Dirichlet BC
        # # u[1:-1, -1] = u_right
        # # Neumann BC (du/dx = du_right)
        # # (u[i    , j + 1] - u[i    , j - 1]) / (2 * self.dx) = du_right
        # u[1:-1, -1] = 2 * self.dx * du_right + u[1:-1, -3]

        # v[ 0, 1:-1] = v_top
        # v[-1, 1:-1] = v_bottom
        # v[:,  0] = 2 * v_left - v[:, 1]
        # # # Dirichlet BC
        # # v[:, -1] = 2 * v_right - v[:, -2]

        # # Neumann BC (dv/dx = dv_right)
        # v[:, -1] = self.dx * dv_right + v[:, -2]

        assert np.shares_memory(u, self.U)
        assert np.shares_memory(v, self.V)
        assert np.shares_memory(p, self.P)
        assert np.shares_memory(ut, self.Ut)
        assert np.shares_memory(vt, self.Vt)
        assert np.shares_memory(pt, self.Pt)
        return u, v, p, ut, vt, pt, self.U, self.V, self.P, self.Ut, self.Vt, self.Pt

    def fun(self, t, y, yp):
        # set boundary conditions
        u, v, p, ut, vt, pt, U, V, P, Ut, Vt, Pt = self.create_redundant_coordinates(y, yp)

        # common quantities
        u_x = (self._Du_x @ U).reshape(self.shape_u)
        u_y = (self._Du_y @ U).reshape(self.shape_u)

        v_x = (self._Dv_x @ V).reshape(self.shape_v)
        v_y = (self._Dv_y @ V).reshape(self.shape_v)

        pt_x = (self._Dp_x @ Pt).reshape(self.shape_p)
        pt_y = (self._Dp_y @ Pt).reshape(self.shape_p)

        # ########################################
        # # u-velocity residual
        # # ∂u/∂t + (u ⋅ ∇) u + 1/ρ ∇p - ν ∇²u = 0
        # ########################################
        # Fu = np.zeros(self.shape_u)

        # # ∂u/∂t
        # Fu[self.inner_DOFs_u] += ut[self.inner_DOFs_u]

        # # (u ⋅ ∇) u
        # Fu[self.inner_DOFs_u] += u[self.inner_DOFs_u] * u_x[self.inner_DOFs_u]
        # # TODO: Why we have to slice v with inner_DOFs_u here?
        # Fu[self.inner_DOFs_u] += v[self.inner_DOFs_u] * u_y[self.inner_DOFs_u]

        # # 1/ρ ∇p
        # Fu[self.inner_DOFs_u] += pt_x[:, :-1] / self.rho

        # # -ν ∇²u
        # Fu[self.inner_DOFs_u] -= self.mu * ((self._DDu_x + self._DDu_y) @ U).reshape(self.shape_u)[self.inner_DOFs_u]

        # ########################################
        # # v-velocity residual
        # # ∂u/∂t + (u ⋅ ∇) u + 1/ρ ∇p - ν ∇²u = 0
        # ########################################
        # Fv = np.zeros(self.shape_v)

        # # ∂u/∂t
        # Fv[self.inner_DOFs_v] += vt[self.inner_DOFs_v]

        # # (u ⋅ ∇) u
        # Fv[self.inner_DOFs_v] += u[self.inner_DOFs_v] * v_x[self.inner_DOFs_v]
        # # TODO: Why we have to slice v with inner_DOFs_u here?
        # Fv[self.inner_DOFs_v] += v[self.inner_DOFs_v] * v_y[self.inner_DOFs_v]

        # # 1/ρ ∇p
        # Fv[self.inner_DOFs_v] += pt_y[:-1, :] / self.rho

        # # -ν ∇²u
        # Fv[self.inner_DOFs_v] -= self.mu * ((self._DDv_x + self._DDv_y) @ V).reshape(self.shape_v)[self.inner_DOFs_v]

        ###################
        # incompressibility
        # ∇ ⋅ u = 0
        ###################
        # Fp = np.zeros(self.shape_p)
        # Fp[self.inner_DOFs_p] = u_x[1:-1, :-1] + v_y[:-1, 1:-1]
        
        # # Fu = ut.copy()
        # # Fv = vt.copy()
        # Fp = pt.copy()

        # Re = 1e-2
        Re = 1e1

        # self.shape_u = (Ny + 2, Nx + 1)
        # self.shape_v = (Ny + 1, Nx + 2)
        # self.shape_p = (Ny, Nx)
        # self.shape_u_interior = (Ny    , Nx - 1)
        # self.shape_v_interior = (Ny - 1, Nx    )
        # self.shape_p_interior = (Ny, Nx)

        Fu = np.zeros(self.shape_u)
        for i in range(1, Ny + 1):
            for j in range(1, Nx):
                # Fu[i, j] = (
                #     ut[i, j]
                #     + u[i, j] * (u[i + 1, j] - u[i - 1, j]) / (2 * self.dx)
                #     + v[i, j] * (u[i, j + 1] - u[i, j - 1]) / (2 * self.dy)
                #     # + (1 / rho) * (p[i, j - 1] - p[i - 1, j - 1]) / dx # note index shift in p!
                #     # - nu * (
                #     #     (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2
                #     #     + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
                #     # )
                #     + (pt[i, j - 1] - pt[i - 1, j - 1]) / self.dx # note index shift in p!
                #     - (
                #         (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / self.dx**2
                #         + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / self.dy**2
                #     ) / Re
                # )
                Fu[i, j] = (
                    ut[i, j]
                    + u[i, j] * (u[i    , j + 1] - u[i    , j - 1]) / (2 * self.dx)
                    + v[i, j] * (u[i + 1, j    ] - u[i - 1, j    ]) / (2 * self.dy)
                    + (pt[i - 1, j] - pt[i - 1, j - 1]) / self.dx # note index shift in p!
                    - (
                        (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / self.dx**2
                        + (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / self.dy**2
                    ) / Re
                )
    
        # all interior points of the v-velocity
        Fv = np.zeros(self.shape_v)
        for i in range(1, Ny):
            for j in range(1, Nx + 1):
                # Fv[i, j] = (
                #     vt[i, j]
                #     + u[i, j] * (v[i + 1, j] - v[i - 1, j]) / (2 * self.dx)
                #     + v[i, j] * (v[i, j + 1] - v[i, j - 1]) / (2 * self.dy)
                #     # + (1 / rho) * (p[i - 1, j] - p[i - 1, j - 1]) / dy # note index shift in p!
                #     # - nu * (
                #     #     (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2
                #     #     + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2
                #     # )
                #     + (pt[i - 1, j] - pt[i - 1, j - 1]) / self.dy # note index shift in p!
                #     - (
                #         (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / self.dx**2
                #         + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / self.dy**2
                #     ) / Re
                # )
                Fv[i, j] = (
                    vt[i, j]
                    + u[i, j] * (v[i    , j + 1] - v[i    , j - 1]) / (2 * self.dx)
                    + v[i, j] * (v[i + 1, j    ] - v[i - 1, j    ]) / (2 * self.dy)
                    + (pt[i, j - 1] - pt[i - 1, j - 1]) / self.dy # note index shift in p!
                    - (
                        (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / self.dx**2
                        + (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / self.dy**2
                    ) / Re
                )

        # TODO: This leads to the rank deficiency!
        # continuity equation
        Fp = np.zeros(self.shape_p)
        for i in range(Ny):
            for j in range(Nx):
                u_x = (u[i + 1, j + 1] - u[i + 1, j]) / self.dx
                v_y = (v[i + 1, j + 1] - v[i, j + 1]) / self.dy
                Fp[i, j] = u_x + v_y
                # Fp[i, j] = p[i, j]
        
        return np.concatenate((
            Fu[self.inner_DOFs_u].reshape(-1),
            Fv[self.inner_DOFs_v].reshape(-1),
            Fp[self.inner_DOFs_p].reshape(-1),
        ))

    def jac(self, t, y, yp, f):
        n = len(y)
        z = np.concatenate((y, yp))

        def fun_composite(z):
            y, yp = z[:n], z[n:]
            return self.fun(t, y, yp)
        
        J = approx_derivative(fun_composite, z, method="2-point", f0=f)
        J = J.reshape((n, 2 * n))
        Jy, Jyp = J[:, :n], J[:, n:]
        return Jy, Jyp

    def animate(self, t, y, yp, interval=50):
        fig, ax = plt.subplots()
        
        def update(num):
            ax.clear()
            ax.set_xlim(-0.25 * self.Lx, 1.25 * self.Lx)
            ax.set_ylim(-0.25 * self.Ly, 1.25 * self.Ly)
            ax.set_aspect("equal")

            # compute redundant coordinates together with boundary conditions
            u, v, p, ut, vt, pt, U, V, P, Ut, Vt, Pt = self.create_redundant_coordinates(y[:, num], yp[:, num])

            # 1. interpolate velocity at cell center/cell corner
            # uce = (u[1:-1, :-1,] + u[1:-1, 1:]) / 2
            # vce = (v[:-1, 1:-1] + v[1:, 1:-1]) / 2
            uco = (u[:-1, :] + u[1:, :]) / 2
            vco = (v[:, :-1] + v[:, 1:]) / 2

            # uco = uco.T
            # vco = vco.T
        
            # cmap = "viridis"
            cmap = "Blues"
            cmap = "jet"
            contour = ax.contourf(self.Column_range, self.Row_range, np.sqrt(uco**2 + vco**2), cmap=cmap, levels=100)
            contour = None

            # streamplot = ax.streamplot(self.Xco.T, self.Yco.T, uco.T, vco.T)
            streamplot = None

            # quiver = ax.quiver(self.Xco, self.Yco, uco, vco, angles="uv", alpha=0.4, scale=5, scale_units="xy")
            quiver = ax.quiver(self.Column_range, self.Row_range, uco, vco, alpha=0.75)
            # quiver = None

            return contour, quiver, streamplot
        
        # contour, quiver, streamplot = update(0)
        # cbar = fig.colorbar(contour)

        anim = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    match SCENARIO:
        case "channel":
            Nx = 16 * RESOLUTION
            Ny = 4 * RESOLUTION
            Lx = 4
            Ly = 1
        case "lid cavity":
            Nx = Ny = 4 * RESOLUTION
            Lx = 2
            Ly = 1
        case _:
            raise NotImplementedError

    fluid = IncompressibleFluid(Nx, Ny, Lx, Ly)

    # dummy initial conditions
    t0 = 0
    y0 = np.zeros(fluid.N_interior)
    yp0 = np.zeros(fluid.N_interior)
    f = fluid.fun(t0, y0, yp0)
    # print(f"f: {f}")

    # # solve for consistent initial conditions
    # y0, yp0, fnorm = consistent_initial_conditions(fluid.fun, fluid.jac, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")

    # time span
    t0 = 0
    t1 = 3
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e2))

    # solver options
    # method = "BDF"
    method = "Radau"
    atol = rtol = 1e-3
    first_step = 1e-5
    # first_step = None

    start = time.time()
    sol = solve_dae(fluid.fun, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, first_step=first_step, t_eval=t_eval)
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

    fluid.animate(t, y, yp)
