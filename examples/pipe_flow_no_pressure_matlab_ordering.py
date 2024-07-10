import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags, eye, kron
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions

SCENARIO = "channel"
# SCENARIO = "lid cavity"

RESOLUTION = 1

# Reynolds number
RE = 300
# RE = 1e2
# RE = 1e-1


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
    """The implementation follows mathworks. A similar discretization is introduced in Rook1996 within compressible Navier-Stokes problems.

    The staggered grid with ghost cells

        →       →       →       →       →    
        |       |       |       |       |   
    ↑ - ❖---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |    
        →   •   →   •   →   •   →   •   →    
        |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |    
        →   •   →   •   →   •   →   •   →    
        |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |    
        →   •   →   •   →   •   →   •   →    
        |       |       |       |       |    
    ↑ - 0---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |    
        →       →       →       →       →    
    

    References:
    -----------
    mathworks: https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver/blob/master/docs_part1/vanilaCavityFlow_EN.md \\
    Rook1996: https://fse.studenttheses.ub.rug.nl/8741/1/Math_Drs_1996_RRook.CV.pdf
    """
    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx = Nx # number of cells in x-direction
        self.Ny = Ny # number of cells in y-direction
        self.Lx = Lx
        self.Ly = Ly

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Coordinate of each grid (cell corner)
        self.x_range = (np.arange(Nx + 1)) * self.dx
        self.y_range = (np.arange(Ny + 1)) * self.dy
        self.Y_range, self.X_range = np.meshgrid(self.y_range, self.x_range, indexing="xy")

        # shapes
        self.shape_u = (Nx + 1, Ny + 2)
        self.shape_v = (Nx + 2, Ny + 1)
        self.shape_p = (Nx, Ny)
        self.shape_u_interior = (Nx - 1, Ny    )
        self.shape_v_interior = (Nx    , Ny - 1)
        self.shape_p_interior = (Nx, Ny)
        
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
        self.inner_DOFs_u = np.s_[1:-1, 1:-1]
        self.inner_DOFs_v = np.s_[1:-1, 1:-1]
        self.inner_DOFs_p = np.s_[:, :] # TODO: Eliminate this since it is useless

        self.u = np.zeros(self.shape_u)
        self.v = np.zeros(self.shape_v)
        self.p = np.zeros(self.shape_p)
        self.U = np.zeros(self.Nu)
        self.V = np.zeros(self.Nv)
        self.P = np.zeros(self.Np)
        self.Ut = np.zeros(self.Nu)
        self.Vt = np.zeros(self.Nv)
        self.Pt = np.zeros(self.Np)

        # # operators
        # # - first derivatives
        # self._Du_x = kron(
        #     eye(Ny + 2), D_central(Nx + 1, self.dx), format="csr"
        # )
        # self._Du_y = kron(
        #     D_central(Ny + 2, self.dy), eye(Nx + 1), format="csr"
        # )

        # self._Dv_x = kron(
        #     eye(Ny + 1), D_central(Nx + 2, self.dx), format="csr"
        # )
        # self._Dv_y = kron(
        #     D_central(Ny + 1, self.dy), eye(Nx + 2), format="csr"
        # )

        # self._Dp_x = kron(
        #     eye(Ny), D_forward(Nx, self.dx), format="csr"
        # )
        # self._Dp_y = kron(
        #     D_forward(Ny, self.dy), eye(Nx), format="csr"
        # )

        # # - second derivatives
        # self._DDu_x = kron(
        #     eye(Ny + 2), DD_central(Nx + 1, self.dx), format="csr"
        # )
        # self._DDu_y = kron(
        #     DD_central(Ny + 2, self.dy), eye(Nx + 1), format="csr"
        # )

        # self._DDv_x = kron(
        #     eye(Ny + 1), DD_central(Nx + 2, self.dx), format="csr"
        # )
        # self._DDv_y = kron(
        #     DD_central(Ny + 1, self.dy), eye(Nx + 2), format="csr"
        # )

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

        # TODO: Adapt these boundary conditions
        match SCENARIO:
            case "channel":
                # Dirichlet boundary conditions
                u_left = 1
                u_top = u_bottom = 0
                v_top = v_bottom = v_left = 0

                u[:, -1] = 2 * u_top - u[:, -2]
                u[:,  0] = 2 * u_bottom - u[:, 1]
                u[0, 1:-1] = u_left

                v[1:-1, -1] = v_top
                v[1:-1, 0] = v_bottom
                v[0, :] = 2 * v_left - v[1, :]

                # Neumann boundary conditions (du/dx = du_right, dv/dx = dv_right)
                du_right = dv_right = 0
                u[-1, 1:-1] = 2 * self.dx * du_right + u[-3, 1:-1]
                # v[-1, :] = self.dx * dv_right + v[-2, :]
                # TODO: Check this derivative!
                v[-1, :] = 2 * self.dx * dv_right + v[-3, :]
            case "lid cavity":
                # Dirichlet boundary conditions
                u_top = 1
                u_bottom = u_left = u_right = 0
                v_top = v_bottom = v_left = v_right = 0

                u[0, :] = u_left
                u[-1, :] = u_right
                u[:, -1] = 2 * u_top - u[:, -2]
                u[:,  0] = 2 * u_bottom - u[:, 1]
                # u[0, 1:-1] = u_left
                # u[-1, 1:-1] = u_right

                v[:, -1] = v_top
                v[:, 0] = v_bottom
                v[0, :] = 2 * v_left - v[1, :]
                v[-1, :] = 2 * v_right - v[-2, :]
                # v[1:-1, -1] = v_top
                # v[1:-1, 0] = v_bottom
            case _:
                raise NotImplementedError

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

        # # common quantities
        # u_x = (self._Du_x @ U).reshape(self.shape_u)
        # u_y = (self._Du_y @ U).reshape(self.shape_u)

        # v_x = (self._Dv_x @ V).reshape(self.shape_v)
        # v_y = (self._Dv_y @ V).reshape(self.shape_v)

        # pt_x = (self._Dp_x @ Pt).reshape(self.shape_p)
        # pt_y = (self._Dp_y @ Pt).reshape(self.shape_p)

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

        # self.shape_u = (Ny + 2, Nx + 1)
        # self.shape_v = (Ny + 1, Nx + 2)
        # self.shape_p = (Ny, Nx)
        # self.shape_u_interior = (Ny    , Nx - 1)
        # self.shape_v_interior = (Ny - 1, Nx    )
        # self.shape_p_interior = (Ny, Nx)

        Fu = np.zeros(self.shape_u)
        for i in range(1, Nx):
            for j in range(1, Ny + 1):
                # Fu[i, j] = (
                #     ut[i, j]
                #     + u[i, j] * (u[i + 1, j    ] - u[i - 1, j    ]) / (2 * self.dx)
                #     + v[i, j] * (u[i    , j + 1] - u[i    , j - 1]) / (2 * self.dy)
                #     + (pt[i, j - 1] - pt[i - 1, j - 1]) / self.dx # note index shift in p!
                #     - (
                #         (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / self.dx**2
                #         + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / self.dy**2
                #     ) / RE
                # )

                Fu[i, j] += ut[i, j]
                # central differences
                Fu[i, j] += u[i, j] * (u[i + 1, j    ] - u[i - 1, j    ]) / (2 * self.dx)
                Fu[i, j] += v[i, j] * (u[i    , j + 1] - u[i    , j - 1]) / (2 * self.dy)
                # # central differences with interpolated v-velocity
                # vi2j2 = (v[i, j - 1] + v[i + 1, j - 1] + v[i + 1, j] + v[i, j]) / 4
                # Fu[i, j] += u[i, j] * (u[i + 1, j    ] - u[i - 1, j    ]) / (2 * self.dx)
                # Fu[i, j] += vi2j2 * (u[i    , j + 1] - u[i    , j - 1]) / (2 * self.dy)
                # # first-order upwind
                # if u[i + 1, j] > 0:
                #     Fu[i, j] += u[i, j] * (u[i + 1, j] - u[i    , j]) / self.dx
                # else:
                #     Fu[i, j] += u[i, j] * (u[i    , j] - u[i - 1, j]) / self.dx
                # # first-order upwind
                # if u[i, j + 1] > 0:
                #     Fu[i, j] += v[i, j] * (u[i, j + 1] - u[i, j    ]) / self.dy
                # else:
                #     Fu[i, j] += v[i, j] * (u[i, j    ] - u[i, j - 1]) / self.dy
                # # divergence form
                # ucell_east = (u[i    , j    ] + u[i + 1, j    ]) / 2
                # ucell_west = (u[i - 1, j    ] + u[i    , j    ]) / 2
                # unode_top  = (u[i    , j + 1] + u[i    , j    ]) / 2
                # unode_bot  = (u[i    , j    ] + u[i    , j - 1]) / 2
                # vnode_top  = (v[i    , j    ] + v[i + 1, j    ]) / 2
                # vnode_bot  = (v[i    , j - 1] + v[i + 1, j - 1]) / 2
                # Fu[i, j] += (
                #     (ucell_east**2 - ucell_west**2) / self.dx
                #     + (unode_top * vnode_top - unode_bot * vnode_bot) / self.dy
                # )

                Fu[i, j] += (pt[i, j - 1] - pt[i - 1, j - 1]) / self.dx # note index shift in p!
                Fu[i, j] += -(
                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / self.dx**2
                    + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / self.dy**2
                ) / RE
    
        # all interior points of the v-velocity
        Fv = np.zeros(self.shape_v)
        for i in range(1, Nx + 1):
            for j in range(1, Ny):
                # Fv[i, j] = (
                #     vt[i, j]
                #     + u[i, j] * (v[i + 1, j    ] - v[i - 1, j    ]) / (2 * self.dx)
                #     + v[i, j] * (v[i    , j + 1] - v[i    , j - 1]) / (2 * self.dy)
                #     + (pt[i - 1, j] - pt[i - 1, j - 1]) / self.dy # note index shift in p!
                #     - (
                #         (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / self.dx**2
                #         + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / self.dy**2
                #     ) / RE
                # )
                Fv[i, j] += vt[i, j]
                # central differences
                Fv[i, j] += u[i, j] * (v[i + 1, j    ] - v[i - 1, j    ]) / (2 * self.dx)
                Fv[i, j] += v[i, j] * (v[i    , j + 1] - v[i    , j - 1]) / (2 * self.dy)
                # # central differences with interpolated u-velocity
                # ui2j2 = (u[i - 1, j] + u[i, j] + u[i, j + 1] + u[i - 1, j + 1]) / 4
                # Fv[i, j] += ui2j2 * (v[i + 1, j    ] - v[i - 1, j    ]) / (2 * self.dx)
                # Fv[i, j] += v[i, j] * (v[i    , j + 1] - v[i    , j - 1]) / (2 * self.dy)
                # # first-order upwind
                # if v[i + 1, j] > 0:
                #     Fv[i, j] += u[i, j] * (v[i + 1, j] - v[i    , j]) / self.dx
                # else:
                #     Fv[i, j] += u[i, j] * (v[i    , j] - v[i - 1, j]) / self.dx
                # # first-order upwind
                # if v[i, j + 1] > 0:
                #     Fv[i, j] += v[i, j] * (v[i, j + 1] - v[i, j    ]) / self.dy
                # else:
                #     Fv[i, j] += v[i, j] * (v[i, j    ] - v[i, j - 1]) / self.dy
                # # divergence form
                # vnode_east = (v[i    , j    ] + v[i + 1, j    ]) / 2
                # vnode_west = (v[i - 1, j    ] + v[i    , j    ]) / 2
                # unode_east = (u[i    , j + 1] + u[i    , j    ]) / 2
                # unode_west = (u[i - 1, j + 1] + u[i - 1, j    ]) / 2
                # vcell_top  = (v[i    , j + 1] + v[i    , j    ]) / 2
                # vcell_bot  = (v[i    , j    ] + v[i    , j - 1]) / 2
                # Fu[i, j] += (
                #     (unode_east * vnode_east - unode_west * vnode_west) / self.dx
                #     + (vcell_top**2 - vcell_bot**2) / self.dy
                # )

                Fv[i, j] += (pt[i - 1, j] - pt[i - 1, j - 1]) / self.dy # note index shift in p!
                Fv[i, j] += -(
                    (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / self.dx**2
                    + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / self.dy**2
                ) / RE

        # all pressure nodes (cell centers)
        Fp = np.zeros(self.shape_p)
        for i in range(Nx):
            for j in range(Ny):
                u_x = (u[i + 1, j + 1] - u[i, j + 1]) / self.dx
                v_y = (v[i + 1, j + 1] - v[i + 1, j]) / self.dy
                Fp[i, j] = u_x + v_y

        # fix pressure value at a single point since only its derivative is defined
        # Fp[0, 0] = pt[0, 0]
        # Fp[1, 1] = pt[1, 1]
        Fp[-1, -1] = pt[-1, -1]
        
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

            # interpolate velocity at cell corners
            uco = (u[:, :-1] + u[:, 1:]) / 2
            vco = (v[:-1, :] + v[1:, :]) / 2
        
            cmap = "jet"
            contour = ax.contourf(self.X_range, self.Y_range, np.sqrt(uco**2 + vco**2), cmap=cmap, levels=100)
            # contour = None

            streamplot = ax.streamplot(self.x_range, self.y_range, uco.T, vco.T)
            # streamplot = None

            quiver = ax.quiver(self.X_range, self.Y_range, uco, vco, alpha=0.75)
            # quiver = None

            return contour, quiver, streamplot
        
        contour, quiver, streamplot = update(0)
        cbar = fig.colorbar(contour)

        anim = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    match SCENARIO:
        case "channel":
            RESOLUTION = 3
            Nx = int(16 * RESOLUTION)
            Ny = int(4 * RESOLUTION)
            # Nx = int(12 * RESOLUTION)
            # Ny = int(3 * RESOLUTION)
            # Nx = int(8 * RESOLUTION)
            # Ny = int(2 * RESOLUTION)
            Lx = 4
            Ly = 1
        case "lid cavity":
            Nx = Ny = int(8 * RESOLUTION)
            # Nx = Ny = int(3 * RESOLUTION)
            Lx = 1
            Ly = 1
        case _:
            raise NotImplementedError

    fluid = IncompressibleFluid(Nx, Ny, Lx, Ly)

    # dummy initial conditions
    t0 = 0
    y0 = np.zeros(fluid.N_interior)
    yp0 = np.zeros(fluid.N_interior)
    jac_sparsity = None

    # y0 = np.random.rand(fluid.N_interior)
    # yp0 = np.random.rand(fluid.N_interior)
    f0 = fluid.fun(t0, y0, yp0)
    # np.set_printoptions(3, suppress=True)
    # print(f"f0:\n{f0}")
    Jy0, Jyp0 = fluid.jac(t0, y0, yp0, f0)
    jac_sparsity = (Jy0, Jyp0)
    J0 = Jy0 + np.pi * Jyp0
    # print(f"Jy0:\n{Jy0}")
    # print(f"Jyp0:\n{Jyp0}")
    # print(f"J0:\n{J0}")
    rank_Jy0 = np.linalg.matrix_rank(Jy0)
    rank_Jyp0 = np.linalg.matrix_rank(Jyp0)
    rank_J0 = np.linalg.matrix_rank(J0)
    print(f"Jy0.shape: {Jy0.shape}")
    print(f"rank_Jy0: {rank_Jy0}")
    print(f"rank_Jyp0: {rank_Jyp0}")
    print(f"rank_J0: {rank_J0}")
    # exit()

    # # solve for consistent initial conditions
    # y0, yp0, fnorm = consistent_initial_conditions(fluid.fun, fluid.jac, t0, y0, yp0)
    # # print(f"y0: {y0}")
    # # print(f"yp0: {yp0}")
    # # print(f"fnorm: {fnorm}")
    # # exit()

    # time span
    t0 = 0
    # t1 = 1
    t1 = 5
    t_span = (t0, t1)
    # num = int(1e2)
    num = 50
    t_eval = np.linspace(t0, t1, num=num)

    # solver options
    # method = "BDF"
    # atol = rtol = 1e-3
    method = "Radau"
    atol = rtol = 1e-4
    first_step = 1e-5
    # first_step = None

    start = time.time()
    sol = solve_dae(fluid.fun, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, first_step=first_step, t_eval=t_eval, jac_sparsity=jac_sparsity)
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
