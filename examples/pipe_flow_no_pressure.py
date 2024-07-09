import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags, eye, kron
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


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
    def __init__(self, Nx, Ny, Lx, Ly, rho=1, nu=1e-1):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Coordinate of each grid (cell center)
        self.xce = (np.arange(Nx) + 0.5) * self.dx
        self.yce = (np.arange(Ny) + 0.5) * self.dy
        self.Xce, self.Yce = np.meshgrid(self.xce, self.yce)

        # Coordinate of each grid (cell corner)
        self.xco = (np.arange(Nx + 1)) * self.dx
        self.yco = (np.arange(Ny + 1)) * self.dy
        self.Xco, self.Yco = np.meshgrid(self.xco, self.yco)

        # fig, ax = plt.subplots()
        # ax.set_xlim(-0.25 * self.Lx, 1.25 * self.Lx)
        # ax.set_ylim(-0.25 * self.Ly, 1.25 * self.Ly)
        # ax.set_aspect("equal")
        # ax.plot(self.Xce, self.Yce, "ok")
        # plt.show()
        # exit()

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
        self.inner_DOFs_u = np.s_[1:Ny + 1, 1:Nx    ]
        self.inner_DOFs_v = np.s_[1:Ny    , 1:Nx + 1]
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
            D_central(Nx + 1, self.dx), eye(Ny + 2), format="csr"
        )
        self._Du_y = kron(
            eye(Nx + 1), D_central(Ny + 2, self.dy), format="csr"
        )

        self._Dv_x = kron(
            D_central(Nx + 2, self.dx), eye(Ny + 1), format="csr"
        )
        self._Dv_y = kron(
            eye(Nx + 2), D_central(Ny + 1, self.dy), format="csr"
        )

        # TODO: I think we have to change the ordering of these calls below
        self._Dp_x = kron(
            D_forward(Nx, self.dx), eye(Ny), format="csr"
        )
        self._Dp_y = kron(
            eye(Nx), D_forward(Ny, self.dy), format="csr"
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

        # TODO: Add these functions as arguments to fluid
        u_left = v_left = 0
        u_right = v_right = 0
        u_bottom = v_bottom = 0
        u_top = v_top = 0
        # u_top = 1

        u_left = u_right = +1
        u_bottom = u_top = +1
        v_left = v_right = -1
        v_bottom = v_top = -1

        # Dirichlet boundary conditions for velocities
        u[ 0, :] = 2 * u_top - u[1, :]
        u[-1, :] = 2 * u_bottom - u[-2, :]
        u[1:-1,  0] = u_left
        u[1:-1, -1] = u_right

        v[ 0, 1:-1] = v_top
        v[-1, 1:-1] = v_bottom
        v[:,  0] = 2 * v_left - v[:, 1]
        v[:, -1] = 2 * v_right - v[:, -2]

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

        ########################################
        # u-velocity residual
        # ∂u/∂t + (u ⋅ ∇) u + 1/ρ ∇p - ν ∇²u = 0
        ########################################
        Fu = np.zeros(self.shape_u)

        # ∂u/∂t
        Fu[self.inner_DOFs_u] += ut[self.inner_DOFs_u]

        # # (u ⋅ ∇) u
        # Fu[self.inner_DOFs_u] += u[self.inner_DOFs_u] * u_x[self.inner_DOFs_u]
        # # TODO: Why we have to slice v with inner_DOFs_u here?
        # Fu[self.inner_DOFs_u] += v[self.inner_DOFs_u] * u_y[self.inner_DOFs_u]

        # # 1/ρ ∇p
        # Fu[self.inner_DOFs_u] += pt_x[:, :-1] / self.rho

        # -ν ∇²u
        Fu[self.inner_DOFs_u] -= self.mu * ((self._DDu_x + self._DDu_y) @ U).reshape(self.shape_u)[self.inner_DOFs_u]

        ########################################
        # v-velocity residual
        # ∂u/∂t + (u ⋅ ∇) u + 1/ρ ∇p - ν ∇²u = 0
        ########################################
        Fv = np.zeros(self.shape_v)

        # ∂u/∂t
        Fv[self.inner_DOFs_v] += vt[self.inner_DOFs_v]

        # # (u ⋅ ∇) u
        # Fv[self.inner_DOFs_v] += u[self.inner_DOFs_v] * v_x[self.inner_DOFs_v]
        # # TODO: Why we have to slice v with inner_DOFs_u here?
        # Fv[self.inner_DOFs_v] += v[self.inner_DOFs_v] * v_y[self.inner_DOFs_v]

        # # 1/ρ ∇p
        # Fv[self.inner_DOFs_v] += pt_y[:-1, :] / self.rho

        # -ν ∇²u
        Fv[self.inner_DOFs_v] -= self.mu * ((self._DDv_x + self._DDv_y) @ V).reshape(self.shape_v)[self.inner_DOFs_v]

        ###################
        # incompressibility
        # ∇ ⋅ u = 0
        ###################
        Fp = np.zeros(self.shape_p)
        Fp[self.inner_DOFs_p] = u_x[1:-1, :-1] + v_y[:-1, 1:-1]
        
        # Fu = ut.copy()
        # Fv = vt.copy()
        Fp = pt.copy()
        
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
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        def update(num):
            ax.clear()
            # ax.set_xlim(-0.25 * self.Lx, 1.25 * self.Lx)
            # ax.set_ylim(-0.25 * self.Ly, 1.25 * self.Ly)
            # ax.set_aspect("equal")
            # ax.plot(self.Xce, self.Yce, "ok")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u(x, y)")
            ax.set_zlim(-2, 2)

            # compute redundant coordinates together with boundary conditions
            u, v, p, ut, vt, pt, U, V, P, Ut, Vt, Pt = self.create_redundant_coordinates(y[:, num], yp[:, num])

            # 1. interpolate velocity at cell center/cell corner
            # uce = (u(1:end-1,2:end-1)+u(2:end,2:end-1))/2;
            uce = (
                # # u(1:end-1,2:end-1)+u(2:end,2:end-1)
                # u[:-1, 1:-1] + u[1:, 1:-1]
                u[1:-1, :-1,] + u[1:-1, 1:]
            ) / 2
            vce = (
                # # v(2:end-1,1:end-1)+v(2:end-1,2:end)
                # v[1:-1, :-1] + v[1:-1, 1:]
                v[:-1, 1:-1] + v[1:, 1:-1]
            ) / 2
            uco = (u[:-1, :] + u[1:, :]) / 2
            vco = (v[:, :-1] + v[:, 1:]) / 2

            # return None,
        
            # contour = ax.contourf(self.Xce, self.Yce, np.sqrt(uce**2 + vce**2), alpha=0.5)
            # # contour = ax.contourf(X, Y, np.sqrt(u[1:, :]**2 + v[:, 1:]**2), alpha=0.5)

            # return contour,
    
            # surf = ax.plot_surface(self.Xce, self.Yce, uce)
            surf = ax.plot_surface(self.Xco, self.Yco, uco)
            surf = ax.plot_surface(self.Xco, self.Yco, vco, alpha=0.5)

            return surf,

            # # # with np.errstate(divide='ignore'):
            # # quiver = ax.quiver(x, y, u[:, :, num], v[:, :, num])
            # # return quiver, contour

            # # contour = ax.contourf(x, y, p[:, :, num], alpha=0.5)

            # streamplot = ax.streamplot(x, y, u[:, :, num], v[:, :, num])
            # return contour, streamplot

        # anim = animation.FuncAnimation(fig, update, frames=u.shape[-1], interval=interval, blit=True)
        anim = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    # Nx = 3
    # Ny = 3
    Nx = 10
    Ny = 10
    Lx = Ly = 1.0
    fluid = IncompressibleFluid(Nx, Ny, Lx, Ly)

    # dummy initial conditions
    t0 = 0
    y0 = np.zeros(fluid.N_interior)
    yp0 = np.zeros(fluid.N_interior)
    # f = fluid.fun(t0, y0, yp0)
    # print(f"f: {f}")

    # # solve for consistent initial conditions
    # y0, yp0, fnorm = consistent_initial_conditions(fluid.fun, fluid.jac, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")

    # time span
    t0 = 0
    t1 = 20
    t_span = (t0, t1)

    # solver options
    method = "BDF"
    # method = "Radau"
    atol = rtol = 1e-4
    first_step = None

    start = time.time()
    sol = solve_dae(fluid.fun, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, first_step=first_step, dense_output=True)
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
