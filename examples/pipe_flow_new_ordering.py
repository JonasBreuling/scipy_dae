"""
Solves the equation of fluid motion in a pipe with inlet and outlet.
This will simulate the inflow behavior into the pipe and the
boundary layer developing over time and space. The system of equations
is solved using a Staggered Grid, Finite Differences (almost Finite
Volume), explicit Euler time-stepping and a P2 pressure correction
scheme (very similar to the SIMPLE algorithm) based on a segregated
approach.


Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0


u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here =0)
ν:  Kinematic Viscosity
ρ:  Density
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

--------

Scenario


                        wall: u=0, v=0
        +-----------------------------------------------+
        |  -->      -->       -->        -->      -->   |
inflow  |                                               | outflow
u = 1   |  -->      -->       -->        -->      -->   | ∂u/∂x = 0
v = 0   |                                               | ∂v/∂x = 0
        |  -->      -->       -->        -->      -->   |
        +-----------------------------------------------+
                        wall: u=0, v=0                       

-> A rectangular domain (think of a slice from a pipe with
   circular cross-section alongside the longitudinal axis)
-> Top and bottom edge represent wall boundary conditions
-> A uniform inflow profile (over the pipe's cross section)
   is prescribed at the left edge - a 1.0 horizontal
   velocity and a 0.0 vertical velocity
-> The right edge represents an outflow, defined by the
   assumption that the flow does not change anymore over
   the horizontal axis, hence the normal derivatives of
   both velocity components are zero
-> Initially, the u velocity is uniform 1.0 over the domain,
   the v velocity is uniform 0.0

--------

Expected Outcome

        +-----------------------------------------------+
        |   -->      -->      ->       >        >       |
        |   -->      -->      --->     --->     --->    |
        |   -->      --->     --->     ---->    ---->   |
        |   -->      -->      --->     --->     --->    |
        |   -->      -->      ->       >        >       |
        +-----------------------------------------------+

The flow is developing the characteristic parabolic Hagen-Poiseulle
profile at some point x in the domain and will keep this profile
until the outflow on the right

-------

A (classical) co-located grid

        + - - - + - - - + - - - + - - - + - - - + - - - +
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        + - - - + - - - + - - - + - - - + - - - + - - - +
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        + - - - + - - - + - - - + - - - + - - - + - - - +
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        |       |       |       |       |       |       |
        0 - - - + - - - + - - - + - - - + - - - + - - - +

All variables (u-velocity, v-velocity, pressure) are saved
at the mesh vertices ("+").

-> Using central differences for pressure gradient and divergence
results in checkerboard pattern - problematic here.

-------

Remedy: The staggered grid

        + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
        |       |       |       |       |       |       |
        →   •   →   •   →   •   →   •   →   •   →   •   →
        |       |       |       |       |       |       |
        + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
        |       |       |       |       |       |       |
        →   •   →   •   →   •   →   •   →   •   →   •   →
        |       |       |       |       |       |       |
        + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
        |       |       |       |       |       |       |
        →   •   →   •   →   •   →   •   →   •   →   •   →
        |       |       |       |       |       |       |
        0 - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +

The pressure is saved at "•"
The horizontal velocity is saved at "→"
The vertical velocity is saved at "↑"

If the "0" indicates the origin of the domain, then

* u-velocities are staggered in y direction
* v-velocities are staggered in x direction
* pressure is staggered in both x and y direction

If we have N_x vertex nodes in x direction and N_y vertex nodes
in y direction, then:

* u_velocities use N_x by (N_y - 1) nodes
* v_velocities use (N_x - 1) by N_y nodes
* pressure use (N_x - 1) by (N_y - 1) nodes

-----

The staggered grid with ghost cells

        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ↑ - ❖---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |       |       |    
    ↑ - 0---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •


Add one padding layer of pressure nodes around the domain
together with its corresponding staggered velocities.

"❖" denotes grid vertices that are on the boundary. Everything
outside of it, is called a ghost node. We need it to enforce the
boundary condition.

Again, if we have N_x vertex nodes in x direction and N_y vertex nodes
in y direction, then (including the ghost nodes):

* u_velocities use N_x by (N_y + 1) nodes
* v_velocities use (N_x + 1) by N_y nodes
* pressure use (N_x + 1) by (N_y + 1) nodes

IMPORTANT: When taking derivatives make sure in which staggered
grid you are thinking.

-----

Solution Strategy:

Usage of a P2 pressure correction scheme (very similar to the SIMPLE
algorithm)

0. Initialization

    0.1 Initialize the u velocity uniformly with ones (+ wall boundaries)

    0.2 Initialize the v velocity uniformly with zeros

    0.3 Initialize the p (=pressure) uniformly with zeros

1. Update the u velocities (+ Boundary Conditions)

    u ← u + dt ⋅ (− ∂p/∂x + ν ∇²u − ∂u²/∂x − v ∂u/∂y)

2. Update the v velocities (+ Boundary Conditions)

    v ← v + dt ⋅ (− ∂p/∂y + ν ∇²v − u ∂v/∂x − ∂v²/∂y)

3. Compute the divergence of the tentative velocity components

    d = ∂u/∂x + ∂v/∂y

4. Solve a Poisson problem for the pressure correction q
   (this problem has homogeneous Neumann BC everywhere except
   for the right edge of the domain (the outlet))

    solve   ∇²q = d / dt   for  q

5. Update the pressure

    p ← p + q

6. Update the velocities to be incompressible

    u ← u − dt ⋅ ∂q/∂x

    v ← v − dt ⋅ ∂q/∂y

7. Repeat time loop until steady-state is reached


For visualizations the velocities have to mapped to the
original vertex-centered grid.

The flow might require a correction at the outlet to ensure
continuity over the entire domain.

The density is assumed to be 1.0

-----

Notes on stability:

1. We are using an explicit diffusion treatment (FTCS) which
   has the stability condition:

   (ν dt) / (dx²) ≤ 1/2

2. We are using a central difference approximation for the
   convection term which is only stable if the diffusive
   transport is dominant (i.e., do not select the kinematic
   viscosity too low).

3. The Pressure Poisson (correction) problem is solved using
   Jacobi smoothing. This is sufficient for this simple
   application, but due to the fixed number of iterations
   does not ensure the residual is sufficiently small. That 
   could introduce local compressibility.
"""

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
    """
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
    """
    def __init__(self, Nx, Ny, Lx, Ly, rho=1, nu=1e-4):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        self.rho = rho
        self.nu = nu
        self.mu = nu / rho

        # shapes
        self.shape_u = (Ny + 1,     Nx)
        self.shape_v = (Ny    , Nx + 1)
        self.shape_p = (Ny + 1, Nx + 1)
        self.shape_u_interior = (Ny - 1,     Nx)
        self.shape_v_interior = (Ny    , Nx - 1)
        self.shape_p_interior = (Ny - 1, Nx - 1)
        
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
        # self.inner_DOFs_u = np.s_[1:Ny, :]
        # self.inner_DOFs_v = np.s_[:, 1:Nx]
        # self.inner_DOFs_p = np.s_[1:Ny, 1:Nx]
        self.inner_DOFs_u = np.s_[   1:Ny, :Nx]
        self.inner_DOFs_v = np.s_[:Ny,    1:Nx]
        self.inner_DOFs_p = np.s_[   1:Ny,    1:Nx]

        # self.boundary_DOFs_top = np.s_[0, :]
        # self.boundary_DOFs_bottom = np.s_[-1, :]
        # self.boundary_DOFs_left = np.s_[:, 0]
        # self.boundary_DOFs_right = np.s_[:, -1]

        # unknown fields
        # self.u = np.zeros((Nx    , Ny + 1))
        # self.v = np.zeros((Nx + 1, Ny    ))
        # self.p = np.zeros((Nx + 1, Ny + 1))
        # self.U = np.zeros(Nx * (Ny + 1))
        # self.V = np.zeros((Nx + 1) * Ny)
        # self.P = np.zeros((Nx + 1) * (Ny + 1))
        # self.Ut = np.zeros(Nx * (Ny + 1))
        # self.Vt = np.zeros((Nx + 1) * Ny)
        # self.Pt = np.zeros((Nx + 1) * (Ny + 1))

        self.u = np.zeros(self.shape_u)
        self.v = np.zeros(self.shape_v)
        self.p = np.zeros(self.shape_p)
        self.U = np.zeros(self.Nu)
        self.V = np.zeros(self.Nv)
        self.P = np.zeros(self.Np)
        self.Ut = np.zeros(self.Nu)
        self.Vt = np.zeros(self.Nv)
        self.Pt = np.zeros(self.Np)

        # # number of interior points
        # # self.Nu = (Nx - 1) * Ny
        # # self.Nv = Nx * (Ny - 1)
        # # self.Np = Nx * Ny
        # self.Nu = (Ny - 1) * Nx
        # self.Nv = Ny * (Nx - 1)
        # self.Np = Ny * Nx
        # self.split = np.cumsum([self.Nu, self.Nv])

        # self.y = np.concatenate

        # operators
        # - first derivatives
        self._Du_x = kron(
            # D_central(Nx, self.dx), eye(Ny + 1), format="csr"
            D_forward(Nx, self.dx), eye(Ny + 1), format="csr"
        )
        self._Du_y = kron(
            # eye(Nx), D_central(Ny + 1, self.dy), format="csr"
            eye(Nx), D_forward(Ny + 1, self.dy), format="csr"
        )

        self._Dv_x = kron(
            # D_central(Nx + 1, self.dx), eye(Ny), format="csr"
            D_forward(Nx + 1, self.dx), eye(Ny), format="csr"
        )
        self._Dv_y = kron(
            # eye(Nx + 1), D_central(Ny, self.dy), format="csr"
            eye(Nx + 1), D_forward(Ny, self.dy), format="csr"
        )

        self._Dp_x = kron(
            D_forward(Nx + 1, self.dx), eye(Ny + 1), format="csr"
        )
        self._Dp_y = kron(
            eye(Nx + 1), D_forward(Ny + 1, self.dy), format="csr"
        )

        # - second derivatives
        self._DDu_x = kron(
            DD_central(Nx, self.dx), eye(Ny + 1), format="csr"
        )
        self._DDu_y = kron(
            eye(Nx), DD_central(Ny + 1, self.dy), format="csr"
        )

        self._DDv_x = kron(
            DD_central(Nx + 1, self.dx), eye(Ny), format="csr"
        )
        self._DDv_y = kron(
            eye(Nx + 1), DD_central(Ny, self.dy), format="csr"
        )

        self._DDp_x = kron(
            DD_central(Nx + 1, self.dx), eye(Ny + 1), format="csr"
        )
        self._DDp_y = kron(
            eye(Nx + 1), DD_central(Ny + 1, self.dy), format="csr"
        )

    def create_redundant_coordinates(self, y, yp):
        # extract interior points
        u_interior, v_interior, p_interior = np.array_split(y, self.split)
        ut_interior, vt_interior, pt_interior = np.array_split(yp, self.split)

        # reshape interior points
        # u_interior = u_interior.reshape((Nx - 1, Ny))
        # v_interior = v_interior.reshape((Nx, Ny - 1))
        # p_interior = p_interior.reshape((Nx, Ny))
        # ut_interior = ut_interior.reshape((Nx - 1, Ny))
        # vt_interior = vt_interior.reshape((Nx, Ny - 1))
        # pt_interior = pt_interior.reshape((Nx, Ny))
        # self.Nu = (Ny - 1) * Nx
        # self.Nv = Ny * (Nx - 1)
        # self.Np = Ny * Nx
        u_interior = u_interior.reshape(self.shape_u_interior)
        v_interior = v_interior.reshape(self.shape_v_interior)
        p_interior = p_interior.reshape(self.shape_p_interior)
        ut_interior = ut_interior.reshape(self.shape_u_interior)
        vt_interior = vt_interior.reshape(self.shape_v_interior)
        pt_interior = pt_interior.reshape(self.shape_p_interior)

        # build redundant coordinates
        # u = np.zeros((Nx + 1, Ny + 2))
        # v = np.zeros((Nx + 2, Ny + 1))
        # p = np.zeros((Nx, Ny))
        # ut = np.zeros((Nx + 1, Ny + 2))
        # vt = np.zeros((Nx + 2, Ny + 1))
        # pt = np.zeros((Nx, Ny))
        # self.u = np.zeros((Ny + 1,     Nx))
        # self.v = np.zeros((Ny    , Nx + 1))
        # self.p = np.zeros((Ny + 1, Nx + 1))
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
        # u[1:-1,    :] = u_interior
        # v[   :, 1:-1] = v_interior
        # p[1:-1, 1:-1] = p_interior
        # assert np.shares_memory(u[1:-1,    :], u)
        # assert np.shares_memory(v[   :, 1:-1], v)
        # assert np.shares_memory(p[1:-1, 1:-1], p)
        u[self.inner_DOFs_u] = u_interior
        v[self.inner_DOFs_v] = v_interior
        p[self.inner_DOFs_p] = p_interior
        assert np.shares_memory(u[self.inner_DOFs_u], u)
        assert np.shares_memory(v[self.inner_DOFs_v], v)
        assert np.shares_memory(p[self.inner_DOFs_p], p)

        # ut[1:-1,    :] = ut_interior
        # vt[   :, 1:-1] = vt_interior
        # pt[1:-1, 1:-1] = pt_interior
        # assert np.shares_memory(ut[1:-1, 1:-1], ut)
        # assert np.shares_memory(vt[   :, 1:-1], vt)
        # assert np.shares_memory(pt[1:-1, 1:-1], pt)
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
        u_top = 1

        # Dirichlet boundary conditions for velocities
        u[0, :] = 2 * u_top - u[1, :]
        u[-1, :] = 2 * u_bottom - u[-2, :]
        u[:, 0] = 2 * u_left - u[:, 1]
        u[:, -1] = 2 * u_right - u[:, -2]

        v[0, :] = 2 * v_top - v[1, :]
        v[-1, :] = 2 * v_bottom - v[-2, :]
        v[:, 0] = 2 * v_left - v[:, 1]
        v[:, -1] = 2 * v_right - v[:, -2]

        # Neumann boundary conditions for the pressure
        # Note: we use pt here since this reduces the DAE index to 1!
        # TODO: Compare this with the discretization
        dp_top = 0
        dp_bottom = 0
        dp_left = 0
        dp_right = 0
        pt[ 0,  :] = pt[ 1,  :] + self.dx * dp_top
        pt[-1,  :] = pt[-2,  :] - self.dx * dp_bottom
        pt[ :,  0] = pt[ :,  1] - self.dx * dp_left
        pt[ :, -1] = pt[ :, -2] + self.dx * dp_right

        assert np.shares_memory(u, self.U)
        assert np.shares_memory(v, self.V)
        assert np.shares_memory(p, self.P)
        assert np.shares_memory(ut, self.Ut)
        assert np.shares_memory(vt, self.Vt)
        assert np.shares_memory(pt, self.Pt)
        return u, v, p, ut, vt, pt, self.U, self.V, self.P, self.Ut, self.Vt, self.Pt
        # return u.reshape(-1), v.reshape(-1), p.reshape(-1), ut.reshape(-1), vt.reshape(-1), pt.reshape(-1)
        # return self.U, self.V, self.P, self.Ut, self.Vt, self.Pt

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

        # (u ⋅ ∇) u
        Fu[self.inner_DOFs_u] += u[self.inner_DOFs_u] * u_x[self.inner_DOFs_u]
        # TODO: Why we have to slice v with inner_DOFs_u here?
        Fu[self.inner_DOFs_u] += v[self.inner_DOFs_u] * u_y[self.inner_DOFs_u]

        # 1/ρ ∇p
        Fu[self.inner_DOFs_u] += pt_x[self.inner_DOFs_u] / self.rho

        # -ν ∇²u
        Fu[self.inner_DOFs_u] -= self.mu * ((self._DDu_x + self._DDu_y) @ U).reshape(self.shape_u)[self.inner_DOFs_u]

        ########################################
        # v-velocity residual
        # ∂u/∂t + (u ⋅ ∇) u + 1/ρ ∇p - ν ∇²u = 0
        ########################################
        Fv = np.zeros(self.shape_v)

        # ∂u/∂t
        Fv[self.inner_DOFs_v] += vt[self.inner_DOFs_v]

        # (u ⋅ ∇) u
        Fv[self.inner_DOFs_v] += u[self.inner_DOFs_v] * v_x[self.inner_DOFs_v]
        # TODO: Why we have to slice v with inner_DOFs_u here?
        Fv[self.inner_DOFs_v] += v[self.inner_DOFs_v] * v_y[self.inner_DOFs_v]

        # 1/ρ ∇p
        Fv[self.inner_DOFs_v] += pt_y[self.inner_DOFs_v] / self.rho

        # -ν ∇²u
        Fv[self.inner_DOFs_v] -= self.mu * ((self._DDv_x + self._DDv_y) @ V).reshape(self.shape_v)[self.inner_DOFs_v]

        ###################
        # incompressibility
        # ∇ ⋅ u = 0
        ###################
        Fp = np.zeros(self.shape_p)
        # Fp[self.inner_DOFs_p] += u_x[self.inner_DOFs_p] + v_y[self.inner_DOFs_p]

        # self.inner_DOFs_p = np.s_[   1:Ny,    1:Nx]
        # Fp[self.inner_DOFs_p] += u_x[1:-1, :-1] + v_y[:-1, 1:-1]
        # Fp[self.inner_DOFs_p] += u_x[1:-1, :-1] + v_y[1:, 1:-1]
        # Fp[1:-1, 1:-1] = u_x[1:-1, :-1] + v_y[:-1, 1:-1]
        Fp[:, :-1] += u_x
        Fp[:-1, :] += v_y
        # Fp[:, 1:] += u_x
        # Fp[1:, :] += v_y

        # # Fu = u.copy()
        # Fv = v.copy()
        # # Fp = pt.copy()
        # Fp = p.copy()
        
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
        X, Y = np.meshgrid(
            np.linspace(0, self.Lx, self.Nx),
            np.linspace(0, self.Ly, self.Ny),
        )
        fig, ax = plt.subplots()
        
        def update(num):
            ax.clear()
            ax.set_xlim(-0.25 * self.Lx, 1.25 * self.Lx)
            ax.set_ylim(-0.25 * self.Ly, 1.25 * self.Ly)
            ax.set_aspect("equal")
            ax.plot(X, Y, "ok")

            # compute redundant coordinates together with boundary conditions
            u, v, p, ut, vt, pt, U, V, P, Ut, Vt, Pt = self.create_redundant_coordinates(y[:, num], yp[:, num])

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

            # return None,
        
            # contour = ax.contourf(X, Y, np.sqrt(uce**2 + vce**2), alpha=0.5)
            contour = ax.contourf(X, Y, np.sqrt(u[1:, :]**2 + v[:, 1:]**2), alpha=0.5)

            return contour,

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
    Nx = 7
    Ny = 4
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
    t1 = 3
    t_span = (t0, t1)

    # solver options
    # method = "BDF"
    method = "Radau"
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
