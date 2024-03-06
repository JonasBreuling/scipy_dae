# from julia.api import Julia
# jl = Julia(compiled_modules=False)
from diffeqpy import de
import numpy as np
# from scipy.sparse import identity
# from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(du,u,p,t):
  # fmt: off
  resid1 = - 0.04*u[0]               + 1e4*u[1]*u[2] - du[0]
  resid2 = + 0.04*u[0] - 3e7*u[1]**2 - 1e4*u[1]*u[2] - du[1]
  resid3 = u[0] + u[1] + u[2] - 1.0
  return [resid1, resid2, resid3]
  # fmt: on

u0 = [1.0, 0.0, 0.0]
du0 = [-0.04, 0.04, 0.0]
# tspan = (0.0, 100000.0)
tspan = (0.0, 1e1)
differential_vars = [True, True, False]
prob = de.DAEProblem(f, du0, u0, tspan, differential_vars=differential_vars)
# sol = de.solve(prob)
sol = de.solve(prob, alg=de.RadauIIA5(autodiff=False))

u1 = [sol.u[i][0] for i in range(0, len(sol.u))]
u2 = [sol.u[i][1] for i in range(0, len(sol.u))]
u3 = [sol.u[i][2] for i in range(0, len(sol.u))]
# t = np.array(sol.t)
# u = np.array(sol.u)

plt.plot(sol.t, u1)
plt.plot(sol.t, u1)
plt.plot(sol.t, u1)
plt.show()


# n = 10 # dimension
# # atol = 1e-7
# # rtol = 1e-6
# atol = 1e-3
# rtol = 1e-3
# tspan = (0., 1.)

# u0 = np.linspace(0.1, 10, n)
# def f(u,p,t):
#     return -2*u

# def df(u,p,t):
#     return -2. * np.eye(n)
#     # return -2. * identity(n)

# prob = de.ODEProblem(f, u0, tspan) # works
# # prob = de.ODEProblem(f, u0, tspan, jac=df) # fails
# sol = de.solve(prob, alg=de.RadauIIA5(autodiff=False), abstol=atol, reltol=rtol)

# sol2 = solve_ivp(fun=lambda t,y: f(y,None,t),
#                  y0=u0, method='Radau', t_span=tspan,
#                  atol=atol, rtol=rtol, jac=lambda t,y: df(y,None,t) )

# plt.figure()
# plt.plot(sol.t,  sol.u, color='tab:red')
# plt.plot(sol2.t, sol2.y.T, linestyle='--', linewidth=3, color='tab:blue')
# plt.xlabel('t')
# plt.ylabel('u')
# plt.grid()
# plt.show()