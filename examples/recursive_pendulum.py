import tqdm
import time
import numpy as np
import scipy.sparse
from scipy.optimize._numdiff import approx_derivative
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


g = 9.81
n = 3
initial_angle = np.pi / 2

## Initial conditions for each pendulum
theta_0 = np.ones(n) * initial_angle
theta_dot0 = np.zeros(n)

if True: # chord with a mass at the end
    m_chord = 1.
    m_final = 1.
    L_chord = 2.
    m_node = m_chord / (n-1)
    L_rod  = L_chord / n

    r0  = np.array([L_rod for i in range(n)])
    r0s = r0**2
    m   = np.array([m_node for i in range(n)])
    m[-1] = m_final
else: # also at the middle
    m_chord  = 1.
    m_final  = 5.
    m_middle = 5.

    L_chord = 2.
    m_node = m_chord / n
    L_rod  = L_chord / n

    r0  = np.array([L_rod for i in range(n)])
    r0s = r0**2
    m   = np.array([m_node for i in range(n)])
    m[-1] = m_final
    m[len(m)//2]=m_middle

###### DAE function and Jacobian
def F(t, vy, vyp):
    x = vy[0::6]
    y = vy[1::6]
    u = vy[2::6]
    v = vy[3::6]
    mu = vy[4::6]
    la = vy[5::6]

    xp = vyp[0::6]
    yp = vyp[1::6]
    up = vyp[2::6]
    vp = vyp[3::6]
    mup = vyp[4::6]
    lap = vyp[5::6]

    F = np.empty((6, x.size), order='C', dtype=np.common_type(y, yp))
    Fx, Fy, Fu, Fv, constraints, constraints_der = F

    # kinematics
    Fx[:] = xp - (u + 2 * x * mup)
    Fy[:] = yp - (v + 2 * y * mup)

    # # kinetics
    # Fx[:] = xp - (u + 2 * x * lap)
    # Fy[:] = yp - (v + 2 * y * lap)

    # dx = np.hstack((x[0], np.diff(x))) # distances between each node, accounting for the fixed zero-th node
    # dy = np.hstack((y[0], np.diff(y)))
    dx = np.empty_like(x)
    dy = np.empty_like(y)
    dx[0]  = x[0]
    dx[1:] = np.diff(x) # distances between each node, accounting for the fixed zero-th node
    dy[0]  = y[0]
    dy[1:] = np.diff(y) # distances between each node, accounting for the fixed zero-th node

    rs = dx**2 + dy**2

    du = np.empty_like(u)
    dv = np.empty_like(v)
    du[0]  = u[0]
    du[1:] = np.diff(u)
    dv[0]  = v[0]
    dv[1:] = np.diff(v)

    rsp = 2 * dx * du + 2 * dy * dv

    Fu[:-1] = up[:-1] - (1/m[:-1]) * (+lap[:-1]*dx[:-1]/rs[:-1] - lap[1:]*dx[1:]/rs[1:])
    Fv[:-1] = vp[:-1] - (1/m[:-1]) * (+lap[:-1]*dy[:-1]/rs[:-1] - lap[1:]*dy[1:]/rs[1:] - m[:-1]*g)

    Fu[-1]  = up[-1] - (1/m[-1]) *  (+lap[-1]*dx[-1]/rs[-1])
    Fv[-1]  = vp[-1] - (1/m[-1]) *  (+lap[-1]*dy[-1]/rs[-1] - m[-1]*g)

    constraints[:] = rs - r0s
    constraints_der[:] = rsp
    return F.reshape((-1,), order='F')

def jac(t, y, yp, f):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)

    J = approx_derivative(lambda z: fun_composite(t, z), 
    z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp

sparsity = scipy.sparse.diags(diagonals=[np.ones(n*5-abs(i)) for i in range(-10,10)], offsets=[i for i in range(-10,10)],
format='csc')
# jac = None

## Initial condition (pendulum at an angle)
x0 =  np.cumsum( r0*np.sin(theta_0))
y0 =  np.cumsum(-r0*np.cos(theta_0))
vx0 = np.cumsum( r0*theta_dot0*np.cos(theta_0))
vy0 = np.cumsum( r0*theta_dot0*np.sin(theta_0))
la0 = np.zeros(n)#(m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis
# the initial value for lbda does not change the solution, since this variable will be readjusted by the solver
mu0 = np.zeros(n)

# the initial condition should be consistent with the algebraic equations
vy0 = np.vstack((x0, y0, vx0, vy0, mu0, la0)).reshape((-1,), order='F')

# theoretical period for a single pendulum of equal length
T_th = 2 * np.pi * np.sqrt(L_chord / g)


if __name__=='__main__':
    atol = rtol = 1e-3 # relative and absolute tolerances for time adaptation

    # F, jac, sparsity, vy0, T_th, m = generateSystem(n, initial_angle)
    t0 = 0
    tf = 2 * T_th # simulate 2 periods
    # tf = 0.1
    t_span = (t0, tf)

    vyp0 = np.zeros_like(vy0)
    vy0, vyp0, fnorm = consistent_initial_conditions(F, jac, t0, vy0, vyp0)
    print(f"fnorm: {fnorm}")
    # exit()

    method = "BDF"
    start = time.time()
    sol = solve_dae(F, t_span, vy0, vyp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    vy = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    if 0:
        #%% Initial state, study Jacobian
        jacfull_fun = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                            fun=lambda x: dae_fun(t,x),
                            x0=x, method='2-point',
                            rel_step=1e-8, f0=None,
                            bounds=(-np.inf, np.inf), sparsity=None,
                            kwargs={})
        jacobian = jacfull_fun(0,Xini+(1e-3+ abs(Xini)*1e-3)*np.random.random(Xini.size))

        dae_fun(0,Xini+(1e-3+ abs(Xini)*1e-3)*np.random.random(Xini.size))

        plt.figure()
        plt.spy(jacobian)
        for i in range(n):
          plt.axhline(i*5-0.5, color='tab:gray', linewidth=0.5)
          plt.axvline(i*5-0.5, color='tab:gray', linewidth=0.5)
        # plt.grid()
        plt.title('Jacobian of the DAE function at perturbed initial state')

    # recover the time history of each variable
    x = vy[0::6]
    y = vy[1::6]
    u = vy[2::6]
    v = vy[3::6]
    mu = vy[4::6]
    la = vy[5::6]

    fig, ax = plt.subplots()
    ax.plot(x.T, y.T, "-o")
    # ax.set_xlim(-1.5 * r0, 1.5 * r0)
    ax.axis("equal")
    plt.show()
    exit()

    # dx = np.vstack((x[0,:], np.diff(x, axis=0)))
    # dy = np.vstack((y[0,:], np.diff(y, axis=0)))
    # r = (dx**2 + dy**2)**0.5
    # theta = np.arctan(dx / dy)

    # np.diff(theta, axis=0) * 180/np.pi
    # length = np.sum(r, axis=0)

    # # plot total energy
    # Ec = 0.5*(u**2 + v**2).T.dot(m)
    # Ep = g * y.T.dot(m)

    # plt.figure(dpi=200)
    # plt.plot(t,Ec,color='r', label='Ec')
    # plt.plot(t,Ep,color='b', label='Ep')
    # plt.plot(t,Ec+Ep,color='g', label='Etot', linestyle='--')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('t (s)')
    # plt.ylabel('Energy (J)')


    # # Relative angles
    # # It does not make much sense to use complex numbers, but anyway...
    # plt.figure()
    # # projvec = dx[:-1] + 1j*dy[:-1]
    # # projvec_norm = (projvec.real**2 + projvec.imag**2)**0.5
    # # projvec = projvec / projvec_norm

    # # vectors = dx[1:] + 1j*dy[1:]
    # # vectors_norm = (vectors.real**2 + vectors.imag**2)**0.5

    # # colinear_part = (vectors.real * projvec.real +  vectors.imag * projvec.imag )* projvec
    # # colinear_part_norm = (colinear_part.real**2 + colinear_part.imag**2)**0.5

    # # perpendicular_part = vectors - colinear_part * projvec
    # # perpendicular_part_norm = (perpendicular_part.real**2 + perpendicular_part.imag**2)**0.5

    # # assert np.allclose(perpendicular_part_norm**2 + colinear_part_norm**2, vectors_norm**2)

    # # angles = computeAngle(x=colinear_part, y=perpendicular_part)

    # # plt.plot(t, angles.T)
    # plt.plot(t, np.diff(theta.T,axis=1))
    # plt.grid()
    # plt.xlabel('t (s)')
    # plt.ylabel('angle (rad)')
    # plt.title('Relative angle between successive rods')

    # # plot joint forces
    # plt.figure()
    # plt.semilogy(t, np.abs(la.T))
    # plt.grid()
    # plt.xlabel('t (s)')
    # plt.ylabel('absolute rod tension (N)')


    # plt.figure(figsize=(5,5))
    # for i in range(n):
    #   plt.plot(x[i, :], y[i, :], linestyle='--')
    # plt.grid()

    # # add first and last form
    # plt.plot( np.hstack((0., x[:,-1])), np.hstack((0., y[:,-1])), color='r')
    # plt.plot( np.hstack((0., x[:,0])), np.hstack((0., y[:,0])), color=[0,0,0])
    # plt.scatter(0.,0.,marker='o', color=[0,0,0])
    # for i in range(n):
    #   for j in [0, -1]:
    #     plt.scatter(x[i, j], y[i, j],marker='o', color=[0, 0, 0])

    # plt.axis('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # L0 = length[0]
    # plt.ylim(-L0, L0)
    # plt.xlim(-L0, L0)

    # # Find period
    # Ep_mean = (np.max(Ep) + np.min(Ep)) / 2
    # t_change = t[:-1][np.diff(np.sign(Ep - Ep_mean)) < 0]
    # print('Numerical period={:.2e} s'.format(2 * np.mean(np.diff(t_change)))) # energy oscillates twice as fast
    # print('Theoretical period={:.2e} s'.format(T_th))


    import numpy as np
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.animation  as animation

    fig = plt.figure(figsize=(5, 5), dpi=200)
    xmax = 1.2*np.max(np.abs(x))
    ymax = max(1.0, 1.2*np.max(y))
    ymin = 1.2*np.min(y)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-xmax,xmax), ylim=(ymin,ymax))
    ax.grid()
    ax.set_aspect('equal')

    lines, = plt.plot([], [], linestyle='-', linewidth=1.5, marker="o", color='tab:blue') # rods
    marker='o'
    # if n<50:
    #     marker='o'
    # else:
    #     marker=None
    points, = plt.plot([],[], marker=marker, linestyle='', color=[0,0,0]) # mass joints
    # paths  = [plt.plot([],[], alpha=0.2, color='tab:orange')[0] for i in range(n)] # paths of the pendulum
    time_template = r'$t = {:.2f}$ s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        return lines, points

    # def update(t):
    #     interped_sol = sol.sol(t)
    #     x,y = np.hstack((0.,interped_sol[::5])), np.hstack((0.,interped_sol[1::5]))
    def update(i):
        x_ = x[:, i]
        y_ = y[:, i]
        lines.set_data([0, *x_], [0, *y_])
        # points.set_data(x,y)
        # points.set_data(x[-1],y[-1]) # only plot last mass
        # points.set_data(x_[-1],y_[-1]) # only plot last mass
        # time_text.set_text(time_template.format(t))

        return lines, time_text, points

    if False:# test layout
        init()
        update(sol.t[-1]/2)
    else:
        # compute how many frames we want for real time
        fps=30
        total_frames = np.ceil((sol.t[-1]-sol.t[0])*fps).astype(int)

        from tqdm import tqdm
        # ani = animation.FuncAnimation(fig, update, frames=tqdm(np.linspace(sol.t[0], sol.t[-1], total_frames)),
        ani = animation.FuncAnimation(fig, update, frames=tqdm(np.arange(total_frames)),
        init_func=init, interval=200, blit=True)
        # ani.save('animation_new12.gif', writer='imagemagick', fps=30)
        # writer = animation.writers['ffmpeg'](fps=24, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save('animation.mp4', writer=writer)
        plt.show()