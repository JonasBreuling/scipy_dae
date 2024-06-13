import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation  as animation
import scipy.sparse
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


g = 9.81
n = 10
initial_angle = np.pi / 2

## Initial conditions for each pendulum
theta_0 = np.ones(n) * initial_angle
theta_dot0 = np.zeros(n)

m_chord = 1.0
L_chord = 2.0
m_node = m_chord / (n-1)
L_rod = L_chord / n

r0 = L_rod * np.ones(n)
m = m_node * np.ones(n)

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

    # distances between each node, accounting for the fixed zero-th node
    dx = np.hstack((x[0], np.diff(x)))
    dy = np.hstack((y[0], np.diff(y)))

    # same for velocities
    du = np.hstack((u[0], np.diff(u)))
    dv = np.hstack((v[0], np.diff(v)))

    F = np.empty((6, x.size), order='C', dtype=np.common_type(y, yp))
    Fx, Fy, Fu, Fv, constraints, constraints_der = F

    # kinematics
    Fx[:-1] = xp[:-1] - (u[:-1] + 2 * mup[:-1] * dx[:-1] - 2 * mup[1:] * dx[1:])
    Fy[:-1] = yp[:-1] - (v[:-1] + 2 * mup[:-1] * dy[:-1] - 2 * mup[1:] * dy[1:])
    Fx[-1]  = xp[-1] - (u[-1] + 2 * mup[-1] * dx[-1])
    Fy[-1]  = yp[-1] - (v[-1] + 2 * mup[-1] * dy[-1])

    # kinetics
    Fu[:-1] = m[:-1] * up[:-1] - (2 * lap[:-1] * dx[:-1] - 2 * lap[1:] * dx[1:])
    Fv[:-1] = m[:-1] * vp[:-1] - (2 * lap[:-1] * dy[:-1] - 2 * lap[1:] * dy[1:] - m[:-1] * g)
    Fu[-1]  = m[-1] * up[-1] - (2 * lap[-1] * dx[-1])
    Fv[-1]  = m[-1] * vp[-1] - (2 * lap[-1] * dy[-1] - m[-1] * g)

    # constraints
    constraints[:] = dx**2 + dy**2 - r0 * r0
    constraints_der[:] = 2 * dx * du + 2 * dy * dv

    return F.reshape((-1,), order='F')

def jac(t, y, yp, f=None):
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


# theoretical period for a single pendulum of equal length
T_th = 2 * np.pi * np.sqrt(L_chord / g)

## Initial condition (pendulum at an angle)
x0 =  np.cumsum( r0 * np.sin(theta_0))
y0 =  np.cumsum(-r0 * np.cos(theta_0))
vx0 = np.cumsum( r0 * theta_dot0 * np.cos(theta_0))
vy0 = np.cumsum( r0 * theta_dot0 * np.sin(theta_0))
la0 = np.zeros(n)
mu0 = np.zeros(n)

# the initial condition should be consistent with the algebraic equations
t0 = 0
t1 = 2 * T_th
vy0 = np.vstack((x0, y0, vx0, vy0, mu0, la0)).reshape((-1,), order='F')
vyp0 = np.zeros_like(vy0)
vy0, vyp0, fnorm = consistent_initial_conditions(F, jac, 0, vy0, vyp0)
print(f"fnorm: {fnorm}")

# F0 = F(t0, vy0, vyp0)
# Jy0, Jyp0 = jac(t0, vy0, vyp0, F0)
# # sparsity = Jy0, Jyp0
sparsity = None


if __name__=='__main__':
    atol = rtol = 1e-3 # relative and absolute tolerances for time adaptation
    t_span = (t0, t1)
    num = 200
    t_eval = np.linspace(t0, t1, num=num)

    # method = "BDF"
    method = "Radau"
    start = time.time()
    sol = solve_dae(F, t_span, vy0, vyp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, jac=jac)
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

    # #####################
    # # positions over time
    # #####################
    # fig, ax = plt.subplots(2, 1)

    # ax[0].set_title("x")
    # ax[0].plot(t, x.T)
    # ax[0].grid()

    # ax[1].set_title("y")
    # ax[1].plot(t, y.T)
    # ax[1].grid()

    # ##############
    # # trajectories
    # ##############
    # fig, ax = plt.subplots()
    # ax.plot(x.T, y.T)
    # ax.axis("equal")
    # ax.grid()

    ###########
    # animation
    ###########
    fig, ax = plt.subplots()
    xmax = ymax = 1.5 * L_chord
    xmin = ymin = -1.5 * L_chord
    # xmax = 1.2*np.max(np.abs(x))
    # ymax = max(1.0, 1.2*np.max(y))
    # ymin = 1.2*np.min(y)
    # ax = fig.add_subplot(111, autoscale_on=False, xlim=(-xmax,xmax), ylim=(ymin,ymax))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid()
    ax.set_aspect('equal')

    lines, = plt.plot([], [], linestyle='-', linewidth=1.5, marker="o", color='tab:blue') # rods
    points, = plt.plot([],[], marker='o', linestyle='', color=[0,0,0]) # mass joints
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
        time_text.set_text(time_template.format(t[i]))

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