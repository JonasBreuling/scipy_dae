import numpy as np
from scipy.linalg import lu, solve

class RideData:
    def __init__(self):
        self.A = np.array([[0.1968154772236604, -0.0655354258501984, 0.0237709743482202],
                           [0.3944243147390873, 0.2920734116652284, -0.0415487521259979],
                           [0.3764030627004673, 0.5124858261884216, 0.1111111111111111]])
        self.Ai = np.array([[3.224744871391587, 1.167840084690408, -0.253197264742182],
                            [-3.567840084690405, 0.775255128608409, 1.053197264742182],
                            [5.531972647421809, -7.531972647421808, 4.999999999999999]])
        self.b = self.A[2, :]
        self.c = np.array([0.155051025721682, 0.644948974278318, 1])
        self.T = np.array([[0.091232394870893, -0.128458062178301, 0.027308654751321],
                           [0.241717932707107, 0.185635951030957, -0.348248904396575],
                           [0.966048182615093, 0.909403517646862, -0.0]])
        self.Ti = np.array([[4.325579890063155, 0.339199251815810, 0.541770539935875],
                            [-4.595010367196070, -0.360327197335856, 0.524105686036759],
                            [0.552969749058155, -2.828147131551269, 0.655417747196001]])
        self.mu = np.array([3.63783425274449, 2.68108287362775, 3.05043019924741])
        self.c1c2c3 = self.c[0] * self.c[1] * self.c[2]
        self.c1mc2 = self.c[0] - self.c[1]
        self.c1mc3 = self.c[0] - self.c[2]
        self.c2mc1 = self.c[1] - self.c[0]
        self.c2mc3 = self.c[1] - self.c[2]
        self.b0 = 1.0 / self.mu[0]
        self.v = np.array([0.428298294115369, -0.245039074384917, 0.366518439460903])
        self.MAX_NEWT_ITER = 7

        self.rtol = 1.0e-6
        self.atol = 1.0e-6
        self.MAX_STEPS = 1000
        self.DIFF_INDEX = np.array([])
        self.INITIAL_STEP_SIZE = 1.0e-8
        self.T_EVENT = np.array([])

        self.s = 3
        self.n = 0
        self.NTOL = 0.01

        self.M = None
        self.J = None
        self.L = None
        self.U = None
        self.P = None

        self.nfun = 0
        self.njac = 0
        self.n_newt = 0
        self.nr_last = 1
        self.h_last = 0
        self.Z = None
        self.y0 = None
        self.ym1 = None
        self.t = None
        self.Yp = None

        self.ifail = 0
        self.ifail_count = 0
        self.no_update = 0

ride_data = RideData()

def ride(IDEFUN, IDEJAC, tspan, y0, yp0, options=None):
    global ride_data

    if options:
        ride_data.rtol = options.get('RTOL', 1.0e-6)
        ride_data.atol = options.get('ATOL', 1.0e-6)
        ride_data.MAX_STEPS = options.get('MAX_STEPS', 1000)
        ride_data.DIFF_INDEX = options.get('DIFF_INDEX', np.array([]))
        ride_data.INITIAL_STEP_SIZE = options.get('INITIAL_STEP_SIZE', 1.0e-8)
        ride_data.T_EVENT = options.get('T_EVENT', np.array([]))

    s = ride_data.s
    A = ride_data.A
    Ai = ride_data.Ai
    b = ride_data.b
    c = ride_data.c
    T = ride_data.T
    Ti = ride_data.Ti
    mu = ride_data.mu
    c1c2c3 = ride_data.c1c2c3
    c1mc2 = ride_data.c1mc2
    c1mc3 = ride_data.c1mc3
    c2mc1 = ride_data.c2mc1
    c2mc3 = ride_data.c2mc3
    b0 = ride_data.b0
    v = ride_data.v
    MAX_NEWT_ITER = ride_data.MAX_NEWT_ITER

    n = y0.shape[0]
    if y0.ndim != 1:
        print('Error: expecting an n by 1 matrix for y0')
        return None, None, None
    if yp0.shape != y0.shape:
        print('Error: expecting an n by 1 matrix for yp0')
        return None, None, None

    ride_data.n = n
    if ride_data.DIFF_INDEX.size == 0:
        ride_data.DIFF_INDEX = np.zeros(n)
    else:
        if ride_data.DIFF_INDEX.shape != (n,):
            print('Error: expecting an n by 1 matrix for DIFF_INDEX')
            return None, None, None
        if np.any(ride_data.DIFF_INDEX < 0):
            print('Error: invalid data in the matrix DIFF_INDEX')
            return None, None, None

    if (len(ride_data.rtol) != 1 and
        ((len(ride_data.rtol) != len(ride_data.atol)) or (len(ride_data.rtol) != ride_data.n))):
        print('Error: invalid dimensions for RTOL and ATOL')
        return None, None, None
    if np.any(np.array(ride_data.rtol) <= 0):
        print('Error: invalid input for RTOL')
        return None, None, None
    if np.any(np.array(ride_data.atol) <= 0):
        print('Error: invalid input for ATOL')
        return None, None, None

    if not IDEJAC:
        ride_data.FD_JACOBIAN = 1

    t0, tend = tspan[0], tspan[-1]
    dt = tend - t0
    h = min(abs(dt), abs(ride_data.INITIAL_STEP_SIZE))
    inext = 1
    tnext = tspan[inext]

    ntevent = len(ride_data.T_EVENT)
    next_event = 0
    if ntevent > 0:
        dt = ride_data.T_EVENT[next_event] - t0
        h = min(abs(dt), h)

    if h * (tend - t0) < 0:
        h = -h
    t = t0
    y = y0
    yp = yp0
    hnew = h

    if ride_data.FD_JACOBIAN == 0:
        J, M = IDEJAC(y, yp, t)
    else:
        J, M = ride_fd_jacobian(IDEFUN, y, yp, t)
        ride_data.nfun += 2 * n + 1

    ride_data.njac += 1
    ride_data.M = M
    ride_data.J = J
    ride_data.h_last = h
    ride_data.y0 = y0
    ride_data.ym1 = y0
    ride_data.Z = np.zeros(n * s)
    ride_data.Yp = np.zeros(n * (s + 1))
    ride_data.t = np.array([t0, t0 + ride_data.c[0] * h, t0 + ride_data.c[1] * h, t0 + ride_data.c[2] * h])
    ctol = 0.001

    if len(ride_data.rtol) == 1:
        ride_data.NTOL = max(10 * np.finfo(float).eps / ride_data.rtol,
                             min(ctol, np.sqrt(ride_data.rtol)))
    else:
        rtol = min(ride_data.rtol)
        ride_data.NTOL = max(10 * np.finfo(float).eps / rtol,
                             min(ctol, np.sqrt(rtol)))

    Tout = [t0]
    Yout = [y0]

    for step in range(ride_data.MAX_STEPS):
        Y, Yp, err = ride_newton(IDEFUN, y, yp, t, h)
        if err == 0:
            hnew, err = ride_lte(IDEFUN, Y, Yp, y, yp, t, h)
            if err == 0:
                h = hnew
                yp = Yp[:, 2]
                y = Y[:, 2]
                t = ride_data.t[3]
                Tout.append(t)
                Yout.append(y)
                ride_data.ym1 = y
                ride_data.h_last = h

                if abs(t - tnext) < 10 * np.finfo(float).eps:
                    inext += 1
                    tnext = tspan[inext]

                if ntevent > 0:
                    if abs(t - ride_data.T_EVENT[next_event]) < 10 * np.finfo(float).eps:
                        next_event += 1
                        if next_event == ntevent:
                            ntevent = 0

                if ntevent > 0:
                    dt = ride_data.T_EVENT[next_event] - t
                else:
                    dt = tnext - t

                h = min(abs(h), abs(dt))

                if h * (tend - t) < 0:
                    h = -h

                if t == tend:
                    break
            else:
                h = 0.5 * h
                ride_data.h_last = h
        else:
            h = 0.5 * h
            ride_data.h_last = h

        if step == ride_data.MAX_STEPS - 1:
            ride_data.ifail = 1
            print('Error: maximum number of steps reached')
            break

    Tout = np.array(Tout)
    Yout = np.array(Yout).T

    return Yout, Tout, ride_data.ifail

def ride_fd_jacobian(IDEFUN, y, yp, t):
    n = y.shape[0]
    f0, g0 = IDEFUN(y, yp, t)
    J = np.zeros((n, n))
    M = np.zeros((n, n))
    dy = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y), 1.0 / ride_data.NTOL)
    dyp = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(yp), 1.0 / ride_data.NTOL)

    for i in range(n):
        yi = y[i]
        y[i] += dy[i]
        f, g = IDEFUN(y, yp, t)
        J[:, i] = (f - f0) / dy[i]
        y[i] = yi

    for i in range(n):
        ypi = yp[i]
        yp[i] += dyp[i]
        f, g = IDEFUN(y, yp, t)
        M[:, i] = (g - g0) / dyp[i]
        yp[i] = ypi

    return J, M

def ride_newton(IDEFUN, y, yp, t, h):
    global ride_data

    err = 0
    y0 = y
    yp0 = yp
    y = np.tile(y, (3, 1))
    yp = np.tile(yp, (3, 1))
    y[:, 1] = y0 + h * ride_data.c[0] * yp0
    y[:, 2] = y[:, 1] + h * ride_data.c[1] * yp[:, 1]
    yp[:, 1] = yp0
    yp[:, 2] = yp[:, 1]

    for k in range(ride_data.MAX_NEWT_ITER):
        Z = np.zeros((ride_data.n, 3))
        rhs = np.zeros(ride_data.n * 3)

        for i in range(3):
            rhs[i * ride_data.n:(i + 1) * ride_data.n] = (y[:, i] - ride_data.y0
                                                          - h * np.sum(ride_data.Ai[i, :] * yp, axis=1))

        rhs -= np.concatenate((h * ride_data.Ai @ (ride_data.M @ y), 
                               h * ride_data.Ai @ (ride_data.J @ yp))).flatten()

        Z = solve(ride_data.P @ (ride_data.L @ (ride_data.U @ rhs.reshape((ride_data.n, 3)))), rhs)
        y = y - Z[:, :ride_data.n]
        yp = yp - Z[:, ride_data.n:2*ride_data.n]

        if np.linalg.norm(rhs) < ride_data.rtol * (np.linalg.norm(y[:, 2]) + ride_data.atol):
            break

    ride_data.n_newt += 1

    if np.linalg.norm(rhs) >= ride_data.rtol * (np.linalg.norm(y[:, 2]) + ride_data.atol):
        err = 1

    return y, yp, err

        self.C = C
        self.Y = Y
        self.Yp = Yp
def ride_lte(IDEFUN, Y, Yp, y, yp, t, h):
    global ride_data

    f, g = IDEFUN(y, yp, t)
    ride_data.nfun += 1
    err = 0

    L = np.sum(Y[:, -1] * ride_data.v)
    LTE = np.linalg.norm(ride_data.c1c2c3 * h * h * h * (g - L) / ride_data.NTOL)
    hnew = h

    if LTE > ride_data.rtol:
        hnew = h * min(0.9 * np.sqrt(ride_data.rtol / LTE), 1.0 / 0.2)
        if hnew < 0.1 * h:
            hnew = 0.1 * h
            err = 1
    else:
        if LTE < 0.1 * ride_data.rtol:
            hnew = h * min(1.0 / 0.9 * np.sqrt(ride_data.rtol / LTE), 2.0)
            if hnew > 5.0 * h:
                hnew = 5.0 * h

    return hnew, err
