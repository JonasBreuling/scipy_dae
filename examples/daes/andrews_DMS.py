import numpy as np
from tqdm import tqdm
from scipy.optimize._numdiff import approx_derivative
from pendulum_DMS3 import half_explicit_Euler
import matplotlib.pyplot as plt

"""Andrews squeezing mechanism - index 3 DAE.

References:
-----------
https://archimede.uniba.it/~testset/problems/andrews.php
"""

# parameters
nq = 7
nla = 6

m1 = 0.04325
I1 = 2.194e-6
ss = 0.035

m2 = 0.00365
I2 = 4.410e-7
sa = 0.01874

m3 = 0.02373
I3 = 5.255e-6
sb = 0.01043

m4 = 0.00706 
I4 = 5667e-7
sc = 0.018

m5 = 0.07050
I5 = 1.169e-5
sd = 0.02

m6 = 0.00706
I6 = 5.667e-7
ta = 0.02308

m7 = 0.05498
I7 = 1.912e-5
tb = 0.00916

xa = 0.06934
d = 0.028
u = 0.04

ya = 0.00227
da = 0.0115
ua = 0.01228

xb = 0.03635
e = 0.02
ub = 0.00449

yb = 0.03273
ea = 0.01421
zf = 0.02

xc = 0.014
rr = 0.007
zt = 0.04

yc = 0.072
ra = 0.00092
fa = 0.01421

c0 = 4530
l0 = 0.07785
mom = 0.033

class Andrews:

    def M(self, t, vq):
        M = np.zeros((nq, nq))

        q1, q2, q3, q4, q5, q6, q7 = vq

        M[0, 0] = m1 * ra**2 + m2 * (rr - 2 * da * rr * np.cos(q2) + da**2) + I1 + I2
        M[1, 0] = M[0, 1] = m2 * (da**2 - da * rr * np.cos(q2)) + I2
        M[1, 1] = m2 * da**2 + I2
        M[2, 2] = m3 * (sa**2 + sb**2) + I3
        M[3, 3] = m4 * (e - ea**2) + I4
        M[4, 3] = M[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * np.sin(q4)) + I4
        M[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * np.sin(q4) + (e - ea)**2) + m5 * (ta**2 + tb**2) + I4 + I5
        M[5, 5] = m6 * (zf - fa)**2 + I6
        M[6, 5] = M[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * np.sin(q6)) + I6
        M[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * np.sin(q6) + u**2) + m7 * (ua**2 + ub**2) + I6 + I7

        return M

    def h(self, t, vq, vu):
        q1, q2, q3, q4, q5, q6, q7 = vq
        u1, u2, u3, u4, u5, u6, u7 = vu

        xd = sd * np.cos(q3) + sc * np.sin(q3) + xb
        yd = sd * np.sin(q3) - sc * np.cos(q3) + yb
        L = np.sqrt((xd - xc)**2 + (yd - yc)**2)
        F = -c0 * (L - l0) / L
        Fx = F * (xd - xc)
        Fy = F * (yd - yc)

        h = np.zeros(nq)
        h[0] = mom - m2 * da * rr * u2 * (u2 + 2 * u1) * np.sin(q2)
        h[1] = m2 * da * rr * u1**2 * np.sin(q2)
        h[2] = Fx * (sc * np.cos(q3) - sd * np.sin(q3)) + Fy * (sd * np.cos(q3) + sc * np.sin(q3))
        h[3] = m4 * zt * (e - ea) * u5**2 * np.cos(q4)
        h[4] = -m4 * zt * (e - ea) * u4 * (u4 + 2 * u5) * np.cos(q4)
        h[5] = -m6 * u * (zf - fa) * u7**2 * np.cos(q6)
        h[6] = m6 * u * (zf - fa) * u6 * (u6 + 2 * q7) * np.cos(q6)

        return h

    def g(self, t, vq):
        q1, q2, q3, q4, q5, q6, q7 = vq

        g = np.zeros(nla)
        g[0] = rr * np.cos(q1) - d * np.cos(q1 + q2) - ss * np.sin(q3) - xb
        g[1] = rr * np.sin(q1) - d * np.sin(q1 + q2) + ss * np.cos(q3) - yb
        g[2] = rr * np.cos(q1) - d * np.cos(q1 + q2) - e * np.sin(q4 + q5) - zt * np.cos(q5) - xa
        g[3] = rr * np.sin(q1) - d * np.sin(q1 + q2) + e * np.cos(q4 + q5) - zt * np.sin(q5) - ya
        g[4] = rr * np.cos(q1) - d * np.cos(q1 + q2) - zf * np.cos(q6 + q7) - u * np.sin(q7) - xa
        g[5] = rr * np.sin(q1) - d * np.sin(q1 + q2) - zf * np.sin(q6 + q7) + u * np.cos(q7) - ya
        return g
    
    def g_q(self, t, vq):
        q1, q2, q3, q4, q5, q6, q7 = vq

        g_q = np.zeros((nla, nq))

        # g[0] = rr * np.cos(q1) - d * np.cos(q1 + q2) - ss * np.sin(q3) - xb
        g_q[0, 0] = -rr * np.sin(q1) + d * np.sin(q1 + q2)
        g_q[0, 1] = d * np.sin(q1 + q2)
        g_q[0, 2] = -ss * np.cos(q3)

        # g[1] = rr * np.sin(q1) - d * np.sin(q1 + q2) + ss * np.cos(q3) - yb
        g_q[1, 0] = rr * np.cos(q1) - d * np.cos(q1 + q2)
        g_q[1, 1] = - d * np.cos(q1 + q2)
        g_q[1, 2] = -ss * np.sin(q3)

        # g[2] = rr * np.cos(q1) - d * np.cos(q1 + q2) - e * np.sin(q4 + q5) - zt * np.cos(q5) - xa
        g_q[2, 0] = -rr * np.sin(q1) + d * np.sin(q1 + q2)
        g_q[2, 1] = d * np.sin(q1 + q2)
        g_q[2, 3] = -e * np.cos(q4 + q5)
        g_q[2, 4] = -e * np.cos(q4 + q5) + zt * np.sin(q5)

        # g[3] = rr * np.sin(q1) - d * np.sin(q1 + q2) + e * np.cos(q4 + q5) - zt * np.sin(q5) - ya
        g_q[3, 0] = rr * np.cos(q1) - d * np.cos(q1 + q2)
        g_q[3, 1] = - d * np.cos(q1 + q2)
        g_q[3, 3] = -e * np.sin(q4 + q5)
        g_q[3, 4] = -e * np.sin(q4 + q5) - zt * np.cos(q5)

        # g[4] = rr * np.cos(q1) - d * np.cos(q1 + q2) - zf * np.cos(q6 + q7) - u * np.sin(q7) - xa
        g_q[4, 0] = -rr * np.sin(q1) + d * np.sin(q1 + q2)
        g_q[4, 1] = d * np.sin(q1 + q2)
        g_q[4, 5] = zf * np.sin(q6 + q7)
        g_q[4, 6] = zf * np.sin(q6 + q7) - u * np.cos(q7)

        # g[5] = rr * np.sin(q1) - d * np.sin(q1 + q2) - zf * np.sin(q6 + q7) + u * np.cos(q7) - ya
        g_q[5, 0] = rr * np.cos(q1) - d * np.cos(q1 + q2)
        g_q[5, 1] = -d * np.cos(q1 + q2)
        g_q[5, 5] = -zf * np.cos(q6 + q7)
        g_q[5, 6] = -zf * np.cos(q6 + q7) - u * np.sin(q7)

        return g_q


    def W(self, t, vq):
        W = self.g_q(t, vq).T
        return W
    
        W_num = approx_derivative(lambda vq: self.g(t, vq), vq, method="2-point").T
        diff = W - W_num
        error = np.linalg.norm(diff)
        print(f"error: {error}")
        return W_num


if __name__ == "__main__":
    # time span
    t0 = 0
    t = 0.03
    t_span = (t0, t)

    # mechanical system
    system = Andrews()

    # initial conditions
    q0 = np.array([
        -0.0617138900142764496358948458001,
        0,
        0.455279819163070380255912382449,
        0.222668390165885884674473185609,
        0.487364979543842550225598953530,
        -0.222668390165885884674473185609,
        1.23054744454982119249735015568,
    ])
    u0 = np.zeros(nq)

    ##############
    # dae solution
    ##############
    atol = 1e-3
    rtol = 1e-6
    h = 1e-4
    sol1 = half_explicit_Euler(system, q0, u0, t_span, h, atol=atol, rtol=rtol)
    t = sol1.t
    q = sol1.q
    u = sol1.u

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, q[0], "-", label="x")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, q[1], "-", label="y")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, q[0]**2 + q[1]**2 - l**2, "-", label="g")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, 2 * q[0] * u[0] + 2 * q[1] * u[1], "-", label="g_dot")
    ax[3].legend()
    ax[3].grid()

    plt.show()
