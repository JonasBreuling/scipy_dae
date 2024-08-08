import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions

m = 1.25
Theta = 0.13
a = 0.1
g = 9.81
# Omega = 1
# alpha = np.pi / 4
Omega = 2 * np.random.rand(1)[0]
alpha = np.random.rand(1)[0]
salpha = np.sin(alpha)

"""Index 2 DAE found in Section 1.6 of Bloch2015.

References:
-----------
Bloch2015: https://doi.org/10.1007/978-1-4939-3017-3
"""
def F(t, vy, vyp):
    x, y, phi, u, v, omega, _ = vy
    xp, yp, phip, up, vp, omegap, lap = vyp

    sphi, cphi = np.sin(phi), np.cos(phi)

    F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))

    # Bloch 2005, equation (1.7.6)
    F[0] = xp - u
    F[1] = yp - v
    F[2] = phip - omega
    F[3] = m * up - m * g * salpha - sphi * lap
    F[4] = m * vp + cphi * lap 
    F[5] = Theta * omegap
    F[6] = u * sphi - v * cphi

    return F

def sol_true(t):
    x = (g * salpha / (2 * Omega**2)) * np.sin(Omega * t)**2
    y = (g * salpha / (2 * Omega**2)) * (Omega * t - 0.5 * np.sin(2 * Omega * t))
    phi = Omega * t
    
    u =  (g * salpha / Omega) * np.sin(Omega * t) * np.cos(Omega * t)
    # v = (g * salpha / (2 * Omega)) * (1 - np.cos(2 * Omega * t))
    v = (g * salpha / Omega) * np.sin(Omega * t)**2
    omega = Omega * np.ones_like(t)
    
    # La = -np.sin(0.5 * Omega * t)**2 * salpha * g * 4 / Omega
    # Lap = -np.sin(0.5 * Omega * t) * np.cos(0.5 * Omega * t) * salpha * g * 4
    La = (2 * m * g * salpha / Omega) * (np.cos(Omega * t) - 1)
    Lap = -2 * m * g * salpha * np.sin(Omega * t)

    return np.array([
        x,
        y,
        phi,
        u,
        v,
        omega,
        La,
        Lap,
    ]) 


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = (np.pi / Omega) * 2
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-5

    # initial conditions
    x0 = 0
    y0 = 0
    phi0 = 0
    u0 = 0
    v0 = 0
    omega0 = Omega

    vy0 = np.array([x0, y0, phi0, u0, v0, omega0, 0], dtype=float)
    vyp0 = np.array([u0, v0, omega0, 0, 0, 0, 0], dtype=float)

    print(f"vy0: {vy0}")
    print(f"vyp0: {vyp0}")
    vy0, vyp0, fnorm = consistent_initial_conditions(F, t0, vy0, vyp0)
    print(f"vy0: {vy0}")
    print(f"vyp0: {vyp0}")
    print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    # method = "BDF"
    method = "Radau"
    start = time.time()
    sol = solve_dae(F, t_span, vy0, vyp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    tp = t
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

    # visualization
    fig, ax = plt.subplots(3, 3)

    ax[0, 0].plot(t, sol_true(t)[0], "-ok", label="x_true")
    ax[0, 0].plot(t, y[0], "--xr", label="x")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(t, sol_true(t)[1], "-ok", label="y_true")
    ax[0, 1].plot(t, y[1], "--xr", label="y")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].plot(t, sol_true(t)[2], "-ok", label="phi_true")
    ax[0, 2].plot(t, y[2], "--xr", label="phi")
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].plot(t, sol_true(t)[3], "-ok", label="u_true")
    ax[1, 0].plot(t, y[3], "--xr", label="u")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].plot(t, sol_true(t)[4], "-ok", label="v_true")
    ax[1, 1].plot(t, y[4], "--xr", label="v")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].plot(t, sol_true(t)[5], "-ok", label="omega_true")
    ax[1, 2].plot(t, y[5], "--xr", label="omega")
    ax[1, 2].grid()
    ax[1, 2].legend()

    ax[2, 0].plot(t, sol_true(t)[6], "-ok", label="La_true")
    ax[2, 0].plot(t, y[6], "--xr", label="La")
    ax[2, 0].grid()
    ax[2, 0].legend()


    ax[2, 1].plot(t, sol_true(t)[7], "-ok", label="Lap_true")
    ax[2, 1].plot(t, yp[6], "--xr", label="Lap")
    ax[2, 1].grid()
    ax[2, 1].legend()

    ax[2, 2].plot(y[0], y[1], "-k", label="(x, y)")
    ax[2, 2].grid()
    ax[2, 2].legend()
    ax[2, 2].axis('equal')

    # ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    # ax.plot(t, np.sin(t), "-g", label="y2 true")

    # ax.grid()
    # ax.legend()

    plt.show()
