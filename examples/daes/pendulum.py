import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
import numpy as np
from scipy.special import ellipj, ellipk


"""Cartesian pendulum, see Hairer1996 Section VII Example 2."""
m = 1
l = 1
g = 10

def F(t, vy, vyp):
    # stabilized index 1
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, la, mu = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u - 2 * x * mu
    R[1] = y_dot - v - 2 * y * mu
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g
    R[4] = 2 * x * u + 2 * y * v
    R[5] = x * x + y * y - l * l

    return R


# see https://arxiv.org/pdf/1007.4026
def pendulum_cartesian_solution_velocity(theta0, length, gravity, time_points):
    """
    Compute the Cartesian coordinates and velocities of a pendulum bob.
    
    Parameters:
    - theta0: float, maximum angular displacement (amplitude) in radians
    - length: float, length of the pendulum in meters
    - gravity: float, gravitational acceleration in m/s^2
    - time_points: array-like, time points where the solution and velocity are evaluated
    
    Returns:
    - x: array, x-coordinates of the pendulum bob
    - y: array, y-coordinates of the pendulum bob
    - vx: array, x-velocity of the pendulum bob
    - vy: array, y-velocity of the pendulum bob
    """
    # Parameters
    k = np.sin(theta0 / 2)**2  # elliptic modulus
    
    # Precompute constants
    omega0 = np.sqrt(gravity / length)
    K = ellipk(k)
    scaling_factor = np.sqrt(2 * gravity / length)
        
    # Compute the Jacobi elliptic functions
    sn, cn, dn, _ = ellipj(K - omega0 * time_points, k)

    sn_dot = omega0 / (2 * K) * cn * dn
    
    # Compute angular displacement and angular velocity
    theta = 2 * np.arcsin(np.sqrt(k) * sn)
    thetha_dot = scaling_factor * np.sqrt(np.cos(theta) - np.cos(theta0)) * dn
    thetha_dot = 2 * k / np.sqrt(1 - (k * sn)**2) * sn_dot
    thetha_dot = k * omega0 * cn * dn / (K * np.sqrt(1 - (k * sn)**2))
    # thetha_dot *= 2 * np.pi
    thetha_dot *= 5
    
    # Cartesian coordinates
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    
    # Cartesian velocities
    u = length * thetha_dot * np.cos(theta)
    v = length * thetha_dot * np.sin(theta)
    
    return x, y, u, v


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([l, 0, 0, 0, 0, 0], dtype=float)
    yp0 = np.array([0, 0, 0, -g, 0, 0], dtype=float)

    yp0 = np.zeros_like(yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, f0 = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"f: {f0}")
    assert np.allclose(f0, np.zeros_like(f0))

    # solver options
    atol = rtol = 1e-4

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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

    # elliptical integral solution
    theta0 = -np.pi / 2
    theta0 = np.pi / 2
    x_ell, y_ell, u_ell, v_ell = pendulum_cartesian_solution_velocity(theta0, l, g, t)

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y[0], "-ok", label="x")
    ax[0].plot(t, y[1], "--xk", label="y")
    ax[0].plot(t, x_ell, "xr", label="x_ell")
    ax[0].plot(t, y_ell, "or", label="y_ell")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-ok", label="u")
    ax[1].plot(t, y[3], "--xk", label="v")
    ax[1].plot(t, -u_ell, "xr", label="u_ell")
    ax[1].plot(t, -v_ell, "or", label="v_ell")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, yp[4], "-ok", label="la")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, yp[5], "--xk", label="mu")
    ax[3].legend()
    ax[3].grid()

    plt.show()


# # Example usage
# theta0 = np.radians(30)  # maximum displacement of 30 degrees
# length = 1.0  # pendulum length in meters
# gravity = 9.81  # gravitational acceleration in m/s^2
# time_points = np.linspace(0, 20, 500)  # time points in seconds

# x, y, vx, vy = pendulum_cartesian_solution_velocity(theta0, length, gravity, time_points)

# # For visualization
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.subplot(3, 1, 1)
# plt.plot(x, y, label='Pendulum Path')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.title('Pendulum Cartesian Path')
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(time_points, x, label='X (m)', color='red')
# plt.plot(time_points, y, label='Y (m)', color='blue')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.title('Cartesian Positions')
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(time_points, vx, label='Velocity X (m/s)', color='red')
# plt.plot(time_points, vy, label='Velocity Y (m/s)', color='blue')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Cartesian Velocities')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()
