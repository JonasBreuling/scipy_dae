import matplotlib.pyplot as plt

def plot_constant_current_simulation(ts, w, I_vec=None):
    """
    Function for plotting data from one constant current simulation.

    Parameters:
    ts : numpy.ndarray
        Time steps (in seconds).
    w : numpy.ndarray
        Solution matrix with shape (12, n), where n is the number of time steps.
    I_vec : numpy.ndarray or None
        Optional dynamic current load vector (same length as ts). Default is None.
    """
    Stot = w[0, :] + w[1, :] + w[2, :] + w[3, :] + w[4, :]

    # Plot species masses
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(ts, w[0, :], '.', label='S8')
    plt.plot(ts, w[1, :], '.', label='S4')
    plt.plot(ts, w[2, :], '.', label='S2')
    plt.plot(ts, w[3, :], '.', label='S')
    plt.plot(ts, w[4, :], '.', label='Sp')
    plt.plot(ts, Stot, '.', label='S_tot')
    plt.legend(loc='upper left')
    plt.xlabel('t (s)')
    plt.ylabel('Species mass (g)')

    # Plot reaction currents
    plt.subplot(2, 2, 2)
    plt.plot(ts, w[5, :], label='IH')
    plt.plot(ts, w[6, :], label='IL')
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('Reaction current (A)')

    # Plot reaction overpotentials
    plt.subplot(2, 2, 3)
    plt.plot(ts, w[8, :], label='etaH')
    plt.plot(ts, w[9, :], label='etaL')
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('Reaction overpotential (V)')

    # Plot Nernst and cell voltages
    plt.subplot(2, 2, 4)
    plt.plot(ts, w[10, :], label='ENH')
    plt.plot(ts, w[11, :], label='ENL')
    plt.plot(ts, w[7, :], label='V')
    plt.legend(loc='lower left')
    plt.xlabel('t (s)')
    plt.ylabel('Nernst and cell voltages (V)')

    plt.tight_layout()

    # Dynamic current plot (optional)
    if I_vec is not None:
        plt.figure()
        plt.plot(ts, I_vec)
        plt.xlabel('t (s)')
        plt.ylabel('Current load (A)')

    # Plot mass of Sp
    plt.figure()
    plt.plot(ts, w[4, :])
    plt.xlabel('t (s)')
    plt.ylabel('Mass of Sp (g)')

    # Plot cell voltage
    plt.figure()
    plt.plot(ts, w[7, :])
    plt.xlabel('t (s)')
    plt.ylabel('Cell voltage (V)')

    plt.show()
