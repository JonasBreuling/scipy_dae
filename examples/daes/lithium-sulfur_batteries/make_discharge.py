import numpy as np
from scipy.optimize import fsolve

# Define the Python function
def make_ic_disch_12(params, I, Vin, S8in, Spin):
    """
    Calculate initial values for variables in the system, assuming discharge from completely charged.

    Args:
        params: Object with required constants (iHr, iLr, F, R, T, EH0, EL0).
        I: Initial current.
        Vin: Input voltage.
        S8in: Initial value for S8.
        Spin: Initial value for Sp.

    Returns:
        numpy array: Initial values w0 as an array.
    """
    # Initialize w0 with NaN values
    w0 = np.full(12, np.nan)
    
    # Assign known values
    w0[0] = S8in
    w0[4] = Spin
    w0[7] = Vin

    # Populate the rest of w0 with calculated values

    # Corresponding to discharge from fully charged
    w0[5] = I

    w0[8] = np.arcsinh(-w0[5] / (2 * params.iHr)) / (4 * params.F) * 2 * params.R * params.T
    w0[10] = Vin - w0[8]
    w0[1] = np.sqrt(0.7294 * S8in / np.exp((w0[10] - params.EH0) / params.R / params.T * 4 * params.F))

    w0[6] = I - w0[5]  # Here 0

    w0[9] = np.arcsinh(-w0[6] / (2 * params.iLr)) / (4 * params.F) * 2 * params.R * params.T  # Here 0
    w0[11] = Vin - w0[9]  # Here ENL = V

    Sintemp = np.cbrt(0.0665 * w0[1] / np.exp(4 * params.F / params.R / params.T * (w0[11] - params.EL0)))

    # Define the function for fsolve
    def myfun(x):
        return x**2 * (x + Spin) - Sintemp**3

    # Solve for x using fsolve
    x = fsolve(myfun, Sintemp)[0]

    w0[3] = x
    w0[2] = w0[3] + Spin

    # Check for NaN values in w0
    if np.isnan(w0).any():
        raise ValueError("Initialising w0 has failed")

    return w0
