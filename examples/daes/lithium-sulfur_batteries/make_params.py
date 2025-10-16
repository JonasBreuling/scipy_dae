from scipy._lib._util import _RichResult

def make_params(temp=298, R_area=0.96, ppt=True, BV=True):
    """
    Assigns parameter values for the model. Allows default values to be changed when specified in the function call.

    Args:
        temp (float): Temperature in Kelvin (default: 298 K).
        R_area (float): Reaction area in m^2 (default: 0.96 m^2).
        ppt (bool): Precipitation switch (default: True).
        BV (bool): Butler-Volmer switch (default: True).

    Returns:
        dict: A dictionary containing all parameters.
    """
    if not isinstance(temp, (int, float)) or temp <= 0:
        raise ValueError("temp must be a positive scalar numeric value.")
    if not isinstance(R_area, (int, float)) or R_area <= 0:
        raise ValueError("R_area must be a positive scalar numeric value.")
    if not isinstance(ppt, bool):
        raise ValueError("ppt must be a boolean value.")
    if not isinstance(BV, bool):
        raise ValueError("BV must be a boolean value.")

    # Physical constants
    param = {
        "R": 8.314462175,  # J mol^-1 K^-1 universal gas constant
        "F": 9.64853399e4,  # C mol^-1 Faraday constant
        "NA": 6.0221412927e23,  # mol^-1 Avogadro's number
        "e": 1.60217656535e-19,  # C electron charge
        "T": temp,  # Temperature in Kelvin
    }

    # Cell-specific parameters
    param.update({
        "mm": 32,  # g/mol sulfur molar mass
        "EL0": 2.195,  # V reference potential for low plateau reaction
        "EH0": 2.35,  # V reference potential for high plateau reaction
    })

    # Parameters related to charge transfer reactions (BV)
    param["ar"] = R_area  # Reaction area in m^2

    if BV:
        param["iHr"] = 1 * R_area  # A/m^2 * m^2
        param["iLr"] = 0.5 * R_area  # A/m^2 * m^2
    else:
        # Large values such that overpotentials are very small for fixed reaction current
        param["iHr"] = R_area * 1e6  # A/m^2 * m^2
        param["iLr"] = R_area * 1e6  # A/m^2 * m^2

    # Parameters related to precipitation
    param.update({
        "rhoS": 2e6,  # g/m^3 density of precipitated sulfur
        "vol": 0.0114e-3,  # m^3 cell volume
        "Ksp": 0.0001,  # g saturation mass of sulfur in electrolyte
    })

    param["kp"] = 100 if ppt else 0  # Precipitation/dissolution rate

    # Shuttle rate
    param["ks"] = 0.0002  # s^-1

    return _RichResult(**param)
