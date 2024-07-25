# scipy_dae - solving differential algebraic equations (DAE's) and implicit differential equations (IDE's) in Python

<p align="center">
<a href="https://github.com/JonasBreuling/scipy_dae/actions/workflows/main.yml/badge.svg"><img alt="Actions Status" src="https://github.com/JonasBreuling/scipy_dae/actions/workflows/main.yml/badge.svg"></a>
<a href="https://codecov.io/gh/JonasBreuling/scipy_dae/branch/main">
<img src="https://codecov.io/gh/JonasBreuling/scipy_dae/branch/main/graph/badge.svg" alt="Code coverage status badge">
</a>
<a href="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"><img alt="License: BSD 3" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"></a>
<a href="https://pypi.org/project/scipy_dae/"><img alt="PyPI" src="https://img.shields.io/pypi/v/scipy_dae"></a>
</p>

Python implementation of solvers for differential algebraic equations (DAE's) and implicit differential equations (IDE's) that should be added to scipy one day. See the [GitHub repository](https://github.com/JonasBreuling/scipy_dae).

Currently, two different methods are implemented.

* Implicit **Radau IIA** methods of order 2s - 1 with arbitrary number of odd stages.
* Implicit **backward differentiation formula (BDF)** of variable order with quasi-constant step-size and stability/ accuracy enhancement using numerical differentiation formula (NDF).

More information about both methods are given in the specific class documentation.

## To pique your curiosity

The [Kármán vortex street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street) solved by a finite element discretization of the [weak form of the incompressible Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Weak_form) using [FEniCS](https://fenicsproject.org/) and the three stage Radau IIA method.

![Karman](https://raw.githubusercontent.com/JonasBreuling/scipy_dae/main/data/img/von_Karman.gif)

## Basic usage

The Robertson problem of semi-stable chemical reaction is a simple system of differential algebraic equations of index 1. It demonstrates the basic usage of the package.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae


def F(t, y, yp):
    """Define implicit system of differential algebraic equations."""
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=y.dtype)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1 # algebraic equation

    return F


# time span
t0 = 0
t1 = 1e7
t_span = (t0, t1)
t_eval = np.logspace(-6, 7, num=1000)

# initial conditions
y0 = np.array([1, 0, 0], dtype=float)
yp0 = np.array([-0.04, 0.04, 0], dtype=float)

# solver options
method = "Radau"
# method = "BDF" # alternative solver
atol = rtol = 1e-6

# solve DAE system
sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
t = sol.t
y = sol.y

# visualization
fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.plot(t, y[0], label="y1")
ax.plot(t, y[1] * 1e4, label="y2 * 1e4")
ax.plot(t, y[2], label="y3")
ax.set_xscale("log")
ax.legend()
ax.grid()
plt.show()
```

![Robertson](https://raw.githubusercontent.com/JonasBreuling/scipy_dae/main/data/img/Robertson.png)

## Advanced usage

More examples are given in the [examples](examples/) directory, which includes

* ordinary differential equations (ODE's)
    * [Van der Pol oscillator](examples/van_der_pol.py)
    * [Sparse brusselator](examples/sparse_brusselator.py)
* differential algebraic equations (DAE's)
    * [Robertson problem (index 1)](examples/robertson.py)
    * [Stiff transistor amplifier (index 1)](examples/stiff_transistor_amplifier.py)
    * [Brenan's problem (index 1)](examples/brenan1996.py)
    * [Jay's probem (index 2)](examples/jay1993.py)
    * [Cartesian pendulum (index 3)](examples/pendulum.py)
    * [Particle on circular track (index 3)](examples/particle_on_circular_track.py)
* implicit differential equations (IDE's)
    * [Weissinger's implicit equation](examples/weissinger.py)

## Install

An editable developer mode can be installed via

```bash
python -m pip install -e .[dev]
```

The tests can be started using

```bash
python -m pytest --cov
```
