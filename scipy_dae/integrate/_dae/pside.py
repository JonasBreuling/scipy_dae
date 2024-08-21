import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.linalg import eig, cdf2rdf
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from .base import DAEDenseOutput
from .dae import DaeSolver


C = np.array([
    8.85879595127039430e-02,
    4.09466864440734710e-01,
    7.87659461760847110e-01,
    1.0,
])

A = np.array([
    [1.12999479323156150e-01, -4.03092207235221240e-02, 2.58023774203363160e-02, -9.90467650726640190e-03],
    [2.34383995747400210e-01, 2.06892573935359120e-01, -4.78571280485409760e-02, 1.60474228065163730e-02],
    [2.16681784623249830e-01, 4.06123263867374850e-01, 1.89036518170054850e-01, -2.41821048998323020e-02],
    [2.20462211176767560e-01, 3.88193468843174740e-01, 3.28844319980056810e-01, 6.25000000000008880e-02],
])

D = np.array([
    1.52077368976571730e-01,
    1.98631665602052860e-01,
    1.73704821245586140e-01,
    2.26879766524847200e-01,
])

B = np.array([
    [-3.36398745680207070e+00, -4.46547007540097850e-01, 0.0, 0.0],
    [2.53420388412422460e+01, 3.36398745680206800e+00, 0.0, 0.0],
    [0.0, 0.0, -4.37367276825310510e-01, -5.80576031184040710e-02],
    [0.0, 0.0, 3.29483348541735400e+00, 4.37367276825312510e-01],
])

Q = np.array([
    [2.95256291376448490e+00, 3.15939676526544260e-01, 1.53250361857358990e+00, 2.76001773070919800e-02],
    [-7.26638442522609210e+00, -8.75577120872169210e-01, -1.05525925554083820e+00, -3.11277680445624430e-01],
    [3.42060134189704890e+00, 9.49331950091266920e-01, -1.07997190626525940e+01, -2.13491394363750460e+00],
    [3.48973092842014550e+01, 4.37528029636525950e+00, -4.29039265780423160e+01, -5.89600020104458620e+00],
])

QI = np.array([
    [4.94042191453623210e-01, 2.69406327540352320e-01, -2.07753732935469560e-01, 6.33161132809504090e-02],
    [-3.53358212942913720e+00, -2.98582140519829590e+00, 1.75646748288234010e+00, -4.94914306872105080e-01],
    [4.87641455081039950e-01, 1.23938205146711190e-01, 4.23770339324015460e-02, -1.96050751500893220e-02],
    [-3.24650638473406870e+00, -1.52301305545598620e+00, -2.34591215977400570e-01, -1.94525303087971780e-02],
])

B0 = 1e-2

V = np.array([
    1.57753763977411530e-02,
    -9.73676595200762000e-03,
    6.46138955426500680e-03,
    2.24379766524848060e-01,
])

Lambda = np.array([
    [0.15207736897658, 0.06790969403105, 0, 0],
    [-0.35070903457864, 0.04202359569373, 0, 0],
    [0, 0, 0.17370482124555, 0.01008488557162],
    [0, 0, -0.40058458777036, 0.20362278551270],
])

L = np.array([
    [0.15207736897658, 0, 0, 0],
    [-0.35070903457864, 0.19863166560206, 0, 0],
    [0, 0, 0.17370482124555, 0],
    [0, 0, -0.40058458777036, 0.22687976652481],
])

S = np.array([
    [1, 0, 0, 0],
    [7.53333333333333, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 7.53333333333333, 1],
])

T = np.array([
    [0.57247400465791, 0.31594239005361, 1.32458228286171, 0.02760017730665],
    [-0.67033600112323, -0.87557678542461, 1.28969927047784, -0.31127768044595],
    [-3.73110064036903, 0.94929336342678, 5.28329931272012, -2.13491394363799],
    [1.93668410197760, 4.37526650476817, 1.51260826973776, -5.89600020104167],
])

TI = np.linalg.inv(T)

UI = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [-0.6133376973486749e0, 2.700531288824063e0, -9.587193591475394e0, 7.5e0],
    [-3.927079477247392e0, 15.68094527819332e0, -26.75386580094595e0, 15e0],                  
    [-4.891279419672913e0, 13.95409058457028e0, -17.81281116489738e0, 8.75e0],
])        

DAMPING_RATIO_ERROR_ESTIMATE = 0.8 # Hairer (8.19) is obtained by the choice 1.0. 
                                   # de Swart proposes 0.067 for s=3.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.
KAPPA = 1 # Factor of the smooth limiter
INNER_NEWTON_ITERATIONS = 1 # number of inner Newton iterations

def LUdecompCrout(A):
    """
    Perform Crout's LU decomposition on matrix A.

    Args:
    - A: numpy array, the matrix to decompose
    
    Returns:
    - L: numpy array, lower triangular matrix from Crout's decomposition
    - U: numpy array, upper triangular matrix from Crout's decomposition
    """
    # Get the number of rows and columns
    R, C = A.shape
    
    # Initialize L and U
    L = np.zeros((R, C))
    U = np.zeros((R, C))
    
    # First column initialization of L and diagonal of U
    for i in range(R):
        L[i, 0] = A[i, 0]
        U[i, i] = 1
    
    # First row initialization of U
    for j in range(1, R):
        U[0, j] = A[0, j] / L[0, 0]
    
    # Fill in L and U for the remaining elements
    for i in range(1, R):
        for j in range(1, i + 1):
            L[i, j] = A[i, j] - np.dot(L[i, :j], U[:j, j])
        
        for j in range(i + 1, R):
            U[i, j] = (A[i, j] - np.dot(L[i, :i], U[:i, j])) / L[i, i]
    
    return L, U


def LUdecompCrout(A):
    """
    Perform Crout's LU decomposition on matrix A.
    
    Args:
    - A: numpy array, the matrix to decompose
    
    Returns:
    - L: numpy array, lower triangular matrix from Crout's decomposition
    - U: numpy array, upper triangular matrix from Crout's decomposition
    """
    # Ensure A is a NumPy array
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Initialize L and U matrices
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    # Crout decomposition
    for j in range(n):
        # Compute L[:, j]
        for i in range(j, n):
            L[i, j] = A[i, j] - np.dot(L[i, :j], U[:j, j])

        # Compute U[j, :]
        U[j, j] = 1  # Diagonal of U is set to 1
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - np.dot(L[j, :j], U[:j, i])) / L[j, j]

    return L, U


def butcher_tableau(s):
    # nodes are given by the zeros of the Radau polynomial, see Hairer1999 (7)
    poly = Poly([0, 1]) ** (s - 1) * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # compute coefficients a_ij, see Hairer1999 (11)
    A = np.zeros((s, s))
    for i in range(s):
        Mi = np.zeros((s, s))
        ri = np.zeros(s)
        for q in range(s):
            Mi[q] = c**q
            ri[q] = c[i] ** (q + 1) / (q + 1)
        A[i] = np.linalg.solve(Mi, ri)

    b = A[-1, :]
    p = 2 * s - 1
    return A, b, c, p


def radau_constants(s):
    # Butcher tableau
    A, b, c, p = butcher_tableau(s)

    # inverse coefficient matrix
    A_inv = np.linalg.inv(A)

    # eigenvalues and corresponding eigenvectors of inverse coefficient matrix
    # lambdas, V = eig(A_inv)
    lambdas, V = eig(A)

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    V = V[:, idx]

    # # scale eigenvectors to get a "nice" transformation matrix (used by original 
    # # scipy code) and at the same time minimizes the condition number of V
    # for i in range(s):
    #     V[:, i] /= V[-1, i]

    # convert complex eigenvalues and eigenvectors to real eigenvalues 
    # in a block diagonal form and the associated real eigenvectors
    Gammas, T = cdf2rdf(lambdas, V)
    TI = np.linalg.inv(T)

    # check if everything worked
    assert np.allclose(V @ np.diag(lambdas) @ np.linalg.inv(V), A)
    assert np.allclose(np.linalg.inv(V) @ A @ V, np.diag(lambdas))
    assert np.allclose(T @ Gammas @ TI, A)
    assert np.allclose(TI @ A @ T, Gammas)

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag
    
    # compute embedded method for error estimate
    c_hat = np.array([0, *c])
    vander = np.vander(c_hat, increasing=True).T

    rhs = 1 / np.arange(1, s + 1)
    # b_hats2 = 1 / gammas[0] # real eigenvalue of A, i.e., 1 / gamma[0]
    b_hats2 = 1 / D[-1]
    b_hat1 = DAMPING_RATIO_ERROR_ESTIMATE * b_hats2
    rhs[0] -= b_hat1
    rhs -= b_hats2

    b_hat = np.linalg.solve(vander[:-1, 1:], rhs)
    v = b - b_hat

    rhs2 = 1 / np.arange(1, s + 1)
    # rhs2[0] -= 1 / gammas[0] # Hairer (8.16)
    rhs2[0] -= 1 / D[-1] # Hairer (8.16)

    b_hat2 = np.linalg.solve(vander[:-1, 1:], rhs2)
    v2 = b_hat2 - b

    # Compute the inverse of the Vandermonde matrix to get the interpolation matrix P.
    P = np.linalg.inv(vander)[1:, 1:]

    # Compute coefficients using Vandermonde matrix.
    vander2 = np.vander(c, increasing=True)
    P2 = np.linalg.inv(vander2)

    return A, A_inv, c, T, TI, P, P2, b_hat1, v, v2, b_hat, b_hat2, p


def solve_collocation_system(fun, t, y, h, Yp0, scale, tol,
                             M, LUs, solve_lu, newton_max_iter, atol):
    s, n = Yp0.shape
    tau = t + h * C

    Yp = Yp0
    Y = y + h * A.dot(Yp)
    V_dot = QI.dot(Yp)

    F = np.empty((s, n))

    dY_norm_old = None
    converged = False
    rate = None
    for k in range(newton_max_iter):
        for i in range(s):
            F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        U = QI @ F

        # first inner iteration
        dV_dot = np.zeros_like(V_dot)
        for i in range(s):
            dV_dot[i] = solve_lu(LUs[i], -U[i])

        # other inner iterations
        for _ in range(1, INNER_NEWTON_ITERATIONS):
            # U += (np.kron(B, M) @ dV_dot.reshape(-1)).reshape(s, -1)
            # see https://en.wikipedia.org/wiki/Kronecker_product#Matrix_equations
            U += B.dot(dV_dot.dot(M.T))
            dV_dot_inner = np.zeros((s, n))
            for i in range(s):
                dV_dot_inner[i] = solve_lu(LUs[i], -U[i])
            dV_dot = B.dot(dV_dot) + dV_dot_inner

        dYp = Q.dot(dV_dot)
        dY = h * A.dot(dYp)

        Yp += dYp
        Y += dY

        ###############################
        # original convergence criteria
        ###############################
        dY_norm = norm(dY / scale)
        if dY_norm_old is not None:
            rate = dY_norm / dY_norm_old

        if (rate is not None and (rate >= 1 or rate ** (newton_max_iter - k) / (1 - rate) * dY_norm > tol)):
            break

        # if (rate is not None and rate >= 1.0):
        #     break

        # # TODO: Why this is a bad indicator for divergence of the iteration?
        # if (rate is not None and rate ** (newton_max_iter - k) / (1 - rate) * dY_norm > tol):
        #     break

        if (dY_norm == 0 or rate is not None and rate / (1 - rate) * dY_norm < tol):
            converged = True
            break

        dY_norm_old = dY_norm

        # ##############################
        # # pside.f convergence criteria
        # ##############################
        # dY_norm = norm(dY / scale)

        # growth = False
        # diver = False
        # slow = False
        # solved = False
        # exact = False

        # # parameters of pside.f
        # tau_ = 0.01
        # kappa = 100.0
        # kmax = 15
        # gamma = 1.0
        # theta = 0.5
        # gfac = 100.0
        # rate1 = 0.1

        # # check if solution grows to prevent overflow
        # # TODO: Why only check last stage?
        # growth = np.any(np.abs(Y[-1]) > gfac * np.maximum(np.abs(y), atol))
        # if growth:
        #     break

        # # compute rate of convergence and other conditions if not growing
        # rate = rate1
        # ready = growth
        # if not growth:
        #     if dY_norm_old is None:
        #         exact = (dY_norm == 0.0)
        #         solved = exact
        #     else:
        #         rate = rate ** theta * (dY_norm / dY_norm_old) ** (1 - theta)
        #         # print(f"rate: {rate}")
        #         if rate >= gamma:
        #             diver = True
        #         elif (dY_norm * rate < (1.0 - rate) * tau_ or 
        #             dY_norm < kappa * EPS * norm(y / scale)):
        #             solved = True
        #         elif (k == kmax or dY_norm * rate ** (kmax - k) > tau_ * (1.0 - rate)):
        #             slow = True

        # ready = growth or diver or slow or solved or exact
        # converged = solved or exact

        # if ready:
        #     break

        # dY_norm_old = dY_norm

    return converged, k + 1, Y, Yp, Y - y, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).
    s : int
        Number of stages of the Radau IIA method.

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / (s + 1))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

    #############################
    # pside.f step-size selection
    #############################

    # # constants
    # zeta = 0.8
    # pmin = 0.1
    # fmax = 2.0
    
    # # Local variables
    # h = h_abs
    # hp = h_abs_old
    # eps = error_norm
    # epsp = error_norm_old
    # first = error_norm_old is None or h_abs_old is None

    # # dsinv = 1.0 / D[-1]
    # # epsp, hreje, epsrej, sucrej = 0.0, 0.0, 0.0, False
    # sucrej = False
    # epsrej = 0.0
    # hreje = 0.0
        
    # # step acceptance logic
    # if eps < 1.0:
    #     if eps == 0.0:
    #         hr = fmax * h
    #     elif first or sucrej:
    #         first = False
    #         hr = zeta * h * eps ** (-0.2)
    #     else:
    #         hr = zeta * (h ** 2 / hp) * (epsp / eps ** 2) ** 0.2
        
    #     # accept step        
    #     hp = h
    #     epsp = eps
    #     sucrej = False
    #     jacu2d = False
    # else:
    #     # reject step and adjust step size
    #     if not first and sucrej:
    #         pest = min(5.0, max(pmin, np.log10(eps / epsrej) / np.log10(h / hreje)))
    #         hr = zeta * h * eps ** (-1.0 / pest)
    #     else:
    #         hr = zeta * h * eps ** (-0.2)
        
    #     hreje = h
    #     epsrej = eps
    #     sucrej = True
    #     # nreje += 1

    # factor = hr / h

    # # nonsmooth limiter
    # factor = max(MIN_FACTOR, min(factor, MAX_FACTOR))

    # smooth limiter
    factor = 1 + KAPPA * np.arctan((factor - 1) / KAPPA)

    return factor


def predic(neqn, dysp, h, hp):
    # Constants
    S = 4

    # Initialize matrices and vectors
    dys = np.zeros((neqn, S))
    vm = np.zeros((S, S))
    w = np.zeros((S, S))

    # Constants as given in the Fortran code
    C = np.array([8.85879595127039430e-02, 4.09466864440734710e-01, 7.87659461760847110e-01, 1.00000000000000000])
    A = np.array([[1.12999479323156150e-01, -4.03092207235221240e-02, 2.58023774203363160e-02, -9.90467650726640190e-03],
                  [2.34383995747400210e-01, 2.06892573935359120e-01, -4.78571280485409760e-02, 1.60474228065163730e-02],
                  [2.16681784623249830e-01, 4.06123263867374850e-01, 1.89036518170054850e-01, -2.41821048998323020e-02],
                  [2.20462211176767560e-01, 3.88193468843174740e-01, 3.28844319980056810e-01, 6.25000000000008880e-02]])

    D = np.array([1.52077368976571730e-01, 1.98631665602052860e-01, 1.73704821245586140e-01, 2.26879766524847200e-01])
    B = np.array([[-3.36398745680207070e+00, -4.46547007540097850e-01, 0.0, 0.0],
                  [2.53420388412422460e+01, 3.36398745680206800e+00, 0.0, 0.0],
                  [0.0, 0.0, -4.37367276825310510e-01, -5.80576031184040710e-02],
                  [0.0, 0.0, 3.29483348541735400e+00, 4.37367276825312510e-01]])

    Q = np.array([[2.95256291376448490e+00, 3.15939676526544260e-01, 1.53250361857358990e+00, 2.76001773070919800e-02],
                  [-7.26638442522609210e+00, -8.75577120872169210e-01, -1.05525925554083820e+00, -3.11277680445624430e-01],
                  [3.42060134189704890e+00, 9.49331950091266920e-01, -1.07997190626525940e+01, -2.13491394363750460e+00],
                  [3.48973092842014550e+01, 4.37528029636525950e+00, -4.29039265780423160e+01, -5.89600020104458620e+00]])

    QINV = np.array([[4.94042191453623210e-01, 2.69406327540352320e-01, -2.07753732935469560e-01, 6.33161132809504090e-02],
                     [-3.53358212942913720e+00, -2.98582140519829590e+00, 1.75646748288234010e+00, -4.94914306872105080e-01],
                     [4.87641455081039950e-01, 1.23938205146711190e-01, 4.23770339324015460e-02, -1.96050751500893220e-02],
                     [-3.24650638473406870e+00, -1.52301305545598620e+00, -2.34591215977400570e-01, -1.94525303087971780e-02]])

    B0 = 1.00000000000000000e-02
    V = np.array([1.57753763977411530e-02, -9.73676595200762000e-03, 6.46138955426500680e-03, 2.24379766524848060e-01])

    UINV = np.array([[0.0, 0.0, 0.0, 1.0],
                     [-0.6133376973486749, 2.700531288824063, -9.587193591475394, 7.5],
                     [-3.927079477247392, 15.68094527819332, -26.75386580094595, 15.0],
                     [-4.891279419672913, 13.95409058457028, -17.81281116489738, 8.75]])

    # Calculate RH
    rh = h / hp

    # Calculate VM matrix
    for ks in range(S):
        for ks2 in range(S):
            vm[ks2, ks] = (rh * C[ks2]) ** ks

    # Calculate W matrix
    for ks in range(S):
        for ks2 in range(S):
            w[ks2, ks] = 0.0
        for ks3 in range(S):
            for ks2 in range(S):
                w[ks2, ks] += vm[ks2, ks3] * UINV[ks3, ks]

    # Compute DYS matrix
    for ks in range(S):
        for kn in range(neqn):
            dys[kn, ks] = dysp[kn, S - 1]
        for ks2 in range(S):
            for kn in range(neqn):
                dys[kn, ks] += w[ks, ks2] * (dysp[kn, ks2] - dysp[kn, S - 1])

    return dys


def ctrl(ldlu, neqn, y, dy, t, tend, geval, jbnd, nlj, nuj, hp, h, hlu, tolvect, rtol, atol,
         indgt1, ind, uround, ierr, rpar, ipar, nrejn, nreje, nrejg, nreji, nf, nfb, 
         ys, dys, dysp, declus, ipvts, aux, aux2, alpha, growth, diver, slow, solved, 
         exact, first, jacnew, facnew, jacu2d, idid):

    # Constants
    alpref = 0.15
    alpjac = 0.1
    alplu = 0.2
    fmin = 0.2
    fmax = 2.0
    frig = 2.0
    xi = 1.2
    omega = 0.05
    
    jacnew = False
    
    if not growth:
        halpha = h * alpref / max(alpha, alpref / fmax)

    if ierr == -1:
        hnew = h / frig
        nreji += 1

    elif solved:
        error(ldlu, neqn, y, dy, t, tend, geval, jbnd, nlj, nuj, hp, h, hlu, tolvect, rtol, atol,
              indgt1, ind, uround, ierr, rpar, ipar, nreje, nf, nfb, ys, dys, dysp, declus, ipvts, 
              aux, aux2, first, jacu2d)
        
        if ierr == -1:
            hnew = h / frig
            nreji += 1

        elif abs(tend - t) > 10 * uround * abs(t):
            if jacu2d and alpha > alpref:
                hnew = min(fmax * h, max(fmin * h, min(hlu, halpha)))
            else:
                hnew = min(fmax * h, max(fmin * h, hlu))
            
            if not exact and alpha - abs(h - hlu) / hlu > alpjac:
                if jacu2d:
                    hnew = h / frig
                else:
                    jacnew = True
    
    elif growth:
        hnew = h / frig
        nrejg += 1

    elif diver:
        hnew = min(fmax * h, max(fmin * h, halpha))
        jacnew = not jacu2d
        nrejn += 1

    elif slow:
        if jacu2d:
            if alpha > xi * alpref:
                hnew = min(fmax * h, max(fmin * h, halpha))
            else:
                hnew = h / frig
        else:
            hnew = h
            jacnew = True
        nrejn += 1

    else:
        print('PSIDE: ERROR: impossible(?) error in CTRL')
        raise RuntimeError('Impossible(?) error in CTRL')
    
    if abs(tend - t) > 10 * uround * abs(t):
        if hnew < 10 * uround * abs(t):
            idid = -1
            return

        remain = (tend - t) / hnew
        if remain - np.floor(remain) > omega or np.floor(remain) == 0.0:
            remain = np.floor(remain) + 1.0
        else:
            remain = np.floor(remain)
        
        h = (tend - t) / remain
    
    facnew = jacnew or abs(h - hlu) > alplu * hlu
    if facnew:
        hlu = h
    
    return h, hlu, hp, jacnew, facnew, idid


def snorms(y, uys, h, tolvec, rtol, atol, ind):
    # Placeholder for the actual implementation of snorms (since it's not provided)
    # Add the correct implementation here
    return np.linalg.norm(uys)  # A simple norm as an example


def vergen(y, Y, dY, h, k, tolvec, rtol, atol, ind, uround):
    n = len(y)

    # parameters of pside.f
    tau = 0.01
    kappa = 100.0
    kmax = 15
    gamma = 1.0
    theta = 0.5
    gfac = 100.0
    alpha1 = 0.1
    
    # initialize states
    growth = False # monitors whether the current iterate is too large with respect to y (necessary to prevent overflow)
    diver = False # indicates divergence
    slow = False # Newton process that is converging too slow
    solved = False
    exact = False
    
    # compute GROWTH condition
    growth = False
    growth = np.any(np.abs(Y) > gfac * np.max(np.abs(y), atol))
    # for kn in range(n):
    #     if abs(Y[kn, -1]) > gfac * max(abs(y[kn]), atol[0]):
    #         growth = True
    #         break

    # Compute ALPHA and other conditions if not growing
    alpha = alpha1
    ready = growth
    if not growth:
        if k == 1:
            u = snorms(n, y, dY, h, tolvec, rtol, atol, indgt1, ind)
            exact = (u == 0.0)
            solved = exact
        elif k > 1:
            up = u
            u = snorms(n, y, dY, h, tolvec, rtol, atol, indgt1, ind)
            alpha = alpha ** theta * (u / up) ** (1 - theta)
            if alpha >= gamma:
                diver = True
            elif (u * alpha < (1.0 - alpha) * tau or 
                  u < kappa * uround * snorms(n, y, y, h, tolvec, rtol, atol, indgt1, ind)):
                solved = True
            elif (k == kmax or u * alpha ** (kmax - k) > tau * (1.0 - alpha)):
                slow = True

    # Determine if the process is ready
    ready = growth or diver or slow or solved or exact
    return ready, growth, diver, slow, solved, exact, alpha


# TODO: Incorporate acceptance logic from here to predict_factor
def error(ldlu, neqn, y, dy, t, tend, geval, jbnd, nlj, nuj, hp, h, hr, tolvec, rtol, atol,
          indgt1, ind, uround, ierr, rpar, ipar, nreje, nf, nfb, ys, dys, dysp, declus, ipvts, aux, r, first, jacu2d):
    # Constants
    S = 4
    zeta = 0.8
    pmin = 0.1
    fmax = 2.0
    b0 = 1e-2
    
    # Coefficients
    c = [0.0885879595127039430, 0.40946686444073471, 0.78765946176084711, 1.0]
    d = [0.15207736897657173, 0.19863166560205286, 0.17370482124558614, 0.2268797665248472]
    v = [0.015775376397741153, -0.00973676595200762, 0.0064613895542650068, 0.22437976652484806]
    
    a = np.array([
        [0.11299947932315615, -0.040309220723522124, 0.025802377420336316, -0.009904676507266402],
        [0.23438399574740021, 0.20689257393535912, -0.047857128048540976, 0.016047422806516373],
        [0.21668178462324983, 0.40612326386737485, 0.18903651817005485, -0.024182104899832302],
        [0.22046221117676756, 0.38819346884317474, 0.3288443199800568, 0.0625]
    ])
    
    b = np.array([
        [-3.3639874568020707, -0.44654700754009785, 0.0, 0.0],
        [25.342038841242246, 3.363987456802068, 0.0, 0.0],
        [0.0, 0.0, -0.43736727682531051, -0.058057603118404071],
        [0.0, 0.0, 3.294833485417354, 0.43736727682531251]
    ])
    
    q = np.array([
        [2.9525629137644849, 0.31593967652654426, 1.5325036185735899, 0.02760017730709198],
        [-7.2663844252260921, -0.87557712087216921, -1.0552592555408382, -0.31127768044562443],
        [3.4206013418970489, 0.94933195009126692, -10.799719062652594, -2.1349139436375046],
        [34.897309284201455, 4.3752802963652595, -42.903926578042316, -5.8960002010445862]
    ])
    
    qinv = np.array([
        [0.49404219145362321, 0.26940632754035232, -0.20775373293546956, 0.063316113280950409],
        [-3.5335821294291372, -2.9858214051982959, 1.7564674828823401, -0.49491430687210508],
        [0.48764145508103995, 0.12393820514671119, 0.042377033932401546, -0.019605075150089322],
        [-3.2465063847340687, -1.5230130554559862, -0.23459121597740057, -0.019452530308797178]
    ])
    
    # Local variables
    dsinv = 1.0 / d[S-1]
    epsp, hreje, epsrej, sucrej = 0.0, 0.0, 0.0, False
    
    # Compute AUX
    aux[:] = -b0 * dy
    for ks in range(S):
        aux[:] += v[ks] * dys[:, ks]
    aux[:] *= dsinv
    
    # Call to GEVAL
    ierr = 0
    r = geval(neqn, t + h, ys[:, S-1], aux, r, ierr, rpar, ipar)
    nf += 1
    
    if ierr == -1:
        return
    
    # Solve the system using the appropriate LAPACK routines
    if jbnd:
        r = solve_band_system(neqn, nlj, nuj, declus[:, :, S-1], ldlu, ipvts[:, S-1], r)
    else:
        r = solve_system(neqn, declus[:, :, S-1], ldlu, ipvts[:, S-1], r)
    
    nfb += 1
    
    # Scale result
    r[:] = -h * d[S-1] * r
    
    # Compute the norm of the error
    eps = snorm(neqn, y, r, h, tolvec, rtol, atol, indgt1, ind)
    
    # Step acceptance logic
    if eps < 1.0:
        if eps == 0.0:
            hr = fmax * h
        elif first or sucrej:
            first = False
            hr = zeta * h * eps ** (-0.2)
        else:
            hr = zeta * (h ** 2 / hp) * (epsp / eps ** 2) ** 0.2
        
        # Accept step
        y[:] = ys[:, S-1]
        dy[:] = dys[:, S-1]
        dysp[:] = dys.copy()
        t += h
        
        if abs(tend - t) < 10.0 * uround * abs(t):
            t = tend
        
        hp = h
        epsp = eps
        sucrej = False
        jacu2d = False
    else:
        # Reject step and adjust step size
        if not first and sucrej:
            pest = min(5.0, max(pmin, np.log10(eps / epsrej) / np.log10(h / hreje)))
            hr = zeta * h * eps ** (-1.0 / pest)
        else:
            hr = zeta * h * eps ** (-0.2)
        
        hreje = h
        epsrej = eps
        sucrej = True
        nreje += 1


class PPSIDEDAE(DaeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 2s - 1.

    The implementation follows [4]_, where most of the ideas come from [2]_. 
    The embedded formula of [3]_ is applied to implicit differential equations.
    The error is controlled with a (s)th-order accurate embedded formula as 
    introduced in [2]_ and refined in [3]_. The procedure is slightly adapted 
    by [4]_ to cope with implicit differential equations. The embedded error 
    estimate can be mixed with the contunous error of the lower order 
    collocation polynomial as porposed in [5]_ and [6]_.
    
    A cubic polynomial 
    which satisfies the collocation conditions is used for the dense output of 
    both state and derivatives.

    Parameters
    ----------
    fun : callable
        Function defining the DAE system: ``f(t, y, yp) = 0``. The calling 
        signature is ``fun(t, y, yp)``, where ``t`` is a scalar and 
        ``y, yp`` are ndarrays with 
        ``len(y) = len(yp) = len(y0) = len(yp0)``. ``fun`` must return 
        an array of the same shape as ``y, yp``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    yp0 : array_like, shape (n,)
        Initial derivative.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    stages : int, optional
        Number of used stages. Default is 3, which corresponds to the 
        ``solve_ivp`` method. Only odd number of stages are allowed.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    newton_max_iter : int or None, optional
        Number of allowed (simplified) Newton iterations. Default is ``None`` 
        which uses ``newton_max_iter = 7 + (stages - 3) * 2`` as done in
        Hairer's radaup.f code.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    continuous_error_weight : float, optional
        Weighting of continuous error of the dense output as introduced in 
        [5]_ and [6]_. The embedded error is weighted by (1 - continuous_error_weight). 
        Has to satisfy 0 <= continuous_error_weight <= 1. Default is 0.0, i.e., only 
        the embedded error is considered.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    # TODO: Adapt and test this.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` 
        and ``yp`` of shape ``(n,)``, where ``n = len(y0) = len(yp0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` and ``yp`` of 
        shape ``(n, k)``, where ``k`` is an integer. In this case, `fun` must 
        behave such that ``fun(t, y, yp)[:, i] == fun(t, y[:, i], yp[:, i])``.

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
        Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    yp : ndarray
        Current derivative.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    .. [3] J. de Swart, G. Söderlind, "On the construction of error estimators for 
           implicit Runge-Kutta methods", Journal of Computational and Applied 
           Mathematics, 86, pp. 347-358, 1997.
    .. [4] B. Fabien, "Analytical System Dynamics: Modeling and Simulation", 
           Sec. 5.3.5.
    .. [5] N. Guglielmi, E. Hairer, "Implementing Radau IIA Methods for Stiff 
           Delay Differential Equations", Computing 67, 1-12, 2001.
    .. [6] N. Guglielmi, "Open issues in devising software for the numerical 
           solution of implicit delay differential equations", Journal of 
           Computational and Applied Mathematics 185, 261-277, 2006.
    """
    def __init__(self, fun, t0, y0, yp0, t_bound,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, 
                 continuous_error_weight=0.0, jac=None, 
                 jac_sparsity=None, vectorized=False, 
                 first_step=None, newton_max_iter=None,
                 jac_recompute_rate=1e-3, newton_iter_embedded=1,
                 controller_deadband=(1.0, 1.2),
                 **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, first_step, max_step, vectorized, jac, jac_sparsity)

        self.stages = stages = 4
        (
            self.A, self.A_inv, self.C, self.T, self.TI, self.P, self.P2, 
            self.b0, self.v, self.v2, self.b_hat, self.b_hat2, self.order,
        ) = radau_constants(stages)
        
        self.h_abs_old = None
        self.error_norm_old = None

        # modify tolerances as in radau.f line 824ff and 920ff
        # TODO: This rescaling leads to a saturation of the convergence
        EXPMNS = (stages + 1) / (2 * stages)
        # print(f"atol: {atol}")
        # print(f"rtol: {rtol}")
        # rtol = 0.1 * rtol ** EXPMNS
        # quott = atol / rtol
        # atol = rtol * quott
        # print(f"atol: {atol}")
        # print(f"rtol: {rtol}")

        # newton tolerance as in radau.f line 1008ff
        EXPMI = 1 / EXPMNS
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** (EXPMI - 1)))
        # print(f"newton_tol: {self.newton_tol}")
        # print(f"10 * EPS / rtol: {10 * EPS / rtol}")
        # print(f"0.03")
        # print(f"rtol ** (EXPMI - 1): {rtol ** (EXPMI - 1)}")

        # maximum number of newton terations as in radaup.f line 234
        if newton_max_iter is None:
            newton_max_iter = 15
        
        assert isinstance(newton_max_iter, int)
        assert newton_max_iter >= 1
        self.newton_max_iter = newton_max_iter

        assert 0 <= continuous_error_weight <= 1
        self.continuous_error_weight = continuous_error_weight

        assert 0 < jac_recompute_rate < 1
        self.jac_recompute_rate = jac_recompute_rate

        assert 0 < controller_deadband[0] <= controller_deadband[1]
        self.controller_deadband = controller_deadband

        assert 0 <= newton_iter_embedded
        self.newton_iter_embedded = newton_iter_embedded

        self.sol = None
        self.current_jac = True
        self.LUs = None
        self.Z = None
        self.Y = None
        self.Yp = None

    def _jac_fac(self, h, t, y, yp, jac_new, fac_new):
        if jac_new:
            self.Jy, self.Jyp = self.jac(t, y, yp)
            self.current_jac = True
        if fac_new:
            self.LUs = [self.lu(self.Jyp + h * Di * self.Jy) for Di in self.D]

    def _newton(self, y, h):
        ready = False
        k = 0

        # TODO: predict Yp using extrapolation
        def predict():
            pass
        Yp = predict()

        Y = y + h * A.dot(Yp)

        # TODO: checks whether the Newton process converges
        def vergen():
            pass
        alpha, growth, diver, slow, solved, exact = vergen()

        while not ready:
            k += 1

            # TODO: Parallel Iterative Linear system Solver for Runge-Kutta methods
            def pilsrk():
                pass
            DYp = pilsrk()

            DY = h * A.dot(DYp)
            Y += DY
            Yp += DYp

            alpha, growth, diver, slow, solved, exact = vergen()

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp

        s = self.stages
        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol
        newton_tol = self.newton_tol
        newton_max_iter = self.newton_max_iter

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        Jy = self.Jy
        Jyp = self.Jyp
        LUs = self.LUs

        current_jac = self.current_jac
        jac = self.jac

        factor = None
        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h
            # print(f"t_new: {t_new}")

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Yp0 = np.zeros((s, y.shape[0]))
            else:
                Yp0 = self.sol(t + h * C)[1].T
                # Yp0 = np.zeros((s, y.shape[0]))
                # TODO: Check if this is the same. Yields slightly better results for robertson.
                # Yp0 = predic(self.n, self.Yp.T, self.direction * h, self.direction * self.h_abs_old).T
            scale = atol + np.abs(y) * rtol
            # scale = atol + np.abs(yp) * rtol

            converged = False
            while not converged:
                if LUs is None:
                    LUs = [self.lu(Jyp + h * Di * Jy) for Di in D]

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Yp0, scale, newton_tol,
                    Jyp, LUs, self.solve_lu, newton_max_iter, atol)

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp)
                    current_jac = True
                    LUs = None

            if not converged:
                h_abs *= 0.5
                LUs = None
                continue

            y_new = Y[-1]
            yp_new = Yp[-1]

            # TODO: Use explicit embedded method and only implicit one if error_norm > 1?

            # compute implicit embedded method with a single Newton iteration;
            # R(z) = b_hat1 / b_hats2 = DAMPING_RATIO_ERROR_ESTIMATE for z -> oo
            yp_hat_new = (V @ Yp - B0 * yp) / D[-1]
            F = self.fun(t_new, y_new, yp_hat_new)
            error = - h * D[-1] * self.solve_lu(LUs[-1], F)

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = norm(error / scale)

            safety = 0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + n_iter)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
                # h_abs *= max(MIN_FACTOR, safety * factor)
                h_abs *= safety * factor

                LUs = None
                rejected = True
            else:
                rejected = False
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = (
            jac is not None 
            and n_iter > 2 
            and 
            rate > self.jac_recompute_rate
        )

        # factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
        # factor *= safety
        if factor is None:
            factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
            factor *= safety

        # do not alter step-size in deadband
        if (
            # TODO: This first check yields lots of LU decompositions
            not recompute_jac 
            and 
            self.controller_deadband[0] <= factor <= self.controller_deadband[1]
        ):
            factor = 1
        else:
            LUs = None
        # if recompute_jac:
        #     LUs = None

        if recompute_jac:
            Jy, Jyp = self.jac(t_new, y_new, yp_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_oldold = self.h_abs_old
        self.h_abs_old = self.h_abs
        self.error_norm_oldold = error_norm_old
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y
        self.yp_old = yp

        self.t = t_new
        self.y = y_new
        self.yp = yp_new

        self.Z = Z
        self.Y = Y
        self.Yp = Yp

        self.LUs = LUs
        self.current_jac = current_jac
        self.Jy = Jy
        self.Jyp = Jyp

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message

    def _compute_dense_output(self):
        Q = np.dot(self.Z.T, self.P)
        h = self.t - self.t_old
        Yp = (self.A_inv / h) @ self.Z
        Zp = Yp - self.yp_old
        Qp = np.dot(Zp.T, self.P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q, self.yp_old, Qp)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DAEDenseOutput):
    def __init__(self, t_old, t, y_old, Q, yp_old, Qp):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.Qp = Qp
        self.order = Q.shape[1] - 1
        self.y_old = y_old
        self.yp_old = yp_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        x = np.atleast_1d(x)

        # factors for interpolation polynomial and its derivative
        c = np.arange(1, self.order + 2)[:, None]
        p = x**c
        dp = (c / self.h) * (x**(c - 1))

        # 1. compute derivative of interpolation polynomial for y
        y = np.dot(self.Q, p)
        y += self.y_old[:, None]
        yp = np.dot(self.Q, dp)

        # # 2. compute collocation polynomial for y and yp
        # y = np.dot(self.Q, p)
        # yp = np.dot(self.Qp, p)
        # y += self.y_old[:, None]
        # yp += self.yp_old[:, None]

        # # 3. compute both values by Horner's rule
        # y = np.zeros_like(y)
        # yp = np.zeros_like(y)
        # for i in range(self.order, -1, -1):
        #     y = self.Q[:, i][:, None] + y * x[None, :]
        #     yp = y + yp * x[None, :]
        # y = self.y_old[:, None] + y * x[None, :]
        # yp /= self.h

        if t.ndim == 0:
            y = np.squeeze(y)
            yp = np.squeeze(yp)

        return y, yp
