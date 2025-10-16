from scipy.sparse import diags
import numpy as np
from scipy_dae.integrate import solve_dae

def F(t, y, ydash):
    """
    Define implicit system
    Dimensions (n, m, d) for
        n: u
        m: mu, q
        d: tau
    """

    
    # Extract variables
    mu, nu, u, q, tau = y[:m], y[m:m+d], y[m+d:m+d+n], y[m+d+n:m+d+n+m], y[m+d+n+m:m+d+n+m+d]
    mup, nup, _, _, _ = ydash[:m], ydash[m:m+d], ydash[m+d:m+d+n], ydash[m+d+n:m+d+n+m], ydash[m+d+n+m:m+d+n+m+d]
    
    # Helpers: calculate weights
    W1 = diags(mu / ell)
    W2 = diags(nu / a)  
    W1_inv = diags(ell/mu)
    Lw = B1 * W1 * B1_T

    # Set up F(t, y, y')
    F0 = mup - (-np.abs(q)**beta + mu) #mudash
    F1 = nup - (-np.abs(tau)**beta + nu) #nudash
    F7 = Lw @ u - f0 #u
    F8 = q - (W1 @ B1_T @ W0 @ u) - (W1_inv @ B2 @ W2 @ tau ) #q
    F9 = tau - (W2 @ B2_T @ W1_inv @ q) #tau

    
    F = np.hstack([F0, F1, 
                   F7, F8, F9], dtype=np.common_type(y, ydash))

    return F

t0 = 0
t1 = 1e3
t_span = (t0, t1)
t_eval = np.logspace(-6, 3, num=100)

# solver options
method = "Radau"
# method = "BDF" # alternative solver
atol = rtol = 1e-3

# solve DAE system
sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)