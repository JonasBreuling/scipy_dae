import numpy as np
from scipy.linalg import eig, cdf2rdf


def RadauIIA(s):
    # solve zeros of Radau polynomial, see Hairer1999 (7)
    from numpy.polynomial import Polynomial as P

    poly = P([0, 1]) ** (s - 1) * P([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # compute coefficients a_ij, see Hairer1999 (11)
    A = np.zeros((s, s), dtype=float)
    for i in range(s):
        Mi = np.zeros((s, s), dtype=float)
        ri = np.zeros(s, dtype=float)
        for q in range(s):
            Mi[q] = c**q
            ri[q] = c[i] ** (q + 1) / (q + 1)
        A[i] = np.linalg.solve(Mi, ri)

    b = A[-1, :]
    p = 2 * s - 1
    return A, b, c, p


def radau_constants(s):
    # compute Butcher tableau
    A, b, c, order = RadauIIA(s)
    print(f"A:\n{A}")

    # compute inverse coefficient matrix
    # TODO: Get rid of this and invert the eigenvalues?
    A_inv = np.linalg.inv(A)
    print(f"A_inv:\n{A_inv}")

    # compute eigenvalues and corresponding eigenvectors
    lambdas, V = eig(A_inv)
    print(f"lambdas:\n{lambdas}")
    print(f"V:\n{V}")

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    # idx = np.argsort(np.abs(lambdas.imag))#[::-1]
    lambdas = lambdas[idx]
    V = V[:, idx]
    print(f"lambdas:\n{idx}")
    print(f"lambdas:\n{lambdas}")
    print(f"V:\n{V}")
    # exit()

    # scale eigenvectors to get a "nice" transformation matrix (used by scipy)
    # TODO: This seems to minimize the condition number of T and TI, but why?
    for i in range(s):
        V[:, i] /= V[-1, i]

    # # scale only the first two eigenvectors (used by Hairer)
    # V[:, 0] /= V[-1, 0]
    # V[:, 1] /= V[-1, 1]

    # # normalize eigenvectors
    # V /= np.linalg.norm(V, axis=0)

    print(f"V scaled:\n{V}")

    # convert complex eigenvalues and eigenvectors to real eigenvalues 
    # in a block diagonal form and the associated real eigenvectors
    lambdas_real, V_real = cdf2rdf(lambdas, V)
    print(f"lambdas_real:\n{lambdas_real}")
    print(f"Vr:\n{V_real}")

    # transform to get scipy's/ Hairer's ordering
    R = np.fliplr(np.eye(s))
    R = np.eye(s)
    Mus = R @ lambdas_real @ R.T
    print(f"R:\n{R}")
    print(f"Mus:\n{Mus}")

    T = V_real @ R.T
    TI = np.linalg.inv(T)
    print(f"T:\n{T}")

    if s == 3:
        # radau.py
        T_scipy = np.array([
            [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
            [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
            [1, 1, 0]])
        TI_scipy = np.array([
            [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
            [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
            [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
        print(f"T_scipy:\n{T_scipy}")

        # radau5.f or firk_tableaus.jl, 
        # see https://github.com/SciML/OrdinaryDiffEq.jl/blob/1e8716a6cc2c334ff96049ab781f0aef10fba328/src/tableaus/firk_tableaus.jl#L2
        T11=9.12323948708929427920e-02
        T12=-0.1412552950209542084e+00
        T13=-3.0029194105147424492e-02
        T21=0.24171793270710701896e+00
        T22=0.20412935229379993199e+00
        T23=0.38294211275726193779e+00
        T31=0.96604818261509293619e+00
        T32=1.0
        T33=0.0

        T_fortran_julia = np.array([
            [T11, T12, T13],
            [T21, T22, T23],
            [T31, T32, T33],
        ], dtype=float)
        TI_fortran_julia = np.linalg.inv(T_fortran_julia)
        print(f"T_fortran_julia:\n{T_fortran_julia}")


        print(f"cond(T_sicpy):   {np.linalg.cond(T_scipy)}")
        print(f"cond(T_fortran): {np.linalg.cond(T_fortran_julia)}")
    print(f"cond(T):         {np.linalg.cond(T)}")

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    n_real = len(real_idx)
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag
    print(f"gammas: {gammas}")
    print(f"alphas: {alphas}")
    print(f"betas: {betas}")
    if s == 3:
        gamma_scipy = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
        alpha_scipy = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
        beta_scipy = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
        print(f"gamma_scipy: {gamma_scipy}")
        print(f"alpha_scipy: {alpha_scipy}")
        print(f"beta_scipy: {beta_scipy}")

    # compute embedded method for error estimate,
    # see https://arxiv.org/abs/1306.2392 equation (10)
    # # TODO: This is not correct yet, see also here: https://math.stackexchange.com/questions/4441391/embedding-methods-into-implicit-runge-kutta-schemes
    # TODO: This is correct, document this extended tableau!
    c_hat = np.array([0, *c])
    print(f"c_hat: {c_hat}")

    vander = np.vander(c_hat, increasing=True).T
    print(f"vander:\n{vander}")
    vander = vander[1:, 1:]
    print(f"vander:\n{vander}")

    # de Swart1997
    vander = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            vander[i, j] = c[j]**i
    print(f"vander:\n{vander}")

    rhs = 1 / np.arange(1, s + 1)
    # rhs = 1 / np.arange(2, s + 2)
    # rhs = 1 / np.arange(1, s + 2)
    # if s > 1:
    #     # TODO: Why is this required???
    #     rhs = 1 / np.arange(2, s + 2)
    # else:
    #     rhs = 1 / np.arange(1, s + 1)
    print(f"rhs:\n{rhs}")

    # gamma0 = gammas[0]
    gamma0 = 1 / gammas[0]
    rhs[0] -= gamma0
    print(f"rhs:\n{rhs}")

    b_hat = np.linalg.solve(vander, rhs)
    print(f"b_hat:\n{b_hat}")

    E = (b_hat - b) @ A_inv
    E /= gamma0 # TODO: Why is this done in scipy?

    print(f"c: {c}")
    print(f"E:\n{E}")

    # from nodepy.runge_kutta_method import RungeKuttaMethod

    # A_hat = np.zeros((s + 1, s + 1))
    # A_hat[1:, 1:] = A
    # b_hat = np.zeros(s + 1)
    # b_hat[0] = gamma0
    # b_hat[1:] = np.linalg.solve(vander, rhs)

    # A = A_hat
    # b = A[-1, :]

    # method = RungeKuttaMethod(A, b)
    # embedded_method = RungeKuttaMethod(A_hat, b_hat)

    # print(f"order:            {method.order()}")
    # print(f"order (embedded): {embedded_method.order()}")

    # print(f"stage order:            {method.stage_order()}")
    # print(f"stage order (embedded): {embedded_method.stage_order()}")

    # print(f"zero-stable:            {method.is_zero_stable()}")
    # print(f"zero-stable (embedded): {embedded_method.is_zero_stable()}")

    # left = -40
    # method.plot_stability_function([left, 0])
    # embedded_method.plot_stability_function([left, 0])

    # import matplotlib.pyplot as plt
    # plt.show()

    if s in [1, 3, 5, 7, 9]:

        if s == 1:
            E_scipy = np.array([
                -1.0e0,
            ])
        elif s == 3:
            S6 = 6 ** 0.5
            E_scipy = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
        elif s == 5:
            E_scipy = np.array([
                -0.2778093394406463730479e+02,
                 0.3641478498049213152712e+01,
                -0.1252547721169118720491e+01,
                 0.5920031671845428725662e+00,
                -0.2000000000000000000000e+00,
            ])
        elif s == 7:
            E_scipy = np.array([
                -0.5437443689412861451458e+02,
                 0.7000024004259186512041e+01,
                -0.2355661091987557192256e+01,
                 0.1132289066106134386384e+01,
                -0.6468913267673587118673e+00,
                 0.3875333853753523774248e+00,
                -0.1428571428571428571429e+00,
            ])
        elif s == 9:
            E_scipy = np.array([
                -89.831539704037684586502729e+00,
                 11.474276609468772159022261e+00,
                -3.81419058476042873698615187e+00,
                 1.81155300867853110911564243e+00,
                -1.03663781378817415276482837e+00,
                 0.66086568819371648375769004e+00,
                -0.44418925628052673008702343e+00,
                 0.29097316363690556555625116e+00,
                -0.11111111111111111111111111e+00,
            ])
        
        print(f"E_scipy:\n{E_scipy}")
        print(f"E / E_scipy: {E / E_scipy}")

    # TODO Compute collocation weights?
    # exit()

    return alphas, betas, gammas, T, TI, c, E


if __name__ == "__main__":
    # s = 1
    s = 3
    # s = 5
    # s = 7
    # s = 9
    radau_constants(s)
    exit()

# gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
# beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))

# s = 3
# A, b, c, p = RadauIIA(s)
# print(f"A:\n{A}")

# A_inv = np.linalg.inv(A)
# # TODO: We can decompose A and compute the correct matrix as the block 
# # diagonal inverse of Mu_real!
# # A_inv = A
# print(f"A_inv:\n{A_inv}")
# print(f"A_inv @ A:\n{A_inv @ A}")
# # exit()

# H, Q = hessenberg(A_inv, calc_q=True)
# # TODO: hessenberg form seems to be not necessary!
# H, Q = A_inv, np.eye(s)
# print(f"H:\n{H}")
# print(f"Q:\n{Q}")
# print(f"A_inv = Q @ H @ Q.T:\n{Q @ H @ Q.T}")
# print(f"A_inv - Q @ H @ Q.T:\n{A_inv - Q @ H @ Q.T}")
# print(f"Q.T @ Q:\n{Q.T @ Q}")
# # exit()

# lambdas, V = eig(H)

# # sort eigenvalues
# # print(np.sort(lambdas))
# idx = np.argsort(lambdas)[::-1]
# # print(lambdas[idx])

# # permut eigenvalues and eigenvectors accordingly
# lambdas = lambdas[idx]
# V = V[:, idx]

# # scale eigenvectors to get a "nice" transformation matrix (used by scipy)
# # TODO: This seems to minimize the condition number of T and TI, but why?
# for i in range(s):
#     V[:, i] /= V[-1, i]

# # # scale only the first two eigenvectors (used by Hairer)
# # V[:, 0] /= V[-1, 0]
# # V[:, 1] /= V[-1, 1]

# # # normalize eigenvectors
# # V /= np.linalg.norm(V, axis=0)

# V_inv = np.linalg.inv(V)
# Lambdas = np.diag(lambdas)
# print(f"Lambdas:\n{Lambdas}")
# print(f"V:\n{V}")
# H_reconstructed = V @ Lambdas @ V_inv
# H_reconstructed = H_reconstructed.real # prune zero imaginary parts
# print(f"H = V @ Lambdas @ V_inv:\n{H_reconstructed}")
# print(f"H - V @ Lambdas @ V_inv:\n{H - H_reconstructed}")
# # exit()

# # convert complex eigenvalues and eigenvectors to real eigenvalues 
# # in a block diagonal form and the associated real eigenvectors
# lambdas_real, Vr = cdf2rdf(lambdas, V)
# print(f"lambdas_real:\n{lambdas_real}")
# print(f"Vr:\n{Vr}")
# # exit()

# # fortran/ julia ordering
# P = np.array([
#     [-1j, 1j, 0],
#     [1, 1, 0],
#     [0, 0, 1]
# ])
# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [-1j, 1j, 0],
# ])
# # scipy ordering
# P = np.array([
#     [1j, -1j, 0],
#     [1, 1, 0],
#     [0, 0, 1]
# ])
# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [1j, -1j, 0],
# ])
# P_inv = np.linalg.inv(P)
# Lambdas_real = P @ Lambdas @ P_inv
# Lambdas_real = Lambdas_real.real # prune zero imaginary parts
# print(f"P:\n{P}")
# print(f"P_inv:\n{P_inv}")
# print(f"Lambdas_real:\n{Lambdas_real}")
# print(f"Lambdas - P_inv @ Lambdas_real @ P:\n{Lambdas - P_inv @ Lambdas_real @ P}")
# # exit()

# # fortran/ julia ordering
# R = np.array([
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
# ])
# # scipy ordering
# R = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [1, 0, 0],
# ])
# R = np.eye(3)

# Mus = R @ Lambdas_real @ R.T
# print(f"Mus:\n{Mus}")
# print(f"Mus^-1:\n{np.linalg.inv(Mus)}")
# print(f"Lambdas_real - R.T @ Mus @ R:\n{Lambdas_real - R.T @ Mus @ R}")
# # exit()

# T = Q @ V @ P_inv @ R.T
# T = T.real # prune zero imaginary parts
# # T[:, 0] /= T[-1, 0]
# # T[:, 1] /= T[0, 1]
# # T[:, 2] /= T[-1, 2]
# TI = np.linalg.inv(T)
# Mus2 = TI @ A_inv @ T
# Mus2 = Mus2.real # prune zero imaginary parts
# print(f"Mus2:\n{Mus2}")
# # np.set_printoptions(20)
# print(f"T:\n{T}")
# # print(f"T_inv:\n{T_inv}")
# # exit()

# # radau.py
# T_scipy = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0]])
# TI_scipy = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
# print(f"T_scipy:\n{T_scipy}")

# # radau5.f
# T11=9.12323948708929427920e-02
# T12=-0.1412552950209542084e+00
# T13=-3.0029194105147424492e-02
# T21=0.24171793270710701896e+00
# T22=0.20412935229379993199e+00
# T23=0.38294211275726193779e+00
# T31=0.96604818261509293619e+00

# # # firk_tableaus.jl, 
# # # see https://github.com/SciML/OrdinaryDiffEq.jl/blob/1e8716a6cc2c334ff96049ab781f0aef10fba328/src/tableaus/firk_tableaus.jl#L2
# # T11 = 9.1232394870892942792e-02
# # T12 = -0.14125529502095420843e0
# # T13 = -3.0029194105147424492e-02
# # T21 = 0.24171793270710701896e0
# # T22 = 0.20412935229379993199e0
# # T23 = 0.38294211275726193779e0
# # T31 = 0.96604818261509293619e0

# T_fortran_julia = np.array([
#     [T11, T12, T13],
#     [T21, T22, T23],
#     [T31, 1.0, 0.0],
# ], dtype=float)
# TI_fortran_julia = np.linalg.inv(T_fortran_julia)
# print(f"T_fortran_julia:\n{T_fortran_julia}")

# gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
# beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
# Mus_fortran_julia = np.array([
#     [gamma, 0, 0],
#     [0, alpha, -beta],
#     [0, beta, alpha],
# ])
# Mus_scipy = np.array([
#     [gamma, 0, 0],
#     [0, alpha, beta],
#     [0, -beta, alpha],
# ])
# # print(f"Mus_fortran_julia:\n{Mus_fortran_julia}")
# # print(f"TI_fortran_julia @ A_inv @ T_fortran_julia:\n{TI_fortran_julia @ A_inv @ T_fortran_julia}")
# error_fortran_julia = np.linalg.norm(Mus_fortran_julia - TI_fortran_julia @ A_inv @ T_fortran_julia)
# print(f"error_fortran_julia: {error_fortran_julia}")

# # print(f"Mus_scipy:\n{Mus_scipy}")
# # print(f"TI_scipy @ A_inv @ T:\n{TI_scipy @ A_inv @ T_scipy}")
# error_scipy = np.linalg.norm(Mus_scipy - TI_scipy @ A_inv @ T_scipy)
# print(f"error_scipy: {error_scipy}")

# print(f"{np.linalg.cond(T)}")
# print(f"{np.linalg.cond(T_scipy)}")
# print(f"{np.linalg.cond(T_fortran_julia)}")
# print(f"{np.linalg.cond(TI)}")
# print(f"{np.linalg.cond(TI_scipy)}")
# print(f"{np.linalg.cond(TI_fortran_julia)}")

# exit()

# # A_inv_reconstructed = vl @ np.diag(w) @ np.linalg.inv(vl)
# # A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# # print(f"A_inv_reconstructed:\n{A_inv_reconstructed}")
# # print(f"A_inv - A_inv_reconstructed:\n{A_inv - A_inv_reconstructed}")

# # T = np.array([
# #     [0, 0, 1],
# #     [1, 1, 0],
# #     [1j, -1j, 0],
# # ])
# # T_inv = np.linalg.inv(T)
# # print(f"T:\n{T}")
# # print(f"T_inv:\n{T_inv}")

# # tmp = T @ vl @ np.diag(w) @ np.linalg.inv(vl) @ T_inv

# # T_new = T @ vl
# # T_new_inv = np.linalg.inv(T_new)
# # print(f"T_new:\n{T_new}")
# # print(f"T_new_inv:\n{T_new_inv}")

# T = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0],
# ])
# TI = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.327682820 = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
# beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))

# s = 3
# A, b, c, p = RadauIIA(s)
# print(f"A:\n{A}")

# A_inv = np.linalg.inv(A)
# # TODO: We can decompose A and compute the correct matrix as the block 
# # diagonal inverse of Mu_real!
# # A_inv = A
# print(f"A_inv:\n{A_inv}")
# print(f"A_inv @ A:\n{A_inv @ A}")
# # exit()

# H, Q = hessenberg(A_inv, calc_q=True)
# # TODO: hessenberg form seems to be not necessary!
# H, Q = A_inv, np.eye(s)
# print(f"H:\n{H}")
# print(f"Q:\n{Q}")
# print(f"A_inv = Q @ H @ Q.T:\n{Q @ H @ Q.T}")
# print(f"A_inv - Q @ H @ Q.T:\n{A_inv - Q @ H @ Q.T}")
# print(f"Q.T @ Q:\n{Q.T @ Q}")
# # exit()

# lambdas, V = eig(H)

# # sort eigenvalues
# # print(np.sort(lambdas))
# idx = np.argsort(lambdas)[::-1]
# # print(lambdas[idx])

# # permut eigenvalues and eigenvectors accordingly
# lambdas = lambdas[idx]
# V = V[:, idx]

# # scale eigenvectors to get a "nice" transformation matrix (used by scipy)
# # TODO: This seems to minimize the condition number of T and TI, but why?
# for i in range(s):
#     V[:, i] /= V[-1, i]

# # # scale only the first two eigenvectors (used by Hairer)
# # V[:, 0] /= V[-1, 0]
# # V[:, 1] /= V[-1, 1]

# # # normalize eigenvectors
# # V /= np.linalg.norm(V, axis=0)

# V_inv = np.linalg.inv(V)
# Lambdas = np.diag(lambdas)
# print(f"Lambdas:\n{Lambdas}")
# print(f"V:\n{V}")
# H_reconstructed = V @ Lambdas @ V_inv
# H_reconstructed = H_reconstructed.real # prune zero imaginary parts
# print(f"H = V @ Lambdas @ V_inv:\n{H_reconstructed}")
# print(f"H - V @ Lambdas @ V_inv:\n{H - H_reconstructed}")
# # exit()

# # convert complex eigenvalues and eigenvectors to real eigenvalues 
# # in a block diagonal form and the associated real eigenvectors
# lambdas_real, Vr = cdf2rdf(lambdas, V)
# print(f"lambdas_real:\n{lambdas_real}")
# print(f"Vr:\n{Vr}")
# # exit()

# # fortran/ julia ordering
# P = np.array([
#     [-1j, 1j, 0],
#     [1, 1, 0],
#     [0, 0, 1]
# ])
# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [-1j, 1j, 0],
# ])
# # scipy ordering
# P = np.array([
#     [1j, -1j, 0],
#     [1, 1, 0],
#     [0, 0, 1]
# ])
# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [1j, -1j, 0],
# ])
# P_inv = np.linalg.inv(P)
# Lambdas_real = P @ Lambdas @ P_inv
# Lambdas_real = Lambdas_real.real # prune zero imaginary parts
# print(f"P:\n{P}")
# print(f"P_inv:\n{P_inv}")
# print(f"Lambdas_real:\n{Lambdas_real}")
# print(f"Lambdas - P_inv @ Lambdas_real @ P:\n{Lambdas - P_inv @ Lambdas_real @ P}")
# # exit()

# # fortran/ julia ordering
# R = np.array([
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
# ])
# # scipy ordering
# R = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [1, 0, 0],
# ])
# R = np.eye(3)

# Mus = R @ Lambdas_real @ R.T
# print(f"Mus:\n{Mus}")
# print(f"Mus^-1:\n{np.linalg.inv(Mus)}")
# print(f"Lambdas_real - R.T @ Mus @ R:\n{Lambdas_real - R.T @ Mus @ R}")
# # exit()

# T = Q @ V @ P_inv @ R.T
# T = T.real # prune zero imaginary parts
# # T[:, 0] /= T[-1, 0]
# # T[:, 1] /= T[0, 1]
# # T[:, 2] /= T[-1, 2]
# TI = np.linalg.inv(T)
# Mus2 = TI @ A_inv @ T
# Mus2 = Mus2.real # prune zero imaginary parts
# print(f"Mus2:\n{Mus2}")
# # np.set_printoptions(20)
# print(f"T:\n{T}")
# # print(f"T_inv:\n{T_inv}")
# # exit()

# # radau.py
# T_scipy = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0]])
# TI_scipy = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
# print(f"T_scipy:\n{T_scipy}")

# # radau5.f
# T11=9.12323948708929427920e-02
# T12=-0.1412552950209542084e+00
# T13=-3.0029194105147424492e-02
# T21=0.24171793270710701896e+00
# T22=0.20412935229379993199e+00
# T23=0.38294211275726193779e+00
# T31=0.96604818261509293619e+00

# # # firk_tableaus.jl, 
# # # see https://github.com/SciML/OrdinaryDiffEq.jl/blob/1e8716a6cc2c334ff96049ab781f0aef10fba328/src/tableaus/firk_tableaus.jl#L2
# # T11 = 9.1232394870892942792e-02
# # T12 = -0.14125529502095420843e0
# # T13 = -3.0029194105147424492e-02
# # T21 = 0.24171793270710701896e0
# # T22 = 0.20412935229379993199e0
# # T23 = 0.38294211275726193779e0
# # T31 = 0.96604818261509293619e0

# T_fortran_julia = np.array([
#     [T11, T12, T13],
#     [T21, T22, T23],
#     [T31, 1.0, 0.0],
# ], dtype=float)
# TI_fortran_julia = np.linalg.inv(T_fortran_julia)
# print(f"T_fortran_julia:\n{T_fortran_julia}")

# gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
# beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
# Mus_fortran_julia = np.array([
#     [gamma, 0, 0],
#     [0, alpha, -beta],
#     [0, beta, alpha],
# ])
# Mus_scipy = np.array([
#     [gamma, 0, 0],
#     [0, alpha, beta],
#     [0, -beta, alpha],
# ])
# # print(f"Mus_fortran_julia:\n{Mus_fortran_julia}")
# # print(f"TI_fortran_julia @ A_inv @ T_fortran_julia:\n{TI_fortran_julia @ A_inv @ T_fortran_julia}")
# error_fortran_julia = np.linalg.norm(Mus_fortran_julia - TI_fortran_julia @ A_inv @ T_fortran_julia)
# print(f"error_fortran_julia: {error_fortran_julia}")

# # print(f"Mus_scipy:\n{Mus_scipy}")
# # print(f"TI_scipy @ A_inv @ T:\n{TI_scipy @ A_inv @ T_scipy}")
# error_scipy = np.linalg.norm(Mus_scipy - TI_scipy @ A_inv @ T_scipy)
# print(f"error_scipy: {error_scipy}")

# print(f"{np.linalg.cond(T)}")
# print(f"{np.linalg.cond(T_scipy)}")
# print(f"{np.linalg.cond(T_fortran_julia)}")
# print(f"{np.linalg.cond(TI)}")
# print(f"{np.linalg.cond(TI_scipy)}")
# print(f"{np.linalg.cond(TI_fortran_julia)}")

# exit()

# # A_inv_reconstructed = vl @ np.diag(w) @ np.linalg.inv(vl)
# # A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# # print(f"A_inv_reconstructed:\n{A_inv_reconstructed}")
# # print(f"A_inv - A_inv_reconstructed:\n{A_inv - A_inv_reconstructed}")

# # T = np.array([
# #     [0, 0, 1],
# #     [1, 1, 0],
# #     [1j, -1j, 0],
# # ])
# # T_inv = np.linalg.inv(T)
# # print(f"T:\n{T}")
# # print(f"T_inv:\n{T_inv}")

# # tmp = T @ vl @ np.diag(w) @ np.linalg.inv(vl) @ T_inv

# # T_new = T @ vl
# # T_new_inv = np.linalg.inv(T_new)
# # print(f"T_new:\n{T_new}")
# # print(f"T_new_inv:\n{T_new_inv}")

# T = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0],
# ])
# TI = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
# ])
# TI = np.linalg.inv(T)
# print(f"T:\n{T}")
# print(f"TI:\n{TI}")
# print(f"T_inv:\n{TI}")

# # print(f"{T @ A_inv @ np.linalg.inv(T)}")
# MUS_real = np.linalg.inv(T) @ A_inv @ T
# print(f"np.linalg.inv(T) @ A_inv @ T:\n{MUS_real}")
# # print(f"{T @ A @ np.linalg.inv(T)}")
# # print(f"{np.linalg.inv(T) @ A @ T}")

# # Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# # and a complex conjugate pair. They are written below.
# MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# MU_COMPLEX1 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               + 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# MU_COMPLEX2 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# # print(f"{MU_REAL= }")
# # print(f"{MU_COMPLEX1= }")
# # print(f"{MU_COMPLEX2= }")

# # MUS = np.array([MU_REAL, MU_COMPLEX1, MU_COMPLEX2])
# MUS = np.array([MU_COMPLEX1, MU_COMPLEX2, MU_REAL])
# print(f"MUS:\n{MUS}")

# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [1j, -1j, 0],
# ])
# # MUS_real = np.linalg.inv(P) @ np.diag(MUS) @ P
# MUS_real = P @ np.diag(MUS) @ np.linalg.inv(P)
# print(f"MUS_real:\n{MUS_real}")

# print(f"A_inv = T @ MUS_real @ np.linalg.inv(T):\n{T @ MUS_real @ np.linalg.inv(T)}")


# # print(f"{T @ np.eye(3)}")
# # v1 = null_space(A_inv - MU_REAL * np.eye(s))
# # v2 = null_space(A_inv - MU_COMPLEX1 * np.eye(s))
# # v3 = np.linalg.solve(A_inv - MU_COMPLEX2 * np.eye(s), v2)
# # print(f"v1:\n{v1}")
# # print(f"v2:\n{v2}")
# # print(f"v3:\n{v3}")
# # V = np.hstack((v1, v2, v3))
# # print(f"V:\n{V}")

# # A_inv_reconstructed = V @ np.diag(MUS) @ np.linalg.inv(V)
# # # A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# # print(f"{A_inv_reconstructed}")



# # T, Z = schur(A_inv, output="real")
# # # T, Z = rsf2csf(T, Z)
# # print(f"Z:\n{Z}")
# # print(f"T:\n{T}")76106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
# ])
# TI = np.linalg.inv(T)
# print(f"T:\n{T}")
# print(f"TI:\n{TI}")
# print(f"T_inv:\n{TI}")

# # print(f"{T @ A_inv @ np.linalg.inv(T)}")
# MUS_real = np.linalg.inv(T) @ A_inv @ T
# print(f"np.linalg.inv(T) @ A_inv @ T:\n{MUS_real}")
# # print(f"{T @ A @ np.linalg.inv(T)}")
# # print(f"{np.linalg.inv(T) @ A @ T}")

# # Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# # and a complex conjugate pair. They are written below.
# MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# MU_COMPLEX1 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               + 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# MU_COMPLEX2 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# # print(f"{MU_REAL= }")
# # print(f"{MU_COMPLEX1= }")
# # print(f"{MU_COMPLEX2= }")

# # MUS = np.array([MU_REAL, MU_COMPLEX1, MU_COMPLEX2])
# MUS = np.array([MU_COMPLEX1, MU_COMPLEX2, MU_REAL])
# print(f"MUS:\n{MUS}")

# P = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [1j, -1j, 0],
# ])
# # MUS_real = np.linalg.inv(P) @ np.diag(MUS) @ P
# MUS_real = P @ np.diag(MUS) @ np.linalg.inv(P)
# print(f"MUS_real:\n{MUS_real}")

# print(f"A_inv = T @ MUS_real @ np.linalg.inv(T):\n{T @ MUS_real @ np.linalg.inv(T)}")


# # print(f"{T @ np.eye(3)}")
# # v1 = null_space(A_inv - MU_REAL * np.eye(s))
# # v2 = null_space(A_inv - MU_COMPLEX1 * np.eye(s))
# # v3 = np.linalg.solve(A_inv - MU_COMPLEX2 * np.eye(s), v2)
# # print(f"v1:\n{v1}")
# # print(f"v2:\n{v2}")
# # print(f"v3:\n{v3}")
# # V = np.hstack((v1, v2, v3))
# # print(f"V:\n{V}")

# # A_inv_reconstructed = V @ np.diag(MUS) @ np.linalg.inv(V)
# # # A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# # print(f"{A_inv_reconstructed}")



# # T, Z = schur(A_inv, output="real")
# # # T, Z = rsf2csf(T, Z)
# # print(f"Z:\n{Z}")
# # print(f"T:\n{T}")