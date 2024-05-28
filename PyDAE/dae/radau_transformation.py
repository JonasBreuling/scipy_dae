import numpy as np
from scipy.linalg import schur, rsf2csf, eig, hessenberg, null_space

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

s = 3
A, b, c, p = RadauIIA(s)
print(f"A:\n{A}")

A_inv = np.linalg.inv(A)
print(f"A_inv:\n{A_inv}")

H, Q = hessenberg(A_inv, calc_q=True)
print(f"H:\n{H}")
print(f"Q:\n{Q}")
print(f"A_inv = Q @ H @ Q^H:\n{Q @ H @ Q.conj()}")

# Lambda, V = schur(H)
# print(f"Lambda:\n{Lambda}")
# print(f"V:\n{V}")

# exit()

# w, vl = eig(A_inv)
lambdas, V = eig(H)
V_inv = np.linalg.inv(V)
Lambdas = np.diag(lambdas)

# rearrange values such that first entry is the real eigenvalue
# w = np.roll(w, 1)
# vl = np.roll(vl, 1, axis=1) # roll columns
print(f"Lambdas:\n{Lambdas}")
print(f"V:\n{V}")
rank = np.linalg.matrix_rank(V)
print(f"{rank= }")
H_reconstructed = V @ Lambdas @ V_inv
H_reconstructed = H_reconstructed.real # prune zero imaginary parts
print(f"H = V @ Lambdas @ V_inv:\n{H_reconstructed}")

P = np.array([
    [1j, -1j, 0],
    [1, 1, 0],
    [0, 0, 1]
])
# P = np.eye(3)
# Lambdas_real = Lambdas
P_inv = np.linalg.inv(P)
Lambdas_real = P @ Lambdas @ P_inv
Lambdas_real = Lambdas_real.real # prune zero imaginary parts
print(f"P:\n{P}")
print(f"P_inv:\n{P_inv}")
print(f"Lambdas_real:\n{Lambdas_real}")

R = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

Mus = R @ Lambdas_real @ R.T
print(f"Mus:\n{Mus}")

# T = Q.conj().T @ V @ P @ R
T = R.T @ P_inv @ V_inv @ Q.T
# T = R @ P @ V @ Q
# Mus2 = R.T @ P_inv @ V_inv @ Q.T @ A_inv @ Q.conj().T @ V @ P @ R
Mus2 = R.T @ P_inv @ V_inv @ Q.T @ A_inv @ Q.conj().T @ V @ P @ R
Mus2 = Mus2.real # prune zero imaginary parts
# print(f"Mus2:\n{Mus2}")
# print(f"T:\n{T}")
# print(f"T_inv:\n{np.linalg.inv(T)}")

# radau.py
T_scipy = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI_scipy = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# radau5.f
T11=9.12323948708929427920e-02
T12=-0.1412552950209542084e+00
T13=-3.0029194105147424492e-02
T21=0.24171793270710701896e+00
T22=0.20412935229379993199e+00
T23=0.38294211275726193779e+00
T31=0.96604818261509293619e+00

# # firk_tableaus.jl, 
# # see https://github.com/SciML/OrdinaryDiffEq.jl/blob/1e8716a6cc2c334ff96049ab781f0aef10fba328/src/tableaus/firk_tableaus.jl#L2
# T11 = 9.1232394870892942792e-02
# T12 = -0.14125529502095420843e0
# T13 = -3.0029194105147424492e-02
# T21 = 0.24171793270710701896e0
# T22 = 0.20412935229379993199e0
# T23 = 0.38294211275726193779e0
# T31 = 0.96604818261509293619e0
# TI11 = 4.3255798900631553510e0
# TI12 = 0.33919925181580986954e0
# TI13 = 0.54177053993587487119e0
# TI21 = -4.1787185915519047273e0
# TI22 = -0.32768282076106238708e0
# TI23 = 0.47662355450055045196e0
# TI31 = -0.50287263494578687595e0
# TI32 = 2.5719269498556054292e0
# TI33 = -0.59603920482822492497e0

T_fortran_julia = np.array([
    [T11, T12, T13],
    [T21, T22, T23],
    [T31, 1.0, 0.0],
], dtype=float)
# TI = np.array([
#     [TI11, TI12, TI13],
#     [TI21, TI22, TI23],
#     [TI31, TI32, TI33],
# ], dtype=float)
TI_fortran_julia = np.linalg.inv(T_fortran_julia)

gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
Mus_fortran_julia = np.array([
    [gamma, 0, 0],
    [0, alpha, -beta],
    [0, beta, alpha],
])
Mus_scipy = np.array([
    [gamma, 0, 0],
    [0, alpha, beta],
    [0, -beta, alpha],
])
# print(f"Mus_fortran_julia:\n{Mus_fortran_julia}")
# print(f"TI_fortran_julia @ A_inv @ T_fortran_julia:\n{TI_fortran_julia @ A_inv @ T_fortran_julia}")
error_fortran_julia = np.linalg.norm(Mus_fortran_julia - TI_fortran_julia @ A_inv @ T_fortran_julia)
print(f"error_fortran_julia: {error_fortran_julia}")

# print(f"Mus_scipy:\n{Mus_scipy}")
# print(f"TI_scipy @ A_inv @ T:\n{TI_scipy @ A_inv @ T_scipy}")
error_scipy = np.linalg.norm(Mus_scipy - TI_scipy @ A_inv @ T_scipy)
print(f"error_scipy: {error_scipy}")

exit()

# A_inv_reconstructed = vl @ np.diag(w) @ np.linalg.inv(vl)
# A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# print(f"A_inv_reconstructed:\n{A_inv_reconstructed}")
# print(f"A_inv - A_inv_reconstructed:\n{A_inv - A_inv_reconstructed}")

# T = np.array([
#     [0, 0, 1],
#     [1, 1, 0],
#     [1j, -1j, 0],
# ])
# T_inv = np.linalg.inv(T)
# print(f"T:\n{T}")
# print(f"T_inv:\n{T_inv}")

# tmp = T @ vl @ np.diag(w) @ np.linalg.inv(vl) @ T_inv

# T_new = T @ vl
# T_new_inv = np.linalg.inv(T_new)
# print(f"T_new:\n{T_new}")
# print(f"T_new_inv:\n{T_new_inv}")

T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0],
])
TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
])
T_inv = np.linalg.inv(T)
print(f"T:\n{T}")
print(f"TI:\n{TI}")
print(f"T_inv:\n{T_inv}")

# print(f"{T @ A_inv @ np.linalg.inv(T)}")
MUS_real = np.linalg.inv(T) @ A_inv @ T
print(f"np.linalg.inv(T) @ A_inv @ T:\n{MUS_real}")
# print(f"{T @ A @ np.linalg.inv(T)}")
# print(f"{np.linalg.inv(T) @ A @ T}")

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
MU_COMPLEX1 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              + 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
MU_COMPLEX2 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# print(f"{MU_REAL= }")
# print(f"{MU_COMPLEX1= }")
# print(f"{MU_COMPLEX2= }")

# MUS = np.array([MU_REAL, MU_COMPLEX1, MU_COMPLEX2])
MUS = np.array([MU_COMPLEX1, MU_COMPLEX2, MU_REAL])
print(f"MUS:\n{MUS}")

P = np.array([
    [0, 0, 1],
    [1, 1, 0],
    [1j, -1j, 0],
])
# MUS_real = np.linalg.inv(P) @ np.diag(MUS) @ P
MUS_real = P @ np.diag(MUS) @ np.linalg.inv(P)
print(f"MUS_real:\n{MUS_real}")

print(f"A_inv = T @ MUS_real @ np.linalg.inv(T):\n{T @ MUS_real @ np.linalg.inv(T)}")


# print(f"{T @ np.eye(3)}")
# v1 = null_space(A_inv - MU_REAL * np.eye(s))
# v2 = null_space(A_inv - MU_COMPLEX1 * np.eye(s))
# v3 = np.linalg.solve(A_inv - MU_COMPLEX2 * np.eye(s), v2)
# print(f"v1:\n{v1}")
# print(f"v2:\n{v2}")
# print(f"v3:\n{v3}")
# V = np.hstack((v1, v2, v3))
# print(f"V:\n{V}")

# A_inv_reconstructed = V @ np.diag(MUS) @ np.linalg.inv(V)
# # A_inv_reconstructed = A_inv_reconstructed.real # remove zero imaginary parts
# print(f"{A_inv_reconstructed}")



# T, Z = schur(A_inv, output="real")
# # T, Z = rsf2csf(T, Z)
# print(f"Z:\n{Z}")
# print(f"T:\n{T}")