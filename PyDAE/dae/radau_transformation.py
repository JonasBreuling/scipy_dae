import numpy as np
from scipy.linalg import schur, rsf2csf, eig, null_space

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

w, vl = eig(A_inv)

# rearrange values such that first entry is the real eigenvalue
# w = np.roll(w, 1)
# vl = np.roll(vl, 1, axis=1) # roll columns
print(f"w:\n{w}")
print(f"vl:\n{vl}")
rank = np.linalg.matrix_rank(vl)
print(f"{rank= }")

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