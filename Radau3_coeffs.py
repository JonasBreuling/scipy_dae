from sympy import *
import numpy as np

if __name__ == "__main__":
    s6 = sqrt(6)
    # fmt: off
    # A = Matrix([
    #     [Rational(88, 360) - Rational(7, 360) * s6, Rational(296, 1800) - Rational(169, 1800) * s6, -Rational(2, 225) + Rational(3, 225) * s6],
    #     [Rational(296, 1800) + Rational(169, 1800) * s6, Rational(88, 360) + Rational(7, 360) * s6, -Rational(2, 225) - Rational(3, 225) * s6],
    #     [Rational(16, 36) - Rational(1, 36) * s6, Rational(16, 36) + Rational(1, 36) * s6, Rational(1, 9)]
    # ])
    A = Matrix([
        [(88 - 7 * s6) * Rational(1, 360), (296 - 169 * s6) * Rational(1, 1800), (-2 + 3 * s6) * Rational(1, 225)],
        [(296 + 169 * s6) * Rational(1, 1800), (88 + 7 * s6) * Rational(1, 360), (-2 - 3 * s6) * Rational(1, 225)],
        [(16 - s6) * Rational(1, 36), (16 + s6) * Rational(1, 36), Rational(1, 9)]
    ])
    # fmt: on

    b = A[-1, :]
    # c = A[0] + A[1] + A[2]
    c = A * ones(3, 1)

    print(f"A:\n{A}")
    print(f"b:\n{b}")
    print(f"c:\n{c}")

    # Q, R = A.QRdecomposition()
    # print(f"Q:\n{Q}")
    # print(f"R:\n{R}")

    # U, s, Vh = np.linalg.svd(np.array(Inverse(A).as_explicit()).astype(np.float64))
    U, s, Vh = np.linalg.svd(np.array(A).astype(np.float64))
    print(f"U:\n{U}")
    print(f"s:\n{s}")
    print(f"Vh:\n{Vh}")

    MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
    MU_COMPLEX1 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
                  - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
    MU_COMPLEX2 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
                  + 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
    print(f"MU_REAL: {MU_REAL}")
    print(f"MU_COMPLEX1: {MU_COMPLEX1}")
    print(f"MU_COMPLEX2: {MU_COMPLEX2}")
    # exit()


    T = np.array([
        [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
        [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
        [1, 1, 0]])
    TI = np.array([
        [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
    
    # TT = T + 1j * TI

    D = np.diag([MU_COMPLEX1, MU_COMPLEX2, MU_REAL])

    res = TI @ D @ T
    print(res)

    exit()

    # print(f"A.eigenvals(): {A.eigenvals()}")
    # print(f"A.eigenvects(): {A.eigenvects()}")
    # print(f"")

    P, D = Inverse(A).as_explicit().diagonalize()
    # P, D = A.diagonalize()
    P_real, P_complex = P.as_real_imag()
    print(f"P:\n{np.array(P).astype(np.complex128)}")
    print(f"P_real:\n{np.array(P_real).astype(np.float64)}")
    print(f"P_complex:\n{np.array(P_complex).astype(np.complex128)}")
    print(f"D:\n{np.array(D).astype(np.complex128)}")
    print(f"")
