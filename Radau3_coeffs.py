from sympy import *
import numpy as np

if __name__ == "__main__":
    s6 = sqrt(6)
    # fmt: off
    A = Matrix([
        [(88 - 7 * s6) * Rational(1, 360), (296 - 169 * s6) * Rational(1, 1800), (-2 + 3 * s6) * Rational(1, 225)],
        [(296 + 169 * s6) * Rational(1, 1800), (88 + 7 * s6) * Rational(1, 360), (-2 - 3 * s6) * Rational(1, 225)],
        [(16 - s6) * Rational(1, 36), (16 + s6) * Rational(1, 36), Rational(1, 9)]
    ])
    # fmt: on

    b = A[-1, :]
    c = A * ones(3, 1)

    print(f"A:\n{A}")
    print(f"b:\n{b}")
    print(f"c:\n{c}")

    MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
    MU_COMPLEX2 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
                  - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
    MU_COMPLEX1 = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
                  + 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))

    T_scipy = np.array([
        [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
        [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
        [1, 1, 0]])
    TI_scipy = np.array([
        [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

    T11 =  9.1232394870892942792e-02
    T12 =  -0.14125529502095420843e0
    T13 =  -3.0029194105147424492e-02
    T21 =  0.24171793270710701896e0
    T22 =  0.20412935229379993199e0
    T23 =  0.38294211275726193779e0
    T31 =  0.96604818261509293619e0
    TI11 = 4.3255798900631553510e0
    TI12 = 0.33919925181580986954e0
    TI13 = 0.54177053993587487119e0
    TI21 = -4.1787185915519047273e0
    TI22 = -0.32768282076106238708e0
    TI23 = 0.47662355450055045196e0
    TI31 = -0.50287263494578687595e0
    TI32 = 2.5719269498556054292e0
    TI33 = -0.59603920482822492497e0

    T = np.array([[
        [T11, T12, T13],
        [T21, T22, T23],
        [T31, 1.0, 0.0],
    ]])

    TI = np.array([[
        [TI11, TI12, TI13],
        [TI21, TI22, TI23],
        [TI31, TI32, TI33],
    ]])

    cbrt9 = 9**(1 / 3)
    gamma_prime = (6.0 + cbrt9 * (cbrt9 - 1)) / 30 # eigval of `A`
    alpha_prime = (12.0 - cbrt9 * (cbrt9 - 1)) / 60 # eigval of `A`
    beta_prime = cbrt9 * (cbrt9 + 1) * np.sqrt(3) / 60 # eigval of `A`
    scale = alpha_prime**2 + beta_prime**2
    gamma = 1 / gamma_prime  # eigval of `inv(A)`
    alpha = alpha_prime / scale # eigval of `inv(A)`
    beta = beta_prime / scale # eigval of `inv(A)`

    Lambda = np.array([
        [gamma, 0, 0],
        [0, alpha, -beta],
        [0, beta, alpha]
    ])
    Lambda_scipy = np.array([
        [gamma, 0, 0],
        [0, alpha, beta],
        [0, -beta, alpha]
    ])

    A_inv = np.linalg.inv(np.array(A).astype(np.float64))

    res = TI @ Lambda @ T
    print(f"TI @ A_inv @ T:\n{TI @ A_inv @ T}")
    print(f"Lambda:\n{Lambda}")
    assert np.allclose(Lambda, TI @ A_inv @ T, rtol=0, atol=1e-8)
    assert np.allclose(Lambda_scipy, TI_scipy @ A_inv @ T_scipy, rtol=0, atol=1e-8)

    from scipy.linalg import schur
    T, Z, _ = schur(A_inv, output="real", sort="rhp")
    print(f"T:\n{T}")
    print(f"Z:\n{Z}")

    exit()

    # print(f"A_inv:\n{Inverse(A).as_explicit()}")
    # exit()

    P, J = Inverse(A).as_explicit().jordan_form()
    P = P.evalf()
    J = J.evalf()
    P_real, P_complex = P.as_real_imag()
    print(f"P:\n{P}")
    print(f"P_real:\n{np.array(P_real).astype(np.float64)}")
    print(f"P_complex:\n{np.array(P_complex).astype(np.float64)}")
    print(f"J:\n{J}")

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
