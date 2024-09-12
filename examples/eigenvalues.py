import sympy as sp

# Initialize pretty printing
sp.init_printing()

# Define the variables
x, y, phi = sp.symbols('x, y, phi')
E, A, G, A, E, I, rho, L, h = sp.symbols('E, A, G, A, E, I, rho, L, h', real=True, positive=True)  # Unknown parameters

# w = 1e-2
# L = 1
# A = w**2
# I = w**4 * sp.Rational(1, 12)

# STEEL = True
# # STEEL = False

# # SCALE_UNITS = True
# SCALE_UNITS = False

# if STEEL:
#     # rho = 8e3  # [kg / m^3]
#     # E = 260.0e9 # [N / m^2]
#     # G = 100.0e9 # [N / m^2]
#     rho = 8e3  # [kg / m^3]
#     E = 200.0e9 # [N / m^2]
#     G = 100.0e9 # [N / m^2]
# else:
#     E = 0.5 * (0.1 - 0.01) * 1e9
#     G = 0.0006 * 1e9

# if SCALE_UNITS:
#     # [m] -> [mm]
#     L *= 1e3

#     # cross section properties
#     w *= 1e3
#     rho *= 1e-9

#     # [N / m^2] -> [kN / mm^2]
#     E *= 1e-9
#     G *= 1e-9

M = rho * L * sp.Rational(1, 3) * sp.diag(A, A, I)
# M.simplify()
M_inv = sp.Inverse(M)
# M_inv.simplify()
M_inv = 3 / (rho * L) * sp.diag(1 / A, 1 / A, 1 / I)

fint_q = sp.Matrix([
    [E * A / L, 0, 0],
    [0, G * A / L, -G * A * sp.Rational(1, 2)],
    [0, G * A * sp.Rational(1, 2), G * A * L * sp.Rational(1, 4) + E * I / L],
])
fint_q.simplify()

# Jacobian matrix
# J = sp.BlockMatrix([
#     [sp.ZeroMatrix(3, 3), sp.Identity(3)],
#     [M_inv * fint_q, sp.ZeroMatrix(3, 3)],
# ]).as_explicit().simplify()

J = -M_inv * fint_q

J = -3 / (rho * L) * sp.Matrix([
    [E / L, 0, 0],
    [0, G / L, -G * sp.Rational(1, 2)],
    [0, G * sp.Rational(1, 2), G * A / I * sp.Rational(1, 2) + E / L],
])

print(f"J:")
sp.pretty_print(J)
# print(J)

# # amplification matrix Euler forward
# G = (sp.Identity(6) - h * J).as_explicit()

# eigenvals = G.eigenvals()
eigenvals = J.eigenvals()
# eigenvals = sp.Matrix([key for key, value in eigenvals.items()])
print(f"eigenvals:")
print(eigenvals)
# sp.pretty_print(eigenvals)

h_max = []
for i, eig in enumerate(eigenvals):
    print(f"- i: {i}")
    real, imag = eig.as_real_imag()
    real = real.simplify()
    imag = imag.simplify()
    print(f"  real: {real}")
    print(f"  imag: {imag}")

    # h_max.append(2 / sp.Abs(eig).evalf())

    h_max.append(sp.solve(1 + h * real, h))

# print(f"h_max:")
# print(h_max)

exit()

# Iterate through eigenvalues and determine the step size restrictions
h_max_exprs = []

for eig in eigenvals:
    # Assuming eigenvalue is real and negative
    print(eig)
    if eig.as_real_imag()[1] == 0 and (sp.re(eig) < 0) == True:
        h_max = 2 / sp.abs(eig)
    else:
        # For complex eigenvalues, compute the general condition |1 + h*lambda| <= 1
        re_part, im_part = eig.as_real_imag()
        h_max = 2 / sp.sqrt(re_part**2 + im_part**2)
    
    h_max_exprs.append(h_max)

print(h_max_exprs)

# # Intermediate quantities
# gamma = sp.Rational(1, 2) * sp.Abs(phi) * sp.cot(sp.Rational(1, 2) * sp.Abs(phi))
# K = sp.Matrix([
#     [EA, 0, 0],
#     [0, GA, 0],
#     [0, 0, EI]
# ])
# ga_x_bar = x * gamma + sp.Rational(1, 2) * phi * y
# ga_y_bar = y * gamma - sp.Rational(1, 2) * phi * x
# kappa_bar = phi
# eps = sp.Matrix([
#     ga_x_bar,
#     ga_y_bar,
#     kappa_bar,
# ])
# la = K * eps / L
# W = sp.Matrix([
#     [1 / phi * sp.sin(phi), -1 / phi * (1 - sp.cos(phi)), 0],
#     [1 / phi * (1 - sp.cos(phi)), 1 / phi * sp.sin(phi), 0],
#     [sp.Rational(1, 2) * ga_y_bar, sp.Rational(1, 2) * ga_x_bar, 1],
# ])

# print(f"la:")
# sp.pretty_print(la)

# print(f"W:")
# sp.pretty_print(W)

# # Define the vector-valued function
# # W.simplify()
# # la.simplify()
# F = W * la
# # F.simplify()

# # Define the vector of variables with respect to which we want to compute the Jacobian
# q = sp.Matrix([x, y, phi])

# # Compute the Jacobian
# Jacobian = F.jacobian(q)
# # Jacobian.simplify()

# # Display the Jacobian matrix
# print(f"Jacobian:")
# sp.pretty_print(Jacobian)

# # Jacobian_substituted = Jacobian.subs({x: L, y: 0}).doit().limit(phi, 0)
# # # Jacobian_substituted = Jacobian.subs({x: L, y: 0}).limit(phi, 0)
# # # Jacobian_substituted = Jacobian.limit(phi, 0).subs({x: L, y: 0})
# # # Jacobian_substituted = Jacobian.subs({x: L, y: 0}).evalf()

# # print(f"Jacobian_substituted:")
# # sp.pretty_print(Jacobian_substituted)
