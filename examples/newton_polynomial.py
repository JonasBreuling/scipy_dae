import numpy as np
import matplotlib.pyplot as plt


def divided_diff(x, y):
    """function to calculate the divided differences table"""
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
            
    return coef


def newton_poly(coef, x_data, x):
    """evaluate the newton polynomial and its first derivative at x"""
    # n = len(x_data) - 1 
    # p = coef[n]
    # der = np.zeros_like(p)
    # for k in range(1, n + 1):
    #     der = p + (x - x_data[n-k]) * der
    #     p = coef[n-k] + (x - x_data[n-k]) * p
    # return p, der

    # Horners method, see https://orionquest.github.io/Numacom/lectures/interpolation.pdf
    n = len(x_data) - 1 
    y = coef[n]
    yp = 0
    for j in range(n, -1, -1):
        yp = y + (x - x_data[j]) * yp
        y = coef[j] + (x - x_data[j]) * y

    # yp = np.zeros_like(y)
    return y, yp

# define a function and its derivative
f = lambda x: np.arctan(x)
fp = lambda x: 1 / (1 + x**2)

# sample points
# x = np.array([-1, 0.25, 1])
# x = np.array([-2, -1, 0.25, 1, 2.5])
num = 5
# num = 20
x = np.linspace(-1, 1, num=num)

# evaluate function values
y = f(x)

# get the divided difference coef
a_s = divided_diff(x, y)[0, :]

# evaluate on new data points
x_new = np.linspace(1.25 * min(x), 1.25 * max(x), num=100)
y_new, yp_new = newton_poly(a_s, x, x_new)

fig, ax = plt.subplots(2, 1)

ax[0].plot(x_new, f(x_new), "-k", label="f(x)")
ax[0].plot(x, y, "bo", label="f(x_i)")
ax[0].plot(x_new, y_new, label="P(x)")
ax[0].grid()
ax[0].legend()

ax[1].plot(x_new, fp(x_new), "-k", label="f'(x)")
ax[1].plot(x, fp(x), "bo", label="f'(x_i)")
ax[1].plot(x_new[:-1], (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1]), "rx", label="f(x_{i+1}) - f(x_i))] / (x_{i+1} - x_i)")
ax[1].plot(x_new, yp_new, label="P'(x)")
ax[1].grid()
ax[1].legend()

plt.show()