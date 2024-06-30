import numpy as np
import matplotlib.pyplot as plt

class NewtonPolynomial:
    def __init__(self, x, y):
        """
        Initialize the Newton polynomial with given data points.

        Parameters:
        x : array-like, shape (n,)
            The x-coordinates of the data points.
        y : array-like, shape (n, m)
            The y-coordinates of the data points, can be vector-valued.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.coef = self._divided_differences(self.x, self.y)
    
    def _divided_differences(self, x, y):
        """
        Compute the table of divided differences for the given data points.

        Parameters:
        x : array-like, shape (n,)
            The x-coordinates of the data points.
        y : array-like, shape (n, m)
            The y-coordinates of the data points, can be vector-valued.

        Returns:
        numpy.ndarray, shape (n, n, m)
            The table of divided differences.
        """
        n, m = y.shape
        coef = np.zeros((n, n, m))
        coef[:, 0, :] = y

        for j in range(1, n):
            for i in range(n - j):
                coef[i, j, :] = (coef[i + 1, j - 1, :] - coef[i, j - 1, :]) / (x[i + j] - x[i])
        
        return coef
    
    def __call__(self, val):
        """
        Evaluate the Newton polynomial and its derivative at given points.

        Parameters:
        val : array-like or float
            The x-values at which to evaluate the polynomial and its derivative.

        Returns:
        tuple of numpy.ndarrays
            The polynomial values and their derivatives at the given points.
        """
        val = np.atleast_1d(val)
        poly_result = np.zeros((val.shape[0], self.coef.shape[2]))
        deriv_result = np.zeros_like(poly_result)
        
        for k in range(len(val)):
            v = val[k]
            poly_term = np.zeros((self.coef.shape[0], self.coef.shape[2]))
            deriv_term = np.zeros((self.coef.shape[0], self.coef.shape[2]))
            for i in range(self.coef.shape[0]):
                term = self.coef[0, i, :].copy()
                for j in range(i):
                    term *= (v - self.x[j])
                poly_term[i] = term
                
                if i > 0:
                    deriv_term[i] = self.coef[0, i, :].copy()
                    for j in range(i):
                        product = self.coef[0, i, :].copy()
                        for l in range(i):
                            if l != j:
                                product *= (v - self.x[l])
                        deriv_term[i] += product / (v - self.x[j])
            
            poly_result[k, :] = poly_term.sum(axis=0)
            deriv_result[k, :] = deriv_term.sum(axis=0)

        return poly_result, deriv_result

# Example usage in 2D
x = np.linspace(1, 10, 10)
y = np.column_stack((np.log(x), np.exp(x)))

# Create the Newton polynomial object
newton_poly = NewtonPolynomial(x, y)

# Evaluate the polynomial and its derivative at some points
vals = np.linspace(1, 10, 100)
poly_vals, deriv_vals = newton_poly(vals)

# Plotting the results
plt.figure(figsize=(14, 14))

# Plot ln(x)
plt.subplot(2, 2, 1)
plt.plot(x, np.log(x), 'ro', label='Original ln(x)')
plt.plot(vals, poly_vals[:, 0], 'b-', label='Newton Approx ln(x)')
plt.legend()
plt.title('Newton Polynomial Approximation for ln(x)')
plt.xlabel('x')
plt.ylabel('ln(x)')

# Plot exp(x)
plt.subplot(2, 2, 2)
plt.plot(x, np.exp(x), 'ro', label='Original exp(x)')
plt.plot(vals, poly_vals[:, 1], 'b-', label='Newton Approx exp(x)')
plt.legend()
plt.title('Newton Polynomial Approximation for exp(x)')
plt.xlabel('x')
plt.ylabel('exp(x)')

# Plot derivative of ln(x)
plt.subplot(2, 2, 3)
plt.plot(vals, 1/vals, 'g-', label='Exact Derivative ln(x)')
plt.plot(vals, deriv_vals[:, 0], 'b--', label='Newton Derivative Approx ln(x)')
plt.legend()
plt.title('Newton Polynomial Derivative Approximation for ln(x)')
plt.xlabel('x')
plt.ylabel("ln'(x)")

# Plot derivative of exp(x)
plt.subplot(2, 2, 4)
plt.plot(vals, np.exp(vals), 'g-', label='Exact Derivative exp(x)')
plt.plot(vals, deriv_vals[:, 1], 'b--', label='Newton Derivative Approx exp(x)')
plt.legend()
plt.title('Newton Polynomial Derivative Approximation for exp(x)')
plt.xlabel('x')
plt.ylabel("exp'(x)")

plt.show()













# import numpy as np
# import matplotlib.pyplot as plt

# class NewtonPolynomial:
#     def __init__(self, x, y):
#         """
#         Initialize the Newton polynomial with given data points.

#         Parameters:
#         x : array-like, shape (n,)
#             The x-coordinates of the data points.
#         y : array-like, shape (n, m)
#             The y-coordinates of the data points, can be vector-valued.
#         """
#         self.x = np.array(x)
#         self.y = np.array(y)
#         self.coef = self._divided_differences(self.x, self.y)
    
#     def _divided_differences(self, x, y):
#         """
#         Compute the table of divided differences for the given data points.

#         Parameters:
#         x : array-like, shape (n,)
#             The x-coordinates of the data points.
#         y : array-like, shape (n, m)
#             The y-coordinates of the data points, can be vector-valued.

#         Returns:
#         numpy.ndarray, shape (n, n, m)
#             The table of divided differences.
#         """
#         n, m = y.shape
#         coef = np.zeros((n, n, m))
#         coef[:, 0, :] = y

#         for j in range(1, n):
#             for i in range(n - j):
#                 coef[i, j, :] = (coef[i + 1, j - 1, :] - coef[i, j - 1, :]) / (x[i + j] - x[i])
        
#         return coef
    
#     def __call__(self, val):
#         """
#         Evaluate the Newton polynomial and its derivative at given points.

#         Parameters:
#         val : array-like or float
#             The x-values at which to evaluate the polynomial and its derivative.

#         Returns:
#         tuple of numpy.ndarrays
#             The polynomial values and their derivatives at the given points.
#         """
#         val = np.atleast_1d(val)
#         poly_result = np.zeros((val.shape[0], self.coef.shape[2]))
#         deriv_result = np.zeros_like(poly_result)
        
#         for k in range(len(val)):
#             v = val[k]
#             poly_term = np.zeros((self.coef.shape[0], self.coef.shape[2]))
#             deriv_term = np.zeros((self.coef.shape[0], self.coef.shape[2]))
#             for i in range(self.coef.shape[0]):
#                 term = self.coef[0, i, :].copy()
#                 for j in range(i):
#                     term *= (v - self.x[j])
#                 poly_term[i] = term
                
#                 if i > 0:
#                     term = self.coef[0, i, :].copy()
#                     for j in range(i):
#                         product = term / (v - self.x[j])
#                         for l in range(i):
#                             if l != j:
#                                 product *= (v - self.x[l])
#                         deriv_term[i] += product
            
#             poly_result[k, :] = poly_term.sum(axis=0)
#             deriv_result[k, :] = deriv_term.sum(axis=0)

#         return poly_result, deriv_result

# # Example usage in 2D
# x = np.linspace(1, 10, 10)
# y = np.column_stack((np.log(x), np.exp(x)))

# # Create the Newton polynomial object
# newton_poly = NewtonPolynomial(x, y)

# # Evaluate the polynomial and its derivative at some points
# vals = np.linspace(1, 10, 100)
# poly_vals, deriv_vals = newton_poly(vals)

# # Plotting the results
# plt.figure(figsize=(14, 14))

# # Plot ln(x)
# plt.subplot(2, 2, 1)
# plt.plot(x, np.log(x), 'ro', label='Original ln(x)')
# plt.plot(vals, poly_vals[:, 0], 'b-', label='Newton Approx ln(x)')
# plt.legend()
# plt.title('Newton Polynomial Approximation for ln(x)')
# plt.xlabel('x')
# plt.ylabel('ln(x)')

# # Plot exp(x)
# plt.subplot(2, 2, 2)
# plt.plot(x, np.exp(x), 'ro', label='Original exp(x)')
# plt.plot(vals, poly_vals[:, 1], 'b-', label='Newton Approx exp(x)')
# plt.legend()
# plt.title('Newton Polynomial Approximation for exp(x)')
# plt.xlabel('x')
# plt.ylabel('exp(x)')

# # Plot derivative of ln(x)
# plt.subplot(2, 2, 3)
# plt.plot(vals, 1/vals, 'g-', label='Exact Derivative ln(x)')
# plt.plot(vals, deriv_vals[:, 0], 'b--', label='Newton Derivative Approx ln(x)')
# plt.legend()
# plt.title('Newton Polynomial Derivative Approximation for ln(x)')
# plt.xlabel('x')
# plt.ylabel("ln'(x)")

# # Plot derivative of exp(x)
# plt.subplot(2, 2, 4)
# plt.plot(vals, np.exp(vals), 'g-', label='Exact Derivative exp(x)')
# plt.plot(vals, deriv_vals[:, 1], 'b--', label='Newton Derivative Approx exp(x)')
# plt.legend()
# plt.title('Newton Polynomial Derivative Approximation for exp(x)')
# plt.xlabel('x')
# plt.ylabel("exp'(x)")

# plt.show()






# # import numpy as np
# # import matplotlib.pyplot as plt

# # class NewtonPolynomial:
# #     def __init__(self, x, y):
# #         """
# #         Initialize the Newton polynomial with given data points.

# #         Parameters:
# #         x : array-like, shape (n,)
# #             The x-coordinates of the data points.
# #         y : array-like, shape (n, m)
# #             The y-coordinates of the data points, can be vector-valued.
# #         """
# #         self.x = np.array(x)
# #         self.y = np.array(y)
# #         self.coef = self._divided_differences(self.x, self.y)
    
# #     def _divided_differences(self, x, y):
# #         """
# #         Compute the table of divided differences for the given data points.

# #         Parameters:
# #         x : array-like, shape (n,)
# #             The x-coordinates of the data points.
# #         y : array-like, shape (n, m)
# #             The y-coordinates of the data points, can be vector-valued.

# #         Returns:
# #         numpy.ndarray, shape (n, n, m)
# #             The table of divided differences.
# #         """
# #         n, m = y.shape
# #         coef = np.zeros((n, n, m))
# #         coef[:, 0, :] = y

# #         for j in range(1, n):
# #             for i in range(n - j):
# #                 coef[i, j, :] = (coef[i + 1, j - 1, :] - coef[i, j - 1, :]) / (x[i + j] - x[i])
        
# #         return coef
    
# #     def __call__(self, val):
# #         """
# #         Evaluate the Newton polynomial and its derivative at given points.

# #         Parameters:
# #         val : array-like or float
# #             The x-values at which to evaluate the polynomial and its derivative.

# #         Returns:
# #         tuple of numpy.ndarrays
# #             The polynomial values and their derivatives at the given points.
# #         """
# #         val = np.atleast_1d(val)
# #         poly_result = np.zeros((val.shape[0], self.coef.shape[2]))
# #         deriv_result = np.zeros_like(poly_result)
        
# #         for k in range(len(val)):
# #             v = val[k]
# #             for i in range(self.coef.shape[0]):
# #                 term = self.coef[0, i, :].copy()
# #                 deriv_term = np.zeros_like(term)
# #                 for j in range(i):
# #                     term *= (v - self.x[j])
# #                     product = self.coef[0, i, :].copy()
# #                     for l in range(i):
# #                         if l != j:
# #                             product *= (v - self.x[l])
# #                     deriv_term += product / (v - self.x[j])
# #                 poly_result[k, :] += term
# #                 deriv_result[k, :] += deriv_term

# #         return poly_result, deriv_result

# # # Example usage in 2D
# # num = 5
# # x = np.linspace(1, 10, num)
# # y = np.column_stack((np.log(x), np.exp(x)))

# # # Create the Newton polynomial object
# # newton_poly = NewtonPolynomial(x, y)

# # # Evaluate the polynomial and its derivative at some points
# # num = int(1e3)
# # vals = np.linspace(1, 10, num)
# # poly_vals, deriv_vals = newton_poly(vals)

# # # Plotting the results
# # # plt.figure(figsize=(14, 7))
# # fig, ax = plt.subplots(2, 2)

# # # Plot ln(x)
# # ax[0, 0].plot(x, np.log(x), 'or', label='nodes ln(x)')
# # ax[0, 0].plot(vals, np.log(vals), '-r', label='Original ln(x)')
# # ax[0, 0].plot(vals, poly_vals[:, 0], '--b', label='Newton Approx ln(x)')
# # ax[0, 0].legend()
# # ax[0, 0].grid()
# # ax[0, 0].set_title('Newton Polynomial Approximation for ln(x)')
# # ax[0, 0].set_xlabel('x')
# # ax[0, 0].set_ylabel('ln(x)')

# # # Plot ln'(x)
# # ax[1, 0].plot(x, 1 / x, 'or', label="nodes ln'(x)")
# # ax[1, 0].plot(vals, 1 / vals, '-r', label="Original ln'(x)")
# # ax[1, 0].plot(vals, deriv_vals[:, 0], '--b', label="Newton Approx ln'(x)")
# # ax[1, 0].legend()
# # ax[1, 0].grid()
# # ax[1, 0].set_title("Newton Polynomial Approximation for ln'(x)")
# # ax[1, 0].set_xlabel("x")
# # ax[1, 0].set_ylabel("ln'(x)")

# # # Plot exp(x)
# # ax[0, 1].plot(x, np.exp(x), 'or', label='nodes exp(x)')
# # ax[0, 1].plot(vals, np.exp(vals), '-r', label='Original exp(x)')
# # ax[0, 1].plot(vals, poly_vals[:, 1], '--b', label='Newton Approx exp(x)')
# # ax[0, 1].legend()
# # ax[0, 1].grid()
# # ax[0, 1].set_title('Newton Polynomial Approximation for exp(x)')
# # ax[0, 1].set_xlabel('x')
# # ax[0, 1].set_ylabel('exp(x)')

# # # Plot exp'(x)
# # ax[1, 1].plot(x, np.exp(x), 'or', label="nodes exp'(x)")
# # ax[1, 1].plot(vals, np.exp(vals), '-r', label="Original exp'(x)")
# # ax[1, 1].plot(vals, deriv_vals[:, 1], '--b', label="Newton Approx exp'(x)")
# # ax[1, 1].legend()
# # ax[1, 1].grid()
# # ax[1, 1].set_title("Newton Polynomial Approximation for exp'(x)")
# # ax[1, 1].set_xlabel("x")
# # ax[1, 1].set_ylabel("exp'(x)")

# # plt.show()







# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # def divided_diff(x, y):
# # #     '''
# # #     function to calculate the divided
# # #     differences table
# # #     '''
# # #     n = len(y)
# # #     coef = np.zeros([n, n])
# # #     # the first column is y
# # #     coef[:, 0] = y
    
# # #     for j in range(1, n):
# # #         for i in range(n - j):
# # #             coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
            
# # #     return coef

# # # def newton_poly(coef, x_data, x):
# # #     '''
# # #     evaluate the newton polynomial 
# # #     at x
# # #     '''
# # #     n = len(x_data) - 1 
# # #     p = coef[n]
# # #     for k in range(1,n+1):
# # #         p = coef[n-k] + (x -x_data[n-k])*p
# # #     return p

# # # x = np.array([-5, -1, 0, 2])
# # # y = np.array([-2, 6, 1, 3])
# # # # get the divided difference coef
# # # a_s = divided_diff(x, y)[0, :]

# # # # evaluate on new data points
# # # x_new = np.linspace(-5, 2, num=100)
# # # y_new = newton_poly(a_s, x, x_new)

# # # plt.figure(figsize = (12, 8))
# # # plt.plot(x, y, 'bo')
# # # plt.plot(x_new, y_new)
# # # plt.show()