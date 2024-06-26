import numpy as np
import matplotlib.pyplot as plt

def hermite_confluent_vandermonde(x):
    """
    Compute the confluent Vandermonde matrix for Hermite interpolation.

    Parameters:
    x : array_like
        The array of x points (nodes) where the function and its derivative are known.

    Returns:
    V : ndarray
        The confluent Vandermonde matrix.
    """
    n = len(x)
    V = np.zeros((2*n, 2*n))

    for i in range(n):
        for j in range(2*n):
            V[2*i, j] = x[i]**j  # For function values
            if j == 0:
                V[2*i+1, j] = 0
            else:
                V[2*i+1, j] = j * x[i]**(j-1)  # For derivative values

    return V

def hermite_interpolation(x, y, dy):
    """
    Perform Hermite interpolation.

    Parameters:
    x : array_like
        The array of x points (nodes).
    y : array_like
        The array of function values at the x points.
    dy : array_like
        The array of derivatives at the x points.

    Returns:
    coeffs : ndarray
        The coefficients of the interpolating polynomial.
    """
    n, m = y.shape
    V = hermite_confluent_vandermonde(x)
    
    # Create the right-hand side vector with function values and their derivatives
    b = np.zeros((2*n, m))
    for i in range(n):
        b[2*i] = y[i]
        b[2*i+1] = dy[i]
    
    # Solve the linear system to find the coefficients
    coeffs = np.linalg.solve(V, b)
    
    return coeffs

def evaluate_polynomial(x, x_nodes, y_values, dy_values):
    """
    Evaluate a polynomial given its coefficients at the points x.

    Parameters:
    x : array_like
        The points at which to evaluate the polynomial.
    x_nodes : array_like
        The nodes where the function and its derivative values are known.
    y_values : array_like
        The function values at the nodes.
    dy_values : array_like
        The derivative values at the nodes.

    Returns:
    y : ndarray
        The evaluated polynomial values at the points x.
    dy : ndarray
        The evaluated first derivative values at the points x.
    """
    coeffs = hermite_interpolation(x_nodes, y_values, dy_values)
    n, m = y_values.shape
    p = len(x)
    y = np.zeros((m, p), dtype=float)
    dy = np.zeros((m, p), dtype=float)
    for i in range(2*n):
    # for i in range(n):
        y += np.outer(coeffs[i], x**i)
        if i > 0:
            dy += i * np.outer(coeffs[i], x**(i-1))
    return y.T, dy.T

def test_hermite_interpolation():
    def fun(x):
        return np.array([
            np.cos(2 * x),
            np.sin(x),
            0.5 * x**2 - 3 * x + 2,
        ]).T

    def der(x):
        return np.array([
            -2 * np.sin(2 * x),
            np.cos(x),
            x - 3,
        ]).T
    
    num = 4
    x0, x1 = 0, np.pi / 2
    x_nodes = np.linspace(x0, x1, num=num)
    # three stage Radau IIA nodes
    x_nodes = np.array([
        0, 
        3 / 5 - 6**0.5 / 10, 
        3 / 5 + 6**0.5 / 10,
        1,
    ])
    y_values = fun(x_nodes)
    dy_values = der(x_nodes)

    # Perform Hermite interpolation and evaluation
    x_eval = np.linspace(np.min(x_nodes), 2 * np.max(x_nodes), 100)
    y_eval, dy_eval = evaluate_polynomial(x_eval, x_nodes, y_values, dy_values)
    
    # Evaluate at the nodes to verify interpolation
    y_eval_nodes, dy_eval_nodes = evaluate_polynomial(x_nodes, x_nodes, y_values, dy_values)
    
    # Check if the function values at the nodes match the given values
    function_values_match = np.allclose(y_eval_nodes, y_values, atol=1e-8)
    derivative_values_match = np.allclose(dy_eval_nodes, dy_values, atol=1e-8)

    print("Nodes:", x_nodes)
    print("Polynomial values at nodes:", y_eval_nodes)
    print("Expected function values at nodes:", y_values)
    print("Polynomial first derivatives at nodes:", dy_eval_nodes)
    print("Expected derivative values at nodes:", dy_values)
    print("Function values match:", function_values_match)
    print("Derivative values match:", derivative_values_match)
    print()
    
    assert function_values_match, f"Function values do not match at the nodes for test case {i+1}"
    assert derivative_values_match, f"Derivative values do not match at the nodes for test case {i+1}"
    
    print("All tests passed!")

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x_eval, fun(x_eval), "-k", label="fun")
    ax[0].plot(x_nodes, y_values, "or", label="Hermite y's")
    ax[0].plot(x_eval, y_eval, "--r", label="Hermite interpolation")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(x_eval, der(x_eval), "-k", label="der")
    ax[1].plot(x_nodes, dy_values, "or", label="Hermite yp's")
    ax[1].plot(x_eval, dy_eval, "--r", label="Hermite interpolation")
    ax[1].grid()
    ax[1].legend()

    plt.show()

# Run the test function
test_hermite_interpolation()

