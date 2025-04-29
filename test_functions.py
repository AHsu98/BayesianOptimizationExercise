import numpy as np

def test_function_1d(x):
    """
    1D test function: Modified sine wave with noise
    Input range: [-5, 5]
    Global minimum: near x ≈ -0.65
    """
    return np.sin(x) * x**2 + 0.1 * x

def test_function_2d(x):
    """
    2D test function: Bowl with a peak
    Input range: [-2, 2] for both dimensions
    Global minimum: near (-0.5, -0.5)
    """
    x1, x2 = x
    return (x1**2 + x2**2) - 2*np.exp(-(x1+0.5)**2 - (x2+0.5)**2)

def test_function_3d(x):
    """
    3D test function: Simplified Ackley function
    Input range: [-5, 5] for all dimensions
    Global minimum: at (0, 0, 0)
    """
    x1, x2, x3 = x
    term1 = -20 * np.exp(-0.2 * np.sqrt((x1**2 + x2**2 + x3**2) / 3))
    term2 = -np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2) + np.cos(2*np.pi*x3)) / 3)
    return term1 + term2 + 20 + np.e

def get_rosenbrock(d):
    def f(x):
        coupled_term = 10*(x[1:]**2-x[:-1])**2
        diagonal_term = (x  - 1.)**2
        return np.sum(coupled_term) + np.sum(diagonal_term)
    return f
