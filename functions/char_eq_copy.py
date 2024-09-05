def char_eq_copy(x, *args):
    """
    This function evaluates the charachteristic equation at a given point.

    Parameters:
        x ([float, float]):
            A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.

    Returns:
        array[float, float]:
            An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
    """
    import numpy as np

    par = args[0]

    # Extract real and imaginary parts from the input tuple
    x_real, x_imag = x[0], x[1]

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

    # Create complex numbers element-wise from real and imaginary parts
    l = complex(x_real, x_imag)

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    if np.isclose(p_sqrt, 0, atol=1e-8):
        y = np.exp(l*t + v/2/D) * (v**2/2/D + v + D) - v*R*np.exp(v/D)
    else:
        y = np.exp(l*t + v/2/D) * (
            (v**2+2*D**2)/p_sqrt * np.sinh(p_sqrt/2/D) + v*np.cosh(p_sqrt/2/D)
        ) - v*R*np.exp(v/D)

    return y.real, y.imag