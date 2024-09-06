import numpy as np

# Define expm_P to return a 3x3 2D array
def expm_P(e):
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * e

# Define zeta as an iterable with k elements
zeta = [1, 2, 3, 4, 5, 6, 7]  # Example with 3 elements

# Generate F1e
F1e = np.array([expm_P(e=e) for e in zeta])

# Check the dimensions of F1e
print(F1e.shape)