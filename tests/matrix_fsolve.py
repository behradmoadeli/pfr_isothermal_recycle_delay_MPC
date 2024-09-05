import numpy as np
from scipy.optimize import fsolve

def calculate_residuals(x):
    y = np.zeros((2, 2))
    
    y[0, 0] = x[0,0]**2 + x[0,1]**2 - 25
    y[0, 1] = 3 * x[0,0] - 2 * x[0,1] + x[1,1] - 7
    y[1, 1] = np.exp(x[0,0]) + x[1,1]**2 - 15
    
    # Flatten the y array into a 1D array of residuals
    return y

# Initial guess for the solution
x_guess = np.array([
    [1.0, 2.0], 
    [0.0, 3.0]
])

print(calculate_residuals(x_guess))

# Use fsolve to find the solution
x_solution = fsolve(calculate_residuals, x_guess)

# Reshape the solution into a 2x2 array
y_solution = calculate_residuals(x_solution)

print("Solution for x:")
print(x_solution)
print("Corresponding y values:")
print(y_solution)
