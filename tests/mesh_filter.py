import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *  # Assuming this contains the exponential_func
import scipy.optimize as opt

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def my_fun(x):
    # Construct a complex number from real and imag parts
    z = x[0] + 1j * x[1]
    # Define your function using z
    # Example: f(z) = z**2 + 1
    f_z = z**3 + 7*z**2 + 19*z + 13
    # Return the real and imag parts of f(z)
    return [f_z.real, f_z.imag]

# Define the ranges and the line parameters
guess_range_real = [-4, 0, 5]
guess_range_imag = [0, 5, 5]
popt = [38.062, -7.61376e-02, -37.225]
offset_abs_tol = 5
offset_rel_tol = 0.15

# Create the meshgrid
real_values = np.linspace(guess_range_real[0], guess_range_real[1], guess_range_real[2]+1)
imag_values = np.linspace(guess_range_imag[0], guess_range_imag[1], guess_range_imag[2]+1)
real_grid, imag_grid = np.meshgrid(real_values, imag_values)
mesh = real_grid + 1j * imag_grid

# Flatten both the original mesh and the filtered mesh
real_mesh_flat = real_grid.flatten()
imag_mesh_flat = imag_grid.flatten()

# Define whether to use filtered mesh or not
use_filtered_mesh = True

# Filter the mesh points if needed
filtered_mesh = []
if use_filtered_mesh:
    for real, imag in zip(real_mesh_flat, imag_mesh_flat):
        # Calculate the perpendicular distance from the point to the line
        offset_rel = np.abs(imag - exponential_func(real,*popt))/imag
        offset_abs = np.abs(imag - exponential_func(real,*popt))
        # Check if the distance is within the threshold
        if offset_rel <= offset_rel_tol or offset_abs <= offset_abs_tol:
            # Add the point to the filtered list
            filtered_mesh.append([real, imag])  # Append as [real, imag]

    # Convert filtered_mesh to a numpy array for better handling
    filtered_mesh = np.array(filtered_mesh)

# Use the mesh points for the optimization with fsolve
mesh_to_use = filtered_mesh if use_filtered_mesh else np.column_stack((real_mesh_flat, imag_mesh_flat))

for guess in mesh_to_use:
    solution = opt.fsolve(my_fun, guess)
    print("Solution:", solution)
    # Process the solution as needed
