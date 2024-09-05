from scipy.optimize import fsolve
from functions import *
import numpy as np

default_pars = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
lambdas = [0.57438+0j, 0.22125 - 3.3599j, 0.22125 + 3.3599j]
normal_coefs = [-0.0298-0j, 0.0182-0.1027j, 0.0182+0.1027j]

n_modes = 3

p_0_shape = np.array([*triu_to_flat(np.zeros((n_modes,n_modes)))] * 2).shape
p_0 = np.random.rand(*p_0_shape) # Initial guess


p_sol_flat = ricatti(p_0, default_pars, lambdas, normal_coefs)

# p_sol_flat = fsolve(ricatti, p_0, (default_pars, lambdas, normal_coefs))


slicer = int(len(p_sol_flat)/2)
p_sol_flat_real = p_sol_flat[:slicer]
p_sol_flat_imag = p_sol_flat[slicer:]
p_sol_flat_complex = p_sol_flat_real + p_sol_flat_imag * 1j

p_sol = triu_to_symm(flat_to_triu(p_sol_flat_complex))
    
print(p_sol)

# y_1 = np.array([[1,2,3],[0,4,5],[0,0,6]])
# y_2 = np.array([[1,1,1],[0,1,1],[0,0,1]]) * 1j
# y = y_1 + y_2

# y_complex = triu_to_symm(y)
# y_complex_flat = triu_to_flat(y_complex)
# y_real_flat, y_imag_flat = np.real(y_complex_flat), np.imag(y_complex_flat)
# y_flat_raw = [*y_real_flat, *y_imag_flat]
# print(y_flat_raw)