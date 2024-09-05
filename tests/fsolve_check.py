from functions import *
import numpy as np
working_directory = set_directory(__file__)

default_pars = obtain_default_pars('pars_list.csv')
print(default_pars)
m = [-0.8133721902540637, 2.338310581683859]
par = default_pars
solution_dict = {
    'Sol_r':[], 'Sol_i':[], 'Guess':[], 'g(x)':[], 'g*(x)':[], 'ier':[], 'msg':[], 'infodict':[]
}

solution_dict = my_fsolve(m, par, 1e-9, 1e-6, 200, solution_dict, full_output=True)
# print(solution_dict)
my_fun = [char_eq_dual]
solution_array = [solution_dict['Sol_r'][0], solution_dict['Sol_i'][0]]
print(solution_array)
is_sol = abs(complex(*my_fun[0](solution_array, par)))
print(is_sol)