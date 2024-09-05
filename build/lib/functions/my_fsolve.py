def my_fsolve(m, par, tol_fsolve, tol_is_sol, max_iterations, solution_dict, full_output=True):
    import numpy as np
    import pandas as pd
    import scipy.optimize as opt
    from .char_eq import char_eq
    from .char_eq_dual import char_eq_dual
    from .char_eq_adj import char_eq_adj

    my_fun = [char_eq, char_eq_adj, char_eq_dual]
    
    temp_guess = m
    sol = False
    sol_conj = False
    iter_counter = 0

    while not sol and iter_counter < max_iterations:
        
        iter_counter += 1
        
        solution_array, infodict, ier, msg = opt.fsolve(
            my_fun[2], temp_guess, par, xtol=tol_is_sol, full_output=full_output)
        is_sol = abs(complex(*my_fun[2](solution_array, par)))
        
        if np.isclose(is_sol, 0, atol=tol_is_sol):
            sol = True
            continue
        
        solution_array, infodict, ier, msg = opt.fsolve(
            my_fun[2], solution_array, par, xtol=tol_fsolve, full_output=full_output)
        is_sol = abs(complex(*my_fun[2](solution_array, par)))
        
        if np.isclose(is_sol, 0, atol=tol_is_sol):
            sol = True
            continue
        
        temp_guess = solution_array
        

    if sol:
        solution_dict['Sol_r'].append(solution_array[0])
        solution_dict['Sol_i'].append(solution_array[1])
        solution_dict['Guess'].append(m)
        solution_dict['g(x)'].append(abs(complex(*my_fun[0](solution_array, par))))
        solution_dict['g*(x)'].append(abs(complex(*my_fun[1](solution_array, par))))
        solution_dict['ier'].append(ier)
        solution_dict['msg'].append(msg)
        solution_dict['infodict'].append(infodict)

        if abs(solution_array[1]) > tol_fsolve:

            solution_array_conj_guess = solution_array.copy()
            solution_array_conj_guess[1] *= -1

            iter_counter = 0

            while not sol_conj and iter_counter < np.sqrt(max_iterations):
                
                iter_counter += 1
                
                solution_array_conj, infodict_conj, ier_conj, msg_conj = opt.fsolve(
                    my_fun[2], solution_array_conj_guess, par, xtol=tol_is_sol, full_output=full_output)
                is_sol_conj = abs(complex(*my_fun[2](solution_array_conj, par)))

                if np.isclose(is_sol_conj, 0, atol=tol_is_sol):
                    sol_conj = True
                    continue

                solution_array_conj, infodict_conj, ier_conj, msg_conj = opt.fsolve(
                    my_fun[2], solution_array_conj, par, xtol=tol_fsolve, full_output=full_output)
                is_sol_conj = abs(complex(*my_fun[2](solution_array_conj, par)))

                if np.isclose(is_sol_conj, 0, atol=tol_is_sol):
                    sol_conj = True
                    continue

                solution_array_conj_guess = solution_array_conj
    
    if sol_conj:
        solution_dict['Sol_r'].append(solution_array_conj[0])
        solution_dict['Sol_i'].append(solution_array_conj[1])
        solution_dict['Guess'].append(solution_array_conj_guess)
        solution_dict['g(x)'].append(abs(complex(*my_fun[0](solution_array_conj, par))))
        solution_dict['g*(x)'].append(abs(complex(*my_fun[1](solution_array_conj, par))))
        solution_dict['ier'].append(ier_conj)
        solution_dict['msg'].append(msg_conj)
        solution_dict['infodict'].append(infodict_conj)

    return solution_dict