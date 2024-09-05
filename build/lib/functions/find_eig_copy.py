def find_eig_copy(par=None, default_pars=None, **kwargs):
    """
    This function solves the char equation in complex plane for different initial guesses.

    Parameters:
        - par (dict): A dictionary containing parameters for the system's matrix. If not provided, keys may be passed separately. Absent keys will take default values.

        - default_pars (dict): Default parameters dictionary to make labels.
    
    **kwargs (keyword arguments):            
        - guess_range_real (list): A list specifying the range of real parts of initial guess values. (default [-350, 50, 100])
        - guess_range_imag (list): A list specifying the range of imaginary parts of initial guess values. (default [0, 300, 75])
        - guess_single (complex): A single initial guess for eigenvalue calculation (real + imaginary part).
        - tol_fsolve (float): Tolerance for fsolve array-like comaprison to converge. (default 1e-9)
        - tol_is_sol (float): Tolerance for a complex solution to be accepted. (default 5e-3)
        - round_sig_digits (float): Number of significant digits to either separate two different solutions or merge them as one. (default 3)
        - pars_list_path (str): Path of a .csv file to extract containing default_pars. (default 'pars_list.csv')

    Returns:
        - solution_df (pandas.DataFrame): DataFrame containing found solutions' information.
        - label (str): A label describing the customized parameters used for the computation.
        - metadata (dict): A dictionary containing input parameter values used in the computation.
    """

    import numpy as np
    import pandas as pd
    import scipy.optimize as opt
    from .char_eq_copy import char_eq_copy
    from .create_label import create_label
    from .obtain_default_pars import obtain_default_pars
    from .my_fsolve import my_fsolve

    import warnings
    warnings.simplefilter('ignore')


    # Create appropriate label for par
    if par['label'] == 'default':
        default_pars = par.copy()

    if not default_pars:
        default_pars = obtain_default_pars(kwargs.get('pars_list_path', 'pars_list.csv'))
    
    if par['label'] == '':
        par['label'] = create_label(par, default_pars=default_pars, **kwargs)

    # Assign default values to missing keyword arguments for initial guess values
    if 'guess_single' in kwargs:
        guess_single_r = np.real(kwargs['guess_single'])
        guess_single_i = np.imag(kwargs['guess_single'])

        guess_range_real = [guess_single_r, guess_single_r, 0]
        guess_range_imag = [guess_single_i, guess_single_i, 0]
    else:
        guess_range_real = kwargs.get('guess_range_real', [-350, 50, 100])
        guess_range_imag = kwargs.get('guess_range_imag', [0, 300, 75])

    # Assign default values to the rest of missing keyword arguments
    tol_fsolve = kwargs.get('tol_fsolve', 1e-9)
    tol_is_sol = kwargs.get('tol_is_sol', 5e-3)
    round_sig_digits = kwargs.get('round_sig_digits', 3)

    metadata = {
        'par': par,
        'guess_range': (guess_range_real, guess_range_imag),
        'tols': (tol_fsolve, tol_is_sol, round_sig_digits)
    }

    # Constructiong a dictionary to capture legit solutions
    solution_dict = {
        'Sol_r':[], 'Sol_i':[], 'Guess':[], 'g(x)':[], 'ier':[], 'msg':[], 'infodict':[]
    }

    # Constructiong a 2D (Re-Im plane) mesh for different initial guess values
    mesh_builder = np.meshgrid(
        np.linspace(guess_range_real[0],
                    guess_range_real[1], guess_range_real[2]+1),
        np.linspace(guess_range_imag[0],
                    guess_range_imag[1], guess_range_imag[2]+1)
    )
    mesh = mesh_builder[0] + mesh_builder[1] * 1j

    for i in mesh:
        for m in i:
            # print(f"Getting results for guess = {m.real:.2f} + {m.imag:.2f}j...")
            m = np.array([m.real, m.imag])
            solution_dict = my_fsolve(char_eq_copy, m, par, tol_fsolve, tol_is_sol, 25, solution_dict, full_output=True)
    solution_df = pd.DataFrame(solution_dict)
    return (solution_df, par['label'], metadata)