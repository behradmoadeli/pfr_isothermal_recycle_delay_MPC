def find_eig(par=None, default_pars=None, **kwargs):
    """
    This function solves the char equation in complex plane for different initial guesses.

    Parameters:
        - par (dict): A dictionary containing parameters for the system's matrix. If not provided, keys may be passed separately. Absent keys will take default values.

        - default_pars (dict): Default parameters dictionary to make labels.
    
    **kwargs (keyword arguments):
        - adj (bool): 'True' to slve for A, 'False' to solve for A* (default 'True')
        - guess_range_real (list): A list specifying the range of real parts of initial guess values. (default [-350, 50, 100])
        - guess_range_imag (list): A list specifying the range of imaginary parts of initial guess values. (default [0, 300, 75])
        - guess_single (complex): A single initial guess for eigenvalue calculation (real + imaginary part).
        - tol_fsolve (float): Tolerance for fsolve array-like comaprison to converge. (default 1e-9)
        - tol_is_sol (float): Tolerance for a complex solution to be accepted. (default 5e-3)
        - max_iter (int) : Maximum iteration passed to 'my_fsolve'. (default 25)
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
    from .create_label import create_label
    from .obtain_default_pars import obtain_default_pars
    from .my_fsolve import my_fsolve

    import warnings
    warnings.simplefilter('ignore')


    # Create appropriate label for par
    if par['label'] == 'default':
        default_pars = par.copy()

    max_iter = kwargs.get('max_iter', 25)
    
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
    
    if 'mesh_filter' in kwargs:
        mesh_filtering = True
        popt = kwargs['mesh_filter'][:3]
        mesh_tol = kwargs['mesh_filter'][3:]
    else:
        mesh_filtering = False
        
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
        'Sol_r':[], 'Sol_i':[], 'Guess':[], 'g(x)':[], 'g*(x)':[], 'ier':[], 'msg':[], 'infodict':[]
    }
    
    def exponential_func(x, a, b, c): #Define a function to extrapolate initial guesses (for filtered mesh)
        return a * np.exp(b * x) + c

    # Create the meshgrid
    real_values = np.linspace(guess_range_real[0], guess_range_real[1], guess_range_real[2]+1)
    imag_values = np.linspace(guess_range_imag[0], guess_range_imag[1], guess_range_imag[2]+1)
    real_grid, imag_grid = np.meshgrid(real_values, imag_values)

    # Flatten both the original mesh and the filtered mesh
    real_mesh_flat = real_grid.flatten()
    imag_mesh_flat = imag_grid.flatten()

    # Filter the mesh points if needed
    if mesh_filtering:
        filtered_mesh = []
        for real, imag in zip(real_mesh_flat, imag_mesh_flat):
            # Calculate the perpendicular distance from the point to the line
            offset_rel = np.abs(imag - exponential_func(real,*popt))/imag
            offset_abs = np.abs(imag - exponential_func(real,*popt))
            # Check if the distance is within the threshold
            if offset_rel <= mesh_tol[1] or offset_abs <= mesh_tol[0]:
                # Add the point to the filtered list
                filtered_mesh.append([real, imag])  # Append as [real, imag]

        # Convert filtered_mesh to a numpy array for better handling
        filtered_mesh = np.array(filtered_mesh)
        mesh_to_use = filtered_mesh
    else:
        mesh_to_use = np.column_stack((real_mesh_flat, imag_mesh_flat))

    for guess in mesh_to_use:
        solution_dict = my_fsolve(guess, par, tol_fsolve, tol_is_sol, max_iter, solution_dict, full_output=True)
        # Process the solution as needed

    solution_df = pd.DataFrame(solution_dict)
    return (solution_df, par['label'], metadata)