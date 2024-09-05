path = 'CSV/default.csv'      # B.C.
# path = 'CSV/default_old.csv'  # Determinant

import os
import numpy as np
from functions import *

# Get the directory where the current script is located
script_directory = os.getcwd()

# Change the current working directory to the script's directory
os.chdir(script_directory)

# Now the CWD is the same as the script's directory
print("New Current Working Directory:", os.getcwd())

default_pars = obtain_default_pars('pars_list.csv')

if not os.path.exists(path):
    guess = {
        'guess_range_real':[-25,5,5],
        'guess_range_imag':[0,100,2]
    }
    save_dataframe_to_csv(*find_eig_copy(default_pars, **guess, round_sig_digits=6), 'CSV')
else:
    print("Solution has already been saved in the appropriate location.")