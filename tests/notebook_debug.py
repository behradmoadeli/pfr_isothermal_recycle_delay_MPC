import os
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

# Get the directory where the current script is located
script_directory = os.getcwd()

# Change the current working directory to the script's directory
os.chdir(script_directory)

# Now the CWD is the same as the script's directory
print("New Current Working Directory:", os.getcwd())

default_pars = obtain_default_pars('pars_list.csv')
print(default_pars)

path = "CSV/default.csv"

if not os.path.exists(path):
    guess = {
        'guess_range_real':[-150,50,20],
        'guess_range_imag':[0,100,8]
    }
    save_dataframe_to_csv(*find_eig(default_pars, **guess, round_sig_digits=5, tol_is_sol=1e-7, max_iter=200), 'CSV')
else:
    print("Solution has already been saved in the appropriate location.")
    
df, label, metadata = plot_single_df(
        path, filter=True,
        real_lower_bound=-25, real_upper_bound=10, imag_lower_bound=-100, imag_upper_bound=100
)
n_lambdas = 9
print(df.head(n_lambdas))