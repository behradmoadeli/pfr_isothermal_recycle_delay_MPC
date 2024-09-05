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
print(default_pars)

guess = {
    'guess_range_real':[-25,5,120],
    'guess_range_imag':[0,100,100]
}

save_dataframe_to_csv(*find_eig(default_pars, **guess, round_sig_digits=3), 'CSV')

df, label, metadata = plot_single_df(
        'CSV/default.csv', filter=True,
        real_lower_bound=-25, real_upper_bound=10, imag_lower_bound=-100, imag_upper_bound=100
)
df.head(10)