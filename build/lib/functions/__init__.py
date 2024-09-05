from .set_directory import set_directory
from .obtain_default_pars import obtain_default_pars
from .create_custom_pars_list import create_custom_pars_list
from .load_dataframe_from_csv import load_dataframe_from_csv
from .generate_unique_file_path import generate_unique_file_path
from .significant_digits import significant_digits
from .process_dataframe import process_dataframe
from .cluster_dataframe import cluster_dataframe
from .char_eq import char_eq
from .char_eq_adj import char_eq_adj
from .char_eq_copy import char_eq_copy
from .my_fsolve import my_fsolve
from .find_eig import find_eig
from .find_eig_copy import find_eig_copy
from .save_dataframe_to_csv import save_dataframe_to_csv
from .create_label import create_label
from .plot_single_df import plot_single_df
from .eig_fun import *
from .upper_triangular import *
from .ricatti import *
from .ricatti_finite import *
from .finite_dif_funs import *
from .calc_eig_finite import *

from sys import version_info
# Check if Python version is 3.11 or higher
if version_info >= (3, 11):
    from .plot_multiple_datasets import plot_multiple_datasets