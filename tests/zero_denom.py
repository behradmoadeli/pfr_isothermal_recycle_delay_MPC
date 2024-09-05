import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.integrate import quad, trapz
from scipy.optimize import fsolve

default_pars = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}

(k, v, D, t, R) = (default_pars['k'], default_pars['v'], default_pars['D'], default_pars['tau'], default_pars['R'])
l = [-3.5,0]
dic = {}

my_fsolve(char_eq_copy,l,default_pars,1e-9,1e-4,5,dic,True)