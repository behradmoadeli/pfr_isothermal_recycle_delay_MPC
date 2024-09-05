import numpy as np
from functions import *
par = {'k': 1.5, 'D': 0.2, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
l = [-0.588331, -3.421871]

x1 = char_eq(l, par)

x2 = char_eq_adj(l, par)

print([x1, x2])