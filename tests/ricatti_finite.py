from scipy.optimize import fsolve
from functions import *
import numpy as np
import numpy.linalg as lina

A = np.array([
    [0.9649, 0.9572, 0.1419],
    [0.1576, 0.4854, 0.4218],
    [0.9706, 0.8003, 0.9157]])

B = np.array([1,0,0])

w,v = lina.eig(A)

print('eig_val = ', w)

print('eig_vec = ', v[0])

l = []

for i in range(2):
    l.append(A * v[i] / v[i])

print(l)