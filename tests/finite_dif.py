import pandas as pd
import numpy as np
from functions import *

default_pars = {'k': 1.5, 'D': 0.2, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}

path = "C:/Users/behra/OneDrive/Documents/PhD-behrad-legion/behrad_papers/reports/CSV/FeedbackGain.csv"
k = pd.read_csv(path, index_col=0)
# print(k.iloc[0,2])

t_vec = np.linspace(0,100,50000)
z_vec = np.linspace(0,1,1000)
dt = t_vec[1]
dz = z_vec[1]

M = len(t_vec)
N = len(z_vec)

phi = np.zeros((M,N))
psi = np.zeros((M,N))
u = np.zeros_like(t_vec)

phi[0] = np.ones_like(z_vec) * 6
psi[0] = 10 - 4 * z_vec

j = 1
a=[]
for i, z in enumerate(z_vec):
    a1 = k.iloc[i,0]
    a2 = phi[j-1,i]
    a3 = k.iloc[i,1]
    a4 = psi[j-1,i]


for j, t in enumerate(t_vec[1:], start=1):
    u[j] = sum([k.iloc[i,0] * phi[j-1,i] + k.iloc[i,1] * psi[j-1,i] for i, z in enumerate(z_vec)])
    phi[j, 1:-1] = [finite_dif_fun_1(phi[j-1, i:i+3], dz, dt, default_pars) for i, z in enumerate(z_vec[:-2])]
    psi[j, :-1]  = [finite_dif_fun_2(psi[j-1, i:i+2], dz, dt, default_pars) for i, z in enumerate(z_vec[:-1])]
    phi[j, -1] = phi[j, -2]
    psi[j, -1] = phi[j, -2]
    phi[j, 0] = finite_dif_fun_3(phi[j, 1], psi[j,0], u[j], dz, default_pars)

import matplotlib.pyplot as plt

plt.plot(t_vec, phi[:,995])
plt.show()
