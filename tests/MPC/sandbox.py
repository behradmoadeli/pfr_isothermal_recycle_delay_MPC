import pandas as pd
import numpy as np
from functions import *

lambdas = [0,1,2,3]
Ts = 0.01
par = {'k': 1.5, 'D': 0.2, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
norm_coefs = np.ones_like(lambdas)

n = 3

zeta = np.linspace(0,1,n)
eta = np.linspace(0,1,n)

Zeta, Eta = np.meshgrid(zeta, eta)
df = pd.DataFrame(columns=[z for z in zeta], index=[e for e in eta])

Ad_fun=[eig_fun_1, eig_fun_2]
Ad_fun_adj=[eig_fun_adj_1, eig_fun_adj_2]
i = 1
print(Ad_fun[0](Zeta, par, lambdas[i], norm_coefs[i]))

k = np.zeros((2,2,n,n), dtype=complex)
for row in range(2):
    for col in range(2):
        k[row,col] = sum([np.exp(Ts*lambdas[i]) * Ad_fun[row](Zeta, par, lambdas[i], norm_coefs[i]) * Ad_fun_adj[col](Eta, par, lambdas[i], norm_coefs[i]) for i in range(len(lambdas))])

df.iloc[:,:] = k[0,0]

print(df.head())