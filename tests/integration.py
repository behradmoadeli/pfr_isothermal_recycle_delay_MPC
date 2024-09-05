import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.integrate import quad, trapz
from scipy.optimize import fsolve

default_pars = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
# l = [0.57438205, 0.22125264-3.35991407*1j] # BC
# l = [0.54592629, 0.19074795-3.46261546*1j] # det(A)
(k, v, D, t, R) = (default_pars['k'], default_pars['v'], default_pars['D'], default_pars['tau'], default_pars['R'])

lambdas = [
    (0.5744+0j),
    (0.2213+3.36j),
    (0.2213-3.36j),
    (-0.7285+7.004j),
    (-0.7285-7.004j)
    ]

b_star = [
    (0.4001927535210869+0j),
    (0.3686793328159092-0.08859051721502866j),
    (0.3686793328159092+0.08859051721502866j),
    (0.31110224959830535-0.13373720160985592j),
    (0.31110224959830535+0.13373720160985592j)
    ]

for n in range(len(lambdas)):
    for m in range(n+1):
        l = [lambdas[n], lambdas[m]]
        y = quad(eig_fun_mul_1,0,1,args=(default_pars, l, b_star[n]),complex_func=True)[0]
        print(f'<Phi_{n+1},Phi*_{m+1}> = {y}')

# print(complex(*quad(eig_fun_mul_1,0,1,args=(par, l),complex_func=True)))

# x=0
# print(eig_fun_adj_2(x, par, l[0])-eig_fun_adj_1(x, par, l[0]))
# print(eig_fun_adj_1_prime(x, par, l[0]))
# print(eig_fun_adj_1_prime(x, par, l[1]))

# x=1
# print(eig_fun_2(x, par, l[1])-eig_fun_1(x, par, l[1]))
# print(eig_fun_1_prime(x, par, l[0]))
# print(eig_fun_1_prime(x, par, l[1]))

# print(D * eig_fun_adj_1_prime(1, par, l[1]) + v * eig_fun_adj_1(1, par, l[1]) - R * v * eig_fun_adj_2(1, par, l[1]))
# print(D * eig_fun_1_prime(0, par, l[1]) - v * eig_fun_1(0, par, l[1]) + R * v * eig_fun_2(0, par, l[1]))

# def my_fun(l):
#     l = l[0]+l[1]*1j
#     y = D * eig_fun_adj_1_prime(1, par, l) + v * eig_fun_adj_1(1, par, l) - R * v * eig_fun_adj_2(1, par, l)
#     return y.real, y.imag


# print(fsolve(my_fun,[0.5,0]))
# print(fsolve(char_eq,[0.22125264,3.35991407],par))

# x = np.linspace(0.2,0.25,1000)
# y = []
# for i in x:
#     y.append(my_fun([i, 3.35991407])[0])

# plt.plot(x,np.real(y))
# plt.show()

# x = np.linspace(0,1,10000)
# y=[]

# phi = []
# psi = []
# phi_star = []
# psi_star = []

# for i in x:
#     phi.append(eig_fun_1(i, par, l[0]))
#     psi.append(eig_fun_2(i, par, l[0]))
#     phi_star.append(eig_fun_adj_1(i, par, l[0]))
#     psi_star.append(eig_fun_adj_2(i, par, l[0]))    
    # y.append(eig_fun_mul_1(i, par, l))

# plt.plot(x,phi_star)
# plt.show()
# I = trapz(y,x)
# print(I)

# print(complex(*quad(eig_fun_mul_1,0,1,args=(par, l),complex_func=True)))
