import numpy as np

n = 4

b = np.zeros((n,1))
c = np.zeros((1,n))
b[0] = 3
c[0,int(0.5*n-1)] = 2

print(b)
print(c)
print(c@b)
print(-b@c)