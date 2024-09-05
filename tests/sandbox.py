import numpy as np

a = np.array([3-1j,2j])
b = np.array([3,-4+2j])

y1 = np.dot(a,b)
y2 = np.dot(b,a)

print(y1, y2)