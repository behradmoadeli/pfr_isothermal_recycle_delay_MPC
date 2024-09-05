import numpy as np

a_r = np.random.randint(0, 5, size=(3, 3)) - 2
a_i = np.random.randint(0, 5, size=(3, 3)) - 2
a = a_r + a_i * 1j

b_r = np.random.randint(0, 5, size=(3, 1)) - 2
b_i = np.random.randint(0, 5, size=(3, 1)) - 2
b = b_r + b_i * 1j

print('a = ', a)
print('b = ', b)

print('a[:,1] = ', a)


inner_product_1 = np.dot(a[:,1], b,)
inner_product_2 = np.dot(a[:,2], b)

inner_product = np.dot(inner_product_1, inner_product_2.conjugate())

print(inner_product_1, inner_product_2, inner_product)

print(np.vdot(inner_product_1, inner_product_2).conjugate())