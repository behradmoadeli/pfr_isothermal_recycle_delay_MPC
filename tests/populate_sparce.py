import numpy as np
from scipy.sparse import csr_matrix

# Define the size of the matrix
n = 10  # Size of the matrix (1000x1000)

# Lists to hold data for middle rows
rows_data = [2,2]
rows_row_indices = [0,0]
rows_col_indices = [0,1]

# Fill the middle rows (from index 1 to 998)
for i in range(1, n-1):
    # Fill the element before the diagonal (3), if it exists
    rows_data.extend([3, 4, 5])
    rows_row_indices.extend([i, i, i])
    rows_col_indices.extend([i-1, i, i+1])

# Define the last row
rows_data.extend([6, 6])
rows_row_indices.extend([n-1, n-1])
rows_col_indices.extend([n-2, n-1])

# Construct the middle rows as a sparse matrix
A = csr_matrix((rows_data, (rows_row_indices, rows_col_indices)), shape=(n, n))

# Output the constructed sparse matrix A
print("Sparse matrix A (CSR format):")
# print(A)

# You can use `A.toarray()` to see the full dense matrix (not recommended for very large matrices)
print(A.toarray())
