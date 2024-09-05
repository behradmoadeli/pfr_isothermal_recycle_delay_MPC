def calc_eig_finite(A):
    """
    Calculate finite eigenpairs (eigenvalues and corresponding eigenvectors for both A and A^T) of a square matrix A.

    Args:
    A (numpy.ndarray): The square matrix for which to calculate eigenpairs.

    Returns:
    tuple: A tuple containing three arrays:
        - eigenvalues (numpy.ndarray): Array of eigenvalues.
        - eigenvectors_A (numpy.ndarray): Matrix of eigenvectors corresponding to A.
        - eigenvectors_AT (numpy.ndarray): Matrix of eigenvectors corresponding to A^T.
    """
    import numpy as np
    import numpy.linalg as lina

    n = np.shape(A)[0]  # Get the size of the square matrix
    
    # Compute eigenvalues and eigenvectors of A and A^T
    l = lina.eig(A)[0]
    p = lina.eig(A)[1].transpose()
    s = lina.eig(A.transpose())[1].transpose()

    eigenvalues = l
    eigenvectors_A = p
    eigenvectors_AT = []

    for i in range(n):
        for j in range(n):
            if np.isclose(lina.norm(np.matmul(A.transpose(), s[j]) / l[i] - s[j]), 0):
                eigenvectors_AT.append(s[j])
                break  # Move to the next eigenvalue

    return eigenvalues, eigenvectors_A.transpose(), np.array(eigenvectors_AT).transpose()
