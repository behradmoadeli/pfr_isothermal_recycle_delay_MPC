def n_triu(triu):
    n, r = 0, 1
    while r != 0:
        n += 1
        r = n * (n+1) - 2*len(triu)
    return n


def flat_to_triu(flat):
    
    import numpy  as np

    n = n_triu(flat)
    triu = np.zeros((n,n), dtype=complex)

    row, col = np.triu_indices(n)
    triu[row, col] = flat

    return triu

def triu_to_flat(triu):

    import numpy  as np
    
    n = triu.shape[0]
    flat = [triu[i, j] for i in range(n) for j in range(i, n)]
    return flat

def triu_to_symm(triu):

    import numpy as np

    triu += triu.T - np.diag(triu.diagonal())

    return triu

def triu_to_hermitian(triu):
    import numpy as np
    
    # Add the conjugate transpose of the upper triangular matrix
    triu += triu.conj().T - np.diag(triu.diagonal().conj())
    
    return triu
