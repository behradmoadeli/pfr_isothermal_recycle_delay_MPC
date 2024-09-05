def ricatti_finite(p_flat, *args):
    """
    Function that evaluates Ricatti equation given p[i,j] coefficients.

    Parameters:
    - p_flat: Matrix representing the current state.
    - *args: Additional arguments needed for the Ricatti equation solver.

    Args (in *args):
    - args[0]: Dictionary of parameters including 'k', 'v', 'D', 'tau', and 'R'.
    - args[1]: 1d array of dominant eigenvalues.
    - args[2]: 1d array of eigenfunction normalization coefficients.

    Returns:
    - 2d array that has to be zero for correct p coefs.

    """
    import numpy as np
    import numpy.linalg as lina
    from .eig_fun import eig_fun_adj_1
    from .upper_triangular import n_triu, flat_to_triu, triu_to_flat, triu_to_symm, triu_to_hermitian
    
    A = args[0]
    B = args[1]
    n = np.shape(A)[0]
    
    l, p = lina.eig(A)
    p = p.transpose()
    s = lina.eig(A.transpose())[1]
    s = s.transpose()
    
    eigs = {}
    for i in range(n):
        for j in range(n):
            if np.isclose(lina.norm(np.matmul(A, s[j])/l[i] - s[j]), 0):
                eigs[l] = (p, s)
    
    B_star = B.transpose()
    
    slicer = int(len(p_flat)/2)
    p_flat_real = p_flat[:slicer]
    p_flat_imag = p_flat[slicer:]
    p_flat_complex = p_flat_real + p_flat_imag * 1j

    p = triu_to_hermitian(flat_to_triu(p_flat_complex))
    N = p.shape[0]
    
    y = np.zeros_like(p)
    b = np.zeros_like(lambdas)

    for i in range(len(lambdas)):
        b[i] = eig_fun_adj_1(0, par, lambdas[i], normal_coefs[i]) * v * (1-R)

    for n in range(N):
        for m in range(n,N):
            y[m,n] = (
                p[m,n] * (lambdas[m].conjugate() + lambdas[n]) - (
                    np.dot(np.dot(b, p[:,n]), np.dot(b, p[:,m]).conjugate())
                ) + q_ricatti(n,m, par, lambdas, normal_coefs)
            )
    
    y_complex_flat = triu_to_flat(y)
    y_real_flat, y_imag_flat = np.real(y_complex_flat), np.imag(y_complex_flat)
    y_flat_raw = [*y_real_flat, *y_imag_flat]
    return y_flat_raw

def q_ricatti(n,m,*args):

    import numpy as np
    from scipy.integrate import quad
    from .eig_fun import q_ricatti_fun_mul

    par = args[0]
    lambdas = args[1]
    normal_coefs = args[2]

    q_nm = quad(q_ricatti_fun_mul, 0, 1, args=(par, (lambdas[m], lambdas[n]), (normal_coefs[m], normal_coefs[n])), complex_func=True)[0]
    
    return q_nm

def k_ricatti(x, p_flat, *args):
    
    import numpy as np
    from .eig_fun import eig_fun_adj_1, eig_fun_adj_2
    from .upper_triangular import n_triu, flat_to_triu, triu_to_flat, triu_to_symm, triu_to_hermitian
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    lambdas = args[1]
    normal_coefs = args[2]

    slicer = int(len(p_flat)/2)
    p_flat_real = p_flat[:slicer]
    p_flat_imag = p_flat[slicer:]
    p_flat_complex = p_flat_real + p_flat_imag * 1j

    p = triu_to_hermitian(flat_to_triu(p_flat_complex))
    N = p.shape[0]
    k = np.array([np.zeros_like(x)]*2, dtype=complex)
    b = np.zeros_like(lambdas)

    for i in range(len(lambdas)):
        b[i] = eig_fun_adj_1(0, par, lambdas[i], normal_coefs[i]) * v * (1-R)

    for i in range(N):
        for j in range(N):
            k[0] += p[i,j] * b[i] * eig_fun_adj_1(x, par, lambdas[j], normal_coefs[j]).conjugate()
            k[1] += p[i,j] * b[i] * eig_fun_adj_2(x, par, lambdas[j], normal_coefs[j]).conjugate()
    
    return k