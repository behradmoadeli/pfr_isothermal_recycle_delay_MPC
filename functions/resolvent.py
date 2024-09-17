def expm_P(P, z=1, e=0):
    """
    Matrix exponential of P * (z - e).
    
    Parameters:
    ----------
    P : np.ndarray
        Input matrix.
    z : scalar (default is 1).
    e : scalar (default is 0).
    
    Returns:
    -------
    np.ndarray
        Matrix exponential of P * (z-e).
    """
    import scipy.linalg as lina
    
    return lina.expm(P * (z - e))

def Integral_Rs(x, s, par, adjoint=False):
    """
    Compute the I array for given state x, Laplace variable s, and parameters.
    
    Parameters:
    ----------
    x : np.ndarray
        State array x(z) (2D with shape (2, N_zeta)).
    s : float
        Laplace variable.
    par : dict
        Dictionary containing parameters 'k', 'v', 'D', 'tau', and 'R'.
    adjoint : bool, optional
        If True, compute I for adjoint operator (default is False).
    
    Returns:
    -------
    I : np.ndarray
        Computed I array.
    T : np.ndarray
        Precomputed matrix exponential array T.
    """
    import numpy as np
    
    N_zeta = len(x[0])
    zeta = np.linspace(0,1,N_zeta)
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    if adjoint:
        P = np.array([
            [0, 1, 0],
            [(s-k)/D, -v/D, 0],
            [0, 0, -tau * s]
        ])
    else:
        P = np.array([
            [0, 1, 0],
            [(s-k)/D, v/D, 0],
            [0, 0, tau * s]
        ])
        
    F = np.array([[expm_P(P, z, e) for e in zeta] for z in zeta])
    T = F[:,0]
    
    I = np.zeros((2, 3, 3, N_zeta))
    # Fully obtaining I, time consuming
    # for i,j in range(3):
        # for n_z, z in enumerate(zeta):
            # I[0][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * np.array(x[0,:n_z+1]), zeta[:n_z+1])
            # I[1][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * np.array(x[1,:n_z+1]), zeta[:n_z+1])
    # Compute only the necessary elements of I
    for n_z in range(N_zeta):
        I[0,0,1,n_z] = np.trapz(F[n_z,:n_z+1,0,1] * np.array(x[0,:n_z+1]), zeta[:n_z+1])
        I[1,2,2,n_z] = np.trapz(F[n_z,:n_z+1,2,2] * np.array(x[1,:n_z+1]), zeta[:n_z+1])
    I[0,0,1,-1] = np.trapz(F[-1,:,0,1] * np.array(x[0,:]), zeta)
    I[1,2,2,-1] = np.trapz(F[-1,:,2,2] * np.array(x[1,:]), zeta)
    I[0,1,1,-1] = np.trapz(F[-1,:,1,1] * np.array(x[0,:]), zeta)
    
    return I, T

def Rs(x, s, par):
    """
    Resolvent operator for the state x, Laplace variable s, and parameters.
    
    Parameters:
    ----------
    x : np.ndarray
        State array x(z) (2D with shape (2, N_zeta)).
    s : float
        Laplace variable.
    par : dict
        Dictionary containing parameters 'k', 'v', 'D', 'tau', and 'R'.
    
    Returns:
    -------
    np.ndarray
        Computed resolvent X(s, z) for state x(z, t=0).
    """
    import scipy.linalg as lina
    import numpy as np
    
    I, T = Integral_Rs(x, s, par)
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

    a = np.array([
        [-v, D, R*v],
        [T[-1,0,0], T[-1,0,1], -T[-1,2,2]],
        [T[-1,1,0], T[-1,1,1], 0]
    ])
    b = np.array([
        0,
        I[0,0,1,-1]/D - I[1,2,2,-1] * tau,
        I[0,1,1,-1]/D
    ])
    M = lina.inv(a)
    X0 = M @ b
    [X1_0, X1_prime_0, X2_0] = X0
    
    X1 = X1_0 * T[:,0,0] + X1_prime_0 * T[:,0,1] - 1/D * I[0,0,1,:]
    X2 = X2_0 * T[:,2,2] - tau * I[1,2,2,:]
    # X2[-1] = X1[-1].copy()
    X = np.array([X1, X2])
        
    return X

def A_d(x, s, par, pow=1):
    """
    Discrete-time A operator applied to state x.

    Parameters:
    ----------
    x : np.ndarray
        State array.
    s : float
        Laplace variable.
    par : dict
        Dictionary of system parameters.
    pow : int, optional
        Number of iterations (default is 1).
    
    Returns:
    -------
    np.ndarray
        All of updated states after applying the A operator.
    """
    import numpy as np
    
    X = np.zeros((2, len(x[0]), pow+1))
    X[:,:,0] = x
    
    for i in range(pow):
        X[:,:,i+1] = -X[:,:,i] + 2*s * Rs(X[:,:,i], s, par)

    return X


def B_d(zeta, s, par, u=1):
    """
    Discrete-time B operator for boundary input.
    
    Parameters:
    ----------
    zeta : np.ndarray
        Spatial discretization points.
    s : float
        Laplace variable.
    par : dict
        Dictionary of system parameters.
    u : float, optional
        Control input (default is 1).
    
    Returns:
    -------
    np.ndarray
        B_d operator as function of zeta.
    """
    import scipy.linalg as lina
    import numpy as np
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    P = np.array([
        [0, 1, 0],
        [(s-k)/D, v/D, 0],
        [0, 0, tau * s]
    ])
    T = np.array([expm_P(P, z) for z in zeta])

    a = np.array([
        [-v, D, R*v],
        [T[-1,0,0], T[-1,0,1], -T[-1,2,2]],
        [T[-1,1,0], T[-1,1,1], 0]
    ])
    M = lina.inv(a)
    b_tilde = np.array([1, -T[-1,0,1], -T[-1,1,1]])
    X0_tilde = M @ b_tilde
    [X1_0_tilde, X1_prime_0_tilde, X2_0_tilde] = X0_tilde
    
    X1 = v * (1-R) * (T[:,0,0] * X1_0_tilde + T[:,0,1] * (X1_prime_0_tilde + 1))
    X2 = v * (1-R) * (T[:,2,2] * X2_0_tilde)
    X = np.array([X1, X2])
    B_d = np.sqrt(2*s) * X
        
    return B_d * u

def Rs_adjoint(x, s, par):
    """
    Adjoint resolvent operator for the state x, Laplace variable s, parameters, and precomputed I and T arrays.
    
    Parameters:
    ----------
    x : np.ndarray
        State array x(z) (2D with shape (2, N_zeta)).
    s : float
        Laplace variable.
    par : dict
        Dictionary containing parameters 'k', 'v', 'D', 'tau', and 'R'.
    
    Returns:
    -------
    np.ndarray
        Computed resolvent X(s, z) for state x(z, t=0).
    """
    import scipy.linalg as lina
    import numpy as np
    
    I, T = Integral_Rs(x, s, par, adjoint=True)
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

    a = np.array([
        [0, 1, 0],
        [R*v, 0, 1/tau],
        [v*T[-1,0,0]+D*T[-1,1,0], v*T[-1,0,1]+D*T[-1,1,1], -T[-1,2,2]/tau]
    ])
    b = np.array([
        0,
        0,
        I[0,0,1,-1]* v/D + I[0,1,1,-1] + I[1,2,2,-1]
    ])
    M = lina.inv(a)
    X0 = M @ b
    [X1_0, X1_prime_0, X2_0] = X0
    
    X1 = X1_0 * T[:,0,0] + X1_prime_0 * T[:,0,1] - 1/D * I[0,0,1,:]
    X2 = X2_0 * T[:,2,2] + tau * I[1,2,2,:]
    X = np.array([X1, X2])
        
    return X

def A_d_adjoint(x, s, par, pow=1):
    """
    Discrete-time A* adjoint operator applied to state x.

    Parameters:
    ----------
    x : np.ndarray
        State array.
    s : float
        Laplace variable.
    par : dict
        Dictionary of system parameters.
    pow : int, optional
        Number of iterations (default is 1).
    
    Returns:
    -------
    np.ndarray
        All of updated states after applying the A operator.
    """
    import numpy as np
    
    X = np.zeros((2, len(x[0]), pow+1))
    X[:,:,0] = x
    
    for i in range(pow):
        X[:,:,i+1] = -X[:,:,i] + 2*s * Rs_adjoint(X[:,:,i], s, par)

    return X

def B_d_adjoint(x, s, par):
    """
    Discrete-time B* adjoint operator.
    
    Parameters:
    ----------
    x : np.ndarray
        State array.
    s : float
        Laplace variable.
    par : dict
        Dictionary of system parameters.
    
    Returns:
    -------
    float
        Control input.
    """
    import numpy as np
    
    zeta = np.linspace(0,1,len(x[0]))
    
    B = B_d(zeta, s, par)
    I = B[0,:] * x[0,:] + B[1,:] * x[1,:]
    X = np.trapz(I, zeta)
    
    return X

def Q_bar(x, par, lambdas, normal_coefs, q_ctrl, imag_tol=1e-8):
    
    import numpy as np
    from functions import q_riccati
    from functions import eig_fun_1, eig_fun_2, eig_fun_adj_1, eig_fun_adj_2
    
    zeta = np.linspace(0, 1, len(x[0]))
    Q_vector = np.zeros_like(x, dtype=complex)
    n_lambdas = len(lambdas)

    for i in range(n_lambdas):
        for j in range(n_lambdas):
            q_coef = -q_riccati(i,j, par, lambdas, normal_coefs, q_ctrl)/(lambdas[i]+lambdas[j].conjugate())
            q_inner = np.trapz(x[0,:] *  eig_fun_adj_1(zeta, par, lambdas[j], normal_coefs[j]) + x[1,:] *  eig_fun_adj_2(zeta, par, lambdas[j], normal_coefs[j]))
            q_vector = q_coef * q_inner * np.array([eig_fun_1(zeta, par, lambdas[i], normal_coefs[i]), eig_fun_2(zeta, par, lambdas[i], normal_coefs[i])]).reshape(x.shape)
            Q_vector += q_vector
            
    if np.max(np.imag(Q_vector)) < imag_tol:
        return np.real(Q_vector)
    else:
        raise ValueError(f"The imaginary part for Q_bar is not negligible (max imaginary part: {np.max(np.abs(Q_vector.imag))})")