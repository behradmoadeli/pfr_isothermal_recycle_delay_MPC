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

def Integral_Rs(x, s, par):
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
    
    Returns:
    -------
    I : np.ndarray
        Computed I array.
    F : np.ndarray
        Precomputed matrix exponential array F.
    T : np.ndarray
        Precomputed matrix exponential array T.
    """
    import numpy as np
    
    N_zeta = len(x[0])
    zeta = np.linspace(0,1,N_zeta)
    
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
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
            # I[0][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * x1, zeta[:n_z+1])
            # I[1][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * x2, zeta[:n_z+1])
    # Compute only the necessary elements of I
    for n_z in range(N_zeta):
        I[0,0,1,n_z] = np.trapz(F[n_z,:n_z+1,0,1] * x1, zeta[:n_z+1])
        I[1,2,2,n_z] = np.trapz(F[n_z,:n_z+1,2,2] * x2, zeta[:n_z+1])
    I[0,1,1,-1] = np.trapz(F[-1,:,1,1] * x1, zeta)
    
    return I, T

def Rs(x, par, I, T):
    """
    Resolvent operator for the state x, Laplace variable s, parameters, and precomputed I and T arrays.
    
    Parameters:
    ----------
    x : np.ndarray
        State array x(z) (2D with shape (2, N_zeta)).
    par : dict
        Dictionary containing parameters 'k', 'v', 'D', 'tau', and 'R'.
    I : np.ndarray
        Precomputed integral array I.
    T : np.ndarray
        Precomputed matrix exponential array T.
    
    Returns:
    -------
    np.ndarray
        Computed resolvent X(s, z) for state x(z, t=0).
    """
    import scipy.linalg as lina
    import numpy as np
        
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])
    X1 = np.zeros_like(x1)
    X2 = np.zeros_like(x2)
    
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
    [X1[0], X1_prime_0, X2[0]] = X0
    
    X1 = X1[0] * T[:,0,0] + X1_prime_0 * T[:,0,1] - 1/D * I[0,0,1,:]
    X2 = X2[0] * T[:,2,2] - tau * I[1,2,2,:]
    X = np.array([X1, X2])
        
    return X

def A_d(x, s, par, I, T, pow=1):
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
    I : np.ndarray
        Precomputed integral array I.
    T : np.ndarray
        Precomputed matrix exponential array T.
    pow : int, optional
        Number of iterations (default is 1).
    
    Returns:
    -------
    np.ndarray
        Updated state after applying the A operator.
    """
    for _ in range(pow):
        x = -x + 2*s * Rs(x, par, I, T)

    return x
#________________________________________________
# Fix below this line
# Fix Bd functions of zeta
# Introduce I for adjoint
# get rid of unnecessary I calculations for adjoint
# Reflect changes to where functions are called (such as in Ad adjoint)
#________________________________________________
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
        Computed B operator for the control input.
    """
    import scipy.linalg as lina
    import numpy as np
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    P = np.array([
        [0, 1, 0],
        [(s-k)/D, v/D, 0],
        [0, 0, tau * s]
    ])
    T[-1] = expm_P(P)
    X = np.array(x)

    a = np.array([
        [-v, D, R*v],
        [T[-1,0,0], T[-1,0,1], -T[-1,2,2]],
        [T[-1,1,0], T[-1,1,1], 0]
    ])
    M = lina.inv(a)
    
    X1 = X[0,:]
    X2 = X[1,:]
    
    b = np.array([0, T[-1,0,1], T[-1,1,1]]) * v * (1-R) / D * u
    X0 = M @ b
    [X1[0], X2[0], X1_prime_0] = X0
    
    for i, z in enumerate(zeta[1:], start=1):
        Tz = expm_P(P, z=z)
        X1[i] = Tz[0,0] * X1[0] + Tz[0,1] * X1_prime_0 - v * (1-R) / D * Tz[0,1] * u
        X2[i] = Tz[2,2] * X2[0]
        
    X = np.array([X1, X2])
        
    return np.sqrt(2*s) * X

def Rs_adjoint(x, s, par):
    """
    Adjoint resolvent operator for the state x, Laplace variable s, and parameters.
    
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
    
    N_zeta = len(x[0])
    zeta = np.linspace(0,1,N_zeta)
    
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])
    X1 = np.zeros_like(x1)
    X2 = np.zeros_like(x2)
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    P = np.array([
        [0, 1, 0],
        [(s-k)/D, -v/D, 0],
        [0, 0, -s * tau]
    ])
    F = np.array([[expm_P(P, z, e) for e in zeta] for z in zeta])
    T = F[:,0]
    
    I = np.zeros(2,3,3,N_zeta)
    for i,j in range(3):
        for n_z, z in enumerate(zeta):
            I[0][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * x1, zeta[:n_z+1])
            I[1][i,j] = np.trapz(F[n_z,:n_z+1,i,j] * x2, zeta[:n_z+1])

    a = np.array([
        [0, 1, 0],
        [R*v, 0, 1/tau],
        [D*T[-1,1,0]+v*T[-1,0,0], D*T[-1,1,1]+v*T[-1,0,1], -1/tau*T[-1,2,2]]
    ])
    b = np.array([
        0,
        0,
        v/D * I[0,0,1,-1] + I[0,1,1,-1] + I[1,2,2,-1]
    ])
    M = lina.inv(a)
    X0 = M @ b
    [X1[0], X1_prime_0, X2[0]] = X0
    
    X1 = X1[0] * T[:,0,0] + X1_prime_0 * T[:,0,1] - 1/D * I[0,0,1,:]
    X2 = X2[0] * T[:,2,2] + tau * I[1,2,2,:]
        
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
        Updated state after applying the A operator.
    """
    for i in range(pow):
        x = -x + 2*s * Rs_adjoint(x, s, par)

    return x

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
    
    N_zeta = len(x[0])
    zeta = np.linspace(0,1,N_zeta)
    
    B = B_d(zeta, s, par)
    [B1, B2] = B
    I = B1 * x[0] + B2 * x[1]
    X = np.trapz(I, zeta)
    
    return X