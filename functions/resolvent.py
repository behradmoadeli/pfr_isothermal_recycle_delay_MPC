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
    
    N_zeta = len(x[0])
    zeta = np.linspace(0,1,N_zeta)
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    P = np.array([
        [0, 1, 0],
        [(s-k)/D, v/D, 0],
        [0, 0, tau * s]
    ])
    T1 = expm_P(P)
    X = np.array(x)

    a = np.array([
        [-v, D, R*v],
        [T1[0,0], T1[0,1], -T1[2,2]],
        [T1[1,0], T1[1,1], 0]
    ])
    M = lina.inv(a)
    
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])
    X1 = X[0,:]
    X2 = X[1,:]
    
    F1e = np.array([expm_P(P, e=e) for e in zeta])
    I1_0 = np.array([F1e[e,0,1] * x1[e] / D - tau * F1e[e,2,2] * x2[e] for e in range(N_zeta)])
    I2_0 = np.array([tau * F1e[e,2,2] * x2[e] for e in range(N_zeta)])
    b = np.array([
        0,
        np.trapz(I1_0, zeta),
        np.trapz(I2_0, zeta)
    ])
    X0 = M @ b
    [X1[0], X2[0], X1_prime_0] = X0
    
    for i, z in enumerate(zeta[1:], start=1):
        Tz = expm_P(P, z=z)
        I1 = []
        I2 = []
        for j, e in enumerate(zeta[:i]):
            Fze = expm_P(P, z=z, e=e)
            I1.append(Fze[0,1] * x1[j])
            I2.append(Fze[2,2] * x2[j])
        X1[i] = Tz[0,0] * X1[0] + Tz[0,1] * X1_prime_0 - 1/D * np.trapz(I1, zeta[:i])
        X2[i] = Tz[2,2] * X2[0] - tau * np.trapz(I2, zeta[:i])
        
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
        Updated state after applying the A operator.
    """
    for i in range(pow):
        x = -x + 2*s * Rs(x, s, par)

    return x

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
    T1 = expm_P(P)
    X = np.array(x)

    a = np.array([
        [-v, D, R*v],
        [T1[0,0], T1[0,1], -T1[2,2]],
        [T1[1,0], T1[1,1], 0]
    ])
    M = lina.inv(a)
    
    X1 = X[0,:]
    X2 = X[1,:]
    
    b = np.array([0, T1[0,1], T1[1,1]]) * v * (1-R) / D * u
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
    
    (k, v, D, tau, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    P = np.array([
        [0, 1, 0],
        [(s-k)/D, -v/D, 0],
        [0, 0, -tau * s]
    ])
    T1 = expm_P(P)
    X = np.array(x)

    a = np.array([
        [0, 1, 0],
        [R*v, 0, 1/tau],
        [D*T1[1,0]+v*T1[0,0], D*T1[1,1]+v*T1[0,1], -1/tau*T1[2,2]]
    ])
    M = lina.inv(a)
    
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])
    X1 = X[0,:]
    X2 = X[1,:]
    
    F1e = np.array([expm_P(P, e=e) for e in zeta])
    I1_0 = np.array([(F1e[e,0,1] * v/ D + F1e[e,1,1]) * x1[e] for e in range(N_zeta)])
    I2_0 = np.array([x2[e] for e in range(N_zeta)])
    b = np.array([
        0,
        0,
        -(np.trapz(I1_0, zeta) + np.trapz(I2_0, zeta))
    ])
    X0 = M @ b
    [X1[0], X2[0], X1_prime_0] = X0
    
    for i, z in enumerate(zeta[1:], start=1):
        Tz = expm_P(P, z=z)
        I1 = []
        I2 = []
        for j, e in enumerate(zeta[:i]):
            Fze = expm_P(P, z=z, e=e)
            I1.append(Fze[0,1] * x1[j])
            I2.append(Fze[2,2] * x2[j])
        X1[i] = Tz[0,0] * X1[0] + Tz[0,1] * X1_prime_0 - 1/D * np.trapz(I1, zeta[:i])
        X2[i] = Tz[2,2] * X2[0] + tau * np.trapz(I2, zeta[:i])
        
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