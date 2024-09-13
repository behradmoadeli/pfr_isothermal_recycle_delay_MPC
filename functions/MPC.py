def MPC_matrix_build(
        x, s, par, 
        lambdas, normal_coefs, q_ctrl, r_ctrl, 
        n_modes=7, N=10,
        u_limits=(-100, 100), y_limits=(-1000, 1000)
    ):
    
    import numpy as np
    from joblib import Parallel, delayed
    from functions import A_d, B_d, A_d_adjoint, B_d_adjoint, Q_bar
    
    zeta = np.linspace(0,1,len(x[0]))
    bd = B_d(zeta, s, par)
    q_bd = Q_bar(bd, par, lambdas[:n_modes], normal_coefs, q_ctrl)
    ad_bd = A_d(bd, s, par, pow=N-1)
    ad_x = A_d(x, s, par, pow=N)
    ad_adj_q_bd = A_d_adjoint(q_bd, s, par, pow=N-1)
    
    # Parallelize Q_bar calculations over the indices of ad_bd and ad_x
    q_ad_bd = np.array(Parallel(n_jobs=-1)(delayed(Q_bar)(ad_bd[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl) for i in range(len(ad_bd[0,0]))))
    q_ad_x = np.array(Parallel(n_jobs=-1)(delayed(Q_bar)(ad_x[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl) for i in range(len(ad_x[0,0]))))
    # q_ad_bd = np.array([Q_bar(ad_bd[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl) for i in range(len(ad_bd[0,0]))])
    # q_ad_x = np.array([Q_bar(ad_x[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl) for i in range(len(ad_x[0,0]))])
    
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i < j:
                H[i,j] = B_d_adjoint(ad_adj_q_bd[:,:,j-i-1], s, par)
            elif i == j:
                H[i,j] = B_d_adjoint(q_bd, s, par) + r_ctrl
            else:
                H[i,j] = B_d_adjoint(q_ad_bd[:,:,i-j-1], s, par)
    
    P = np.zeros(N)
    for i in range(N):
        P[i] = B_d_adjoint(q_ad_x[:,:,i], s, par)
    
    T = np.zeros(N)
    for i in range(N):
        T[i] = ad_x[0,-1,i]
        
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            if i == j:
                S[i,j] = bd[0,-1]
            else:
                S[i,j] = ad_bd[0,-1,i-j-1]
    
    u_min, u_max = u_limits
    y_min, y_max = y_limits
    CT = np.vstack((-S, S, np.eye(N), -np.eye(N)))
    b = np.vstack((
            np.array(T-y_max).reshape(-1,1), 
            np.array(y_min-T).reshape(-1,1), 
            u_min*np.ones((N,1)), 
            -u_max*np.ones((N,1))
        )).reshape(-1,)
    
    G = H
    a = -P
    C = CT.T
    b = b.flatten()
    meq = 0
    
    return G, a, C, b, meq