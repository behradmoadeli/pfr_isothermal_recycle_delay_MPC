def MPC_matrix_build(
        x, s, par, 
        lambdas, normal_coefs, q_ctrl, G, S, S_u,
        n_modes=7, N=10,
        u_limits=(-100, 100), y_limits=(-1000, 1000),
        meq=1
    ):
    
    import numpy as np
    from functions import A_d, B_d_adjoint, Q_bar, eig_fun_1, eig_fun_2
    
    ad_x = A_d(x, s, par, pow=N)
    
    P = np.zeros(N)
    q_ad_x = np.zeros_like(ad_x)
    T = np.zeros(N)
    for i in range(N):
        P[i] = B_d_adjoint(q_ad_x[:,:,i], s, par)
        q_ad_x[:,:,i] = Q_bar(ad_x[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl)
        T[i] = ad_x[0,-1,i]
    
    zeta = np.linspace(0,1,len(x[0]))
    T_u = np.trapz(
        ad_x[0,:,-1] * eig_fun_1(zeta, par, lambdas[0], normal_coefs[0]) +
        ad_x[1,:,-1] * eig_fun_2(zeta, par, lambdas[0], normal_coefs[0]),
        zeta)
    if np.max(np.imag(T_u)) > 1e-6:
        raise ValueError(f"The imaginary part for T_u is not negligible (max imaginary part: {np.max(np.abs(T_u.imag))})")
    else:
        T_u =  np.real(T_u)
            
    u_min, u_max = u_limits
    y_min, y_max = y_limits
    N_ctrb = int(np.ceil(s/2/par['v']+1))
    S_ctrb = S[N_ctrb:,:].copy()
    T_ctrb = T[N_ctrb:].copy()

    U_min = u_min*np.ones((N,1)) #- 1e-6
    # U_min[2:] *= 2
    U_max = -u_max*np.ones((N,1)) #+ 1e-6
    # U_max[2:] *= 2

    CT = np.vstack((S_u, -S_ctrb, S_ctrb, np.eye(N), -np.eye(N)))
    b = np.vstack((
            -np.array(T_u).reshape(-1,1),
            np.array(T_ctrb-y_max).reshape(-1,1), 
            np.array(y_min-T_ctrb).reshape(-1,1), 
            U_min, 
            U_max
        )).reshape(-1,)
    
    if meq == 0:
        CT = CT[1:,:]
        b = b[1:]
    a = -P
    C = CT.T
    b = b.flatten()
    
    return G, a, C, b, meq

def G_S_Su(x, s, par, q_ctrl, N, r_ctrl, lambdas, n_modes, normal_coefs):
    import numpy as np
    from functions import A_d, B_d, A_d_adjoint, B_d_adjoint, Q_bar, eig_fun_1, eig_fun_2
    
    bd = B_d(x, s, par)
    ad_bd = A_d(bd, s, par, pow=N-1)
    q_bd = Q_bar(bd, par, lambdas[:n_modes], normal_coefs, q_ctrl)
    ad_adj_q_bd = A_d_adjoint(q_bd, s, par, pow=N-1)

    G_0 = np.zeros((N, N))
    S = np.zeros((N,N))
    q_ad_bd = np.zeros_like(ad_bd)
    ad_bd_phi = np.zeros(N, dtype=complex)
    
    for i in range(N):
        S[i,i] = bd[0,-1]
        q_ad_bd[:,:,i] = Q_bar(ad_bd[:,:,i], par, lambdas[:n_modes], normal_coefs, q_ctrl)
        ad_bd_phi[i] = np.trapz(
            ad_bd[0,:,i]*eig_fun_1(x, par, lambdas[0], normal_coefs[0]) + 
            ad_bd[0,:,i]*eig_fun_2(x, par, lambdas[0], normal_coefs[0])
            , x)
        for j in range(N):
            if i < j:
                G_0[i,j] = B_d_adjoint(ad_adj_q_bd[:,:,j-i-1], s, par)
            elif i == j:
                G_0[i,j] = B_d_adjoint(q_bd, s, par) + r_ctrl
            else:
                S[i,j] = ad_bd[0,-1,i-j-1]
                G_0[i,j] = B_d_adjoint(q_ad_bd[:,:,i-j-1], s, par)
    
    S_u = ad_bd_phi[::-1]
    if np.max(np.imag(S_u)) > 1e-6:
        raise ValueError(f"The imaginary part for S_u is not negligible (max imaginary part: {np.max(np.abs(S_u.imag))})")
    else:
        return G_0, S, np.real(S_u)

                
def G_r(G, r):
    import numpy as np
    G_r = G + r * np.eye(*G.shape)
    return G_r

def is_positive_definite(G):
    import numpy as np
    from scipy.linalg import cholesky
    
    try:
        # Attempt Cholesky decomposition to check positive definiteness
        _ = cholesky(G)
        return True
    except np.linalg.LinAlgError:
        return False
    
def find_smallest_r(Gr_0, r0, r_min_orig, r_max_orig, atol=0.01, rtol=0.01):
    import numpy as np
    G_0 = G_r(Gr_0, -r0)
    if is_positive_definite(G_0):
        return 0
    while not is_positive_definite(G_r(G_0, r_max_orig)):
        r_max_orig *= 1.5
        # if r_max_orig > 1e6:
        #     raise ValueError("r_max_orig is too large")
        
    r_min, r_max = r_min_orig, r_max_orig
    G = G_0
    old_r = 0
    while True:
        r_mid = (r_min + r_max) / 2
        new_r = r_mid
        G = G_r(G, new_r - old_r)
        old_r = new_r
        if is_positive_definite(G):
            # print(f'Found positive definite matrix for r = {r_mid}')
            r_max = r_mid  # Narrow the range to the lower half
        else:
            # print(f'Found non-positive definite matrix for r = {r_mid}')
            r_min = r_mid  # Narrow the range to the upper half
        # print(f'r_min = {r_min}, r_max = {r_max}')
        # Calculate absolute and relative differences
        abs_diff = r_max - r_min
        rel_diff = abs_diff / r_mid
        
        # Check if both tolerances are satisfied
        if abs_diff < atol and rel_diff < rtol:
            break
    
    return r_mid

# def find_smallest_r(Gr_0, r0, r_min_orig, r_max_orig, tol=0.01):
#     import numpy as np
#     G_0 = G_r(Gr_0, -r0)
#     while not is_positive_definite(G_r(G_0, r_max_orig)):
#         r_max_orig = 2*r_max_orig
#         if r_max_orig > 1e6:
#             raise ValueError("r_max_orig is too large")
        
#     r_min, r_max = r_min_orig, r_max_orig
#     G = G_0
#     old_r = 0
#     while r_max - r_min > tol:
#         r_mid = (r_min + r_max) / 2
#         new_r = r_mid
#         G = G_r(G, new_r - old_r)
#         old_r = new_r
#         if is_positive_definite(G):
#             # print(f'Found positive definite matrix for r = {r_mid}')
#             r_max = r_mid  # Narrow the range to the lower half
#         else:
#             # print(f'Found non-positive definite matrix for r = {r_mid}')
#             r_min = r_mid  # Narrow the range to the upper half
#         # print(f'r_min = {r_min}, r_max = {r_max}')
#         if r_max - r_min < tol and np.isclose(r_mid, r_min_orig, atol=tol):
#             r_min = 0.1 * r_min_orig
#             tol = 0.1 * tol
#             if r_min < 1e-6:
#                 raise ValueError("r_min is too small")
    
#     return r_mid