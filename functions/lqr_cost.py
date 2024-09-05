def lqr_cost(x_values, u, q_coef, r_coef, t_eval, N_zeta):
    """
    Calculate the LQR cost function using the trapezoidal rule for both spatial and time integration.
    
    Parameters:
    - x_values: 2D numpy array of shape (n_states, n_time_steps)
    - u: 1D numpy array of shape (n_time_steps,)
    - q_coef: Coefficient for the Q matrix (state weighting)
    - r_coef: Coefficient for the R scalar (input weighting)
    - t_eval: 1D numpy array of time points (n_time_steps,)
    - N_zeta: Number of spatial points
    
    Returns:
    - J: Scalar value of the total cost function
    """
    import numpy as np
    
    n_time_steps = x_values.shape[1]  # Number of time steps
    zeta = np.linspace(0, 1, N_zeta)  # Spatial domain
    
    # Define Q and R
    Q = np.identity(N_zeta) * q_coef
    R = r_coef  # Scalar value for R
    u = u.T  # Ensure that u is a column vector
    
    integrand = np.zeros(n_time_steps)
    
    for k in range(n_time_steps):
        x_k = x_values[:, k]  # State vector at time step k
        x1_k = x_k[:N_zeta]
        x2_k = x_k[N_zeta:]
        
        # Compute the spatial integral using trapezoidal rule
        spatial_integral_1 = np.trapz(x1_k * (Q @ x1_k), zeta)
        spatial_integral_2 = np.trapz(x2_k * (Q @ x2_k), zeta)
        
        # Ensure that the integrals are scalar values
        spatial_integral = float(spatial_integral_1 + spatial_integral_2)
        
        u_k = u[k]  # Input at time step k
        integrand[k] = spatial_integral + R * u_k**2  # Cost function for time step k
    
    # Use trapezoidal rule to integrate over time
    J = np.trapz(integrand, t_eval)  # t_eval provides the time intervals
    
    return J
