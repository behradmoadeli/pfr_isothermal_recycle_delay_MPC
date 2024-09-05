# Example usage:
# Define parameters for the example
N_zeta = 150  # Number of spatial points
n_time_steps = 200  # Number of time steps
q_coef = 0.05
r_coef = 50

# Generate example data
x_values = np.random.rand(2 * N_zeta, n_time_steps)  # Example state vectors
u = np.random.rand(n_time_steps)  # Example control inputs
t_eval = np.linspace(0, 5, n_time_steps)  # Example time vector

# Calculate the cost
J = lqr_cost(x_values, u, q_coef, r_coef, t_eval, N_zeta)
J &#8203;:contentReference[oaicite:0]{index=0}&#8203;