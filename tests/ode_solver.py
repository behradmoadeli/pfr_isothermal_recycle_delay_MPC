import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lotka-Volterra ODE system
def lotka_volterra(t, y, a, b, c, d):
    y1, y2 = y
    dy1_dt = a * y1 - b * y1 * y2
    dy2_dt = -c * y2 + d * y1 * y2
    return [dy1_dt, dy2_dt]

# Set the parameters
a = 0.1
b = 0.02
c = 0.3
d = 0.01

# Set the initial conditions
y0 = [5, 10]  # Initial prey and predator populations

# Set the time span
t_span = (0, 200)  # Time span from 0 to 200

# Solve the ODEs using solve_ivp
sol = solve_ivp(lotka_volterra, t_span, y0, args=(a, b, c, d), t_eval=np.arange(0, 200, 0.1))

# Access the solution
time_points = sol.t
y1_values = sol.y[0]
y2_values = sol.y[1]

# Visualize the solution
plt.figure()
plt.plot(time_points, y1_values, label='Prey (y1)')
plt.plot(time_points, y2_values, label='Predator (y2)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Lotka-Volterra Predator-Prey Model')
plt.show()
