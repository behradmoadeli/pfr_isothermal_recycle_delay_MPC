import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

os.chdir("/Users/behradmoadeli/Documents/PhD/behrads_papers/reports")
path = "CSV/default_positive_imag.csv"

df, label, metadata = plot_single_df(path, filter=False)
n_lambdas = len(df)
print(df.head(n_lambdas))

# Define the exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit the model
popt, pcov = curve_fit(exponential_func, df['Sol_r'], df['Sol_i'])

# Predict new values
X_new = np.linspace(-20, 0, 100)
y_new = exponential_func(X_new, *popt)

# Plot the results
# plt.scatter(df['Sol_r'], df['Sol_i'], label='Original data')
# plt.plot(X_new, y_new, color='red', label='Exponential fit')
# plt.xlabel('Sol_r')
# plt.ylabel('Sol_i')
# plt.title('Exponential Regression Fit')
# plt.legend()
# plt.show()

# Predict values using the fitted model
y_pred = exponential_func(df['Sol_r'], *popt)

# Calculate R²
residuals = df['Sol_i'] - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df['Sol_i'] - np.mean(df['Sol_i']))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f'R²: {r_squared:.4f}')

# plt.scatter(df['Sol_r'], residuals)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel('Sol_r')
# plt.ylabel('Residuals')
# plt.title('Residuals of Exponential Fit')
# plt.show()

print(f'[{popt[0]:.3f}, {popt[1]:.5e}, {popt[2]:.3f}]')