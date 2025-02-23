import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data:
years = np.array([2018,2019,2020,2021,2022,2023,2024])
models = np.array([2.25, 4.00, 3.43, 1.99, 15.70, 88.95, 183.49])
peryear = models

# Offset the years to start from t=0 for 2022
years_offset = years - 2022

# Modified power-law model with safe base
def power_law(years_offset, C_m0, eta_m, lambda_m):
    return C_m0 * (np.maximum(years_offset + lambda_m, 1e-6)) ** eta_m

# Initial guesses and bounds
initial_guess = [31.1, 2, 1]
bounds = ([0, 0, 0], [np.inf, 5, np.inf])

# Perform the regression
try:
    par, covariance = curve_fit(power_law, years_offset, peryear, p0=initial_guess, bounds=bounds)
except ValueError as e:
    print(f"Error during optimization: {e}")
    raise

# Extract the fitted parameters
C_m0, eta_m, lambda_m = par

# Create a smooth curve for the historical data (up to 2024)
years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
eff_fit = power_law(years_fit_offset, C_m0, eta_m, lambda_m)
years_fit = years_fit_offset + 2022  # Convert back to years for plotting

# Projection
future_years_offset = np.linspace(2, 2 + 20, 1000)  # Start offset at 2 (2024 - 2022)
future_eff_fit = power_law(future_years_offset, C_m0, eta_m, lambda_m)
future_years = future_years_offset + 2022  # Convert back to years for plotting

# Plot
plt.scatter(years, peryear, color='red', label='Data')  # Historical data points
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Historical trendline
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Future projection

plt.title('Estimated Model Demand per Year')
plt.xlabel('Year')
plt.ylabel('Demand per Year')
plt.legend()
plt.grid(True)
plt.show()

# Print fitted coefficients
print(f"Fitted coefficients:\n C_m0 = {C_m0:.4f}, eta_m = {eta_m:.4f}, lambda_m = {lambda_m:.4f}")
