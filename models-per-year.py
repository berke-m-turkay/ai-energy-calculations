import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# This dataset reflects the estimated number of large-scale models introduced or deployed each year.
# Values are expressed in the number of models released per year.
# Source: Supplementary Section 3.2.2 of the paper.
years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024])
models = np.array([2.25, 4.00, 3.43, 1.99, 15.70, 88.95, 183.49])
peryear = models  # For clarity — refers to model count/demand per year

# --- Time Normalization ---
# To simplify fitting and improve numerical stability, offset years so that t = 0 corresponds to 2022.
# This allows better interpretation of the time exponent in the power-law fit.
years_offset = years - 2022

# --- Model Definition ---
# Define a modified power-law function with offset and safe base:
#   peryear = C_m0 * (t + lambda_m)^eta_m
# where:
#   C_m0     - base scaling factor
#   eta_m    - power-law growth rate
#   lambda_m - horizontal offset to avoid divergence at t = 0
# The `np.maximum(..., 1e-6)` ensures numerical stability near zero.
def power_law(years_offset, C_m0, eta_m, lambda_m):
    return C_m0 * (np.maximum(years_offset + lambda_m, 1e-6)) ** eta_m

# --- Parameter Estimation ---
# Fit the power-law model to historical data using nonlinear least squares.
# Initial guesses and bounds are based on observed explosive growth in recent years.
initial_guess = [31.1, 2, 1]  # [C_m0, eta_m, lambda_m]
bounds = ([0, 0, 0], [np.inf, 5, np.inf])  # Growth rate (eta_m) capped to prevent runaway estimates

# Fit the model; catch errors if the optimizer fails.
try:
    par, covariance = curve_fit(power_law, years_offset, peryear, p0=initial_guess, bounds=bounds)
except ValueError as e:
    print(f"Error during optimization: {e}")
    raise

# --- Extract Parameters ---
# Best-fit parameters describe the growth trajectory of annual model demand.
C_m0, eta_m, lambda_m = par

# --- Historical Trendline ---
# Generate fitted curve over the observed years for smooth plotting.
years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
eff_fit = power_law(years_fit_offset, C_m0, eta_m, lambda_m)
years_fit = years_fit_offset + 2022  # Convert back to actual calendar years

# --- Future Projection ---
# Predict model demand from 2024 to 2044 using the fitted power-law model.
future_years_offset = np.linspace(2, 2 + 20, 1000)
future_eff_fit = power_law(future_years_offset, C_m0, eta_m, lambda_m)
future_years = future_years_offset + 2022  # Convert back to actual years

# --- Plotting ---
# Visualize data points, fitted trend, and future projection.
plt.scatter(years, peryear, color='red', label='Data')  # Actual model counts
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Fitted trend
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Forecast

# Add plot metadata
plt.title('Estimated Model Demand per Year')
plt.xlabel('Year')
plt.ylabel('Demand per Year')
plt.legend()
plt.grid(True)
plt.show()

# --- Output ---
# Print fitted power-law parameters to examine scaling trends.
print(f"Fitted coefficients:\n C_m0 = {C_m0:.4f}, eta_m = {eta_m:.4f}, lambda_m = {lambda_m:.4f}")
