import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# This dataset represents the estimated computational complexity (e.g., FLOPs) of state-of-the-art models per year.
# Complexity values are in floating-point operations (FLOPs), normalized for scaling across time.
# Notable entries (e.g., GPT-2, GPT-3, PaLM, GPT-4) reflect exponential growth.
# Source: Supplementary Section 3.2.1 of the paper.
years = np.array([2018, 2019, 2020, 2022, 2023, 2024])
complexity = np.array([0.000702, 14890, 153000, 424000, 138000000, 386000000])
peryear = complexity  # Alias for clarity

# --- Time Normalization ---
# Offset years to simplify modeling: t = 0 corresponds to 2022 (reference year).
years_offset = years - 2022

# --- Model Definition ---
# Define a modified power-law function to model algorithmic complexity growth over time.
# Formula: C(t) = C_c0 * (t + beta)^alpha
# Parameters:
#   C_c0   - base scaling coefficient
#   alpha  - growth exponent (controls curvature)
#   beta   - horizontal shift to account for model acceleration
# `np.maximum` ensures numerical stability when t + beta ≈ 0.
def power_law(years_offset, C_c0, alpha, beta):
    return C_c0 * (np.maximum(years_offset + beta, 1e-6)) ** alpha

# --- Curve Fitting ---
# Use nonlinear least squares to fit the power-law to empirical complexity values.
# Initial guesses and parameter bounds reflect plausible rates and shifts based on observed data.
initial_guess = [31.1, 2, 1]  # [C_c0, alpha, beta]
bounds = ([0, 0, 0], [np.inf, 5, np.inf])  # Cap exponent to avoid unrealistic projections

# Attempt model fitting; catch errors to facilitate debugging if fitting fails
try:
    par, covariance = curve_fit(power_law, years_offset, peryear, p0=initial_guess, bounds=bounds)
except ValueError as e:
    print(f"Error during optimization: {e}")
    raise

# --- Extract Fitted Parameters ---
# Parameters are now tuned to fit the observed growth curve.
C_c0, alpha, beta = par

# --- Historical Trendline ---
# Generate fitted curve for years 2018–2024 to visualize historical model complexity trends.
years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
eff_fit = power_law(years_fit_offset, C_c0, alpha, beta)
years_fit = years_fit_offset + 2022  # Convert offset years back to calendar years

# --- Future Projection ---
# Forecast complexity growth for 20 years beyond 2024 (up to 2044).
future_years_offset = np.linspace(2, 2 + 20, 1000)  # Offset starts at 2024 - 2022 = 2
future_eff_fit = power_law(future_years_offset, C_c0, alpha, beta)
future_years = future_years_offset + 2022  # Back to calendar years

# --- Plotting ---
# Plot historical data, fitted trendline, and long-term projection.
plt.scatter(years, peryear, color='red', label='Data')  # Raw complexity values
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Fit curve
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Extrapolation

# Annotation
plt.title('Estimated Model Complexity per Year')
plt.xlabel('Year')
plt.ylabel('Complexity per Year')
plt.legend()
plt.grid(True)
plt.show()

# --- Output ---
# Display the fitted model parameters for downstream use or interpretation.
print(f"Fitted coefficients:\n C_c0 = {C_c0:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}")
