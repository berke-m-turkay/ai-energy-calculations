import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# Dataset: Query complexity of prominent frontier models (e.g., ChatGPT, GPT-4, Gemini) from 2020 to 2024.
# These values are given in estimated PF-days or equivalent compute units, scaled for comparative modeling.
# Source: Supplementary Section 3.4 of the paper.
years = np.array([2020, 2021, 2022, 2023, 2024])
complexity = np.array([5.71, 7.36, 86.98, 525.21, 1238.19])

# --- Time Normalization ---
# Offset all years so that t = 0 aligns with 2022, aiding in curve stability and interpretation.
def preprocess_data(years, reference_year=2022):
    return years - reference_year

years_offset = preprocess_data(years)

# --- Power-Law Model Definition ---
# The core model: C(t) = C_c0 * (t + beta)^alpha
# Where:
#   C_c0  - base scale
#   alpha - growth rate (exponent)
#   beta  - time shift (horizontal translation)
# A minimum offset (1e-6) is added to avoid undefined behavior near zero.
def power_law(years_offset, C_c0, alpha, beta):
    return C_c0 * (np.maximum(years_offset + beta, 1e-6)) ** alpha

# --- Curve Fitting Procedure ---
# This function fits the power-law model to empirical complexity data.
# Initial parameter guesses and bounds reflect prior growth trends in foundation model complexity.
def fit_power_law(years_offset, complexity):
    initial_guess = [31.1, 2, 1]  # [C_c0, alpha, beta]
    bounds = ([0, 0, 0], [np.inf, 5, np.inf])  # Prevent unrealistic exponents

    try:
        params, covariance = curve_fit(power_law, years_offset, complexity, p0=initial_guess, bounds=bounds)
        print(f"Fitted coefficients:\n C_c0 = {params[0]:.4f}, alpha = {params[1]:.4f}, beta = {params[2]:.4f}")
        return params
    except ValueError as e:
        print(f"Error during power-law optimization: {e}")
        return None

# --- Model Fitting ---
# Extract best-fit parameters for power-law model from the training data.
C_c0, alpha, beta = fit_power_law(years_offset, complexity)

# --- Data Generation for Plotting ---
# Generate both historical fit and future projections using the fitted model parameters.
def generate_fitted_data(model_func, params, years_offset, reference_year=2022, future_years=20):
    # Historical fit: dense interpolation over training range
    years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
    eff_fit = model_func(years_fit_offset, *params)

    # Future projection: extend beyond latest known point by 20 years
    future_years_offset = np.linspace(max(years_offset), max(years_offset) + future_years, 1000)
    future_eff_fit = model_func(future_years_offset, *params)

    # Convert offsets back to actual calendar years for plotting
    return years_fit_offset + reference_year, eff_fit, future_years_offset + reference_year, future_eff_fit

# Generate smoothed data for plotting
years_fit, eff_fit, future_years, future_eff_fit = generate_fitted_data(power_law, [C_c0, alpha, beta], years_offset)

# --- Visualization ---
# Display raw data, fitted trendline, and 20-year forward projection on the same chart.
def plot_results(years, complexity, years_fit, eff_fit, future_years, future_eff_fit):
    plt.figure(figsize=(10, 6))
    plt.scatter(years, complexity, color='red', label='Data')  # Actual model complexities
    plt.plot(years_fit, eff_fit, color='blue', label='Power-Law Trendline')  # Historical fit
    plt.plot(future_years, future_eff_fit, color='orange', linestyle='dashed', label='Future Projection')  # Extrapolation

    # Annotation
    plt.title('Estimated Model Complexity per Year (Power Law)')
    plt.xlabel('Year')
    plt.ylabel('Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting
plot_results(years, complexity, years_fit, eff_fit, future_years, future_eff_fit)
