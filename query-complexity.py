import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
years = np.array([2020, 2021, 2022, 2023, 2024])
complexity = np.array([5.71, 7.36, 86.98, 525.21, 1238.19])

# Offset the years to start from t=0 for 2022
def preprocess_data(years, reference_year=2022):
    return years - reference_year

years_offset = preprocess_data(years)

# Power-Law Model
def power_law(years_offset, C_c0, alpha, beta):
    return C_c0 * (np.maximum(years_offset + beta, 1e-6)) ** alpha  # Ensure safe computation

# Fit Power-Law Model
def fit_power_law(years_offset, complexity):
    initial_guess = [31.1, 2, 1]  # [Initial value, growth exponent, time shift]
    bounds = ([0, 0, 0], [np.inf, 5, np.inf])
    
    try:
        params, covariance = curve_fit(power_law, years_offset, complexity, p0=initial_guess, bounds=bounds)
        print(f"Fitted coefficients:\n C_c0 = {params[0]:.4f}, alpha = {params[1]:.4f}, beta = {params[2]:.4f}")
        return params
    except ValueError as e:
        print(f"Error during power-law optimization: {e}")
        return None

# Get fitted parameters
C_c0, alpha, beta = fit_power_law(years_offset, complexity)

# Generate fitted data for visualization
def generate_fitted_data(model_func, params, years_offset, reference_year=2022, future_years=20):
    years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
    eff_fit = model_func(years_fit_offset, *params)
    
    future_years_offset = np.linspace(max(years_offset), max(years_offset) + future_years, 1000)
    future_eff_fit = model_func(future_years_offset, *params)

    return years_fit_offset + reference_year, eff_fit, future_years_offset + reference_year, future_eff_fit

years_fit, eff_fit, future_years, future_eff_fit = generate_fitted_data(power_law, [C_c0, alpha, beta], years_offset)

# Plot Results
def plot_results(years, complexity, years_fit, eff_fit, future_years, future_eff_fit):
    plt.figure(figsize=(10, 6))
    plt.scatter(years, complexity, color='red', label='Data')
    plt.plot(years_fit, eff_fit, color='blue', label='Power-Law Trendline')
    plt.plot(future_years, future_eff_fit, color='orange', linestyle='dashed', label='Future Projection')

    plt.title('Estimated Model Complexity per Year (Power Law)')
    plt.xlabel('Year')
    plt.ylabel('Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(years, complexity, years_fit, eff_fit, future_years, future_eff_fit)
