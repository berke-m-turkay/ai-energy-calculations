import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# Historical data points for hardware efficiency (in TFLOPs/Watt).
# Source: Supplementary Section A.1 of the paper.
# Years correspond to key hardware releases; efficiency values reflect FLOPs per energy used.
years = np.array([2012, 2017, 2020])
flopw = np.array([0.42, 0.78, 1.4])

# --- Model Definition ---
# Define a logistic function to model hardware efficiency growth over time.
# Parameters:
#   E_a0   - maximum efficiency asymptote
#   alpha_a - growth rate
#   beta_a - minimum baseline efficiency
#   t0     - midpoint year (inflection point)
def logistic(years, E_a0, alpha_a, beta_a, t0):
    return E_a0 / (1 + np.exp(-alpha_a * (years - t0))) + beta_a

# --- Curve Fitting ---
# Fit the logistic model to historical data using non-linear least squares.
# Initial guess and parameter bounds are chosen based on empirical trends.
initial_guess = [1.5, 0.1, 0.2, 2015]  # [E_a0, alpha_a, beta_a, t0]
bounds = ([0, 0, 0, 2000], [10, 5, 5, 2100])  # Reasonable parameter ranges
par, covariance = curve_fit(logistic, years, flopw, p0=initial_guess, bounds=bounds)

# --- Extract Parameters ---
# Unpack fitted parameters to use in future projections and plotting.
E_a0, alpha_a, beta_a, t0 = par

# --- Generate Historical Trendline ---
# Create a dense set of year points between min and max year for smooth curve plotting.
years_fit = np.linspace(min(years), max(years), 1000)
eff_fit = logistic(years_fit, E_a0, alpha_a, beta_a, t0)

# --- Future Projection ---
# Extrapolate hardware efficiency trends 20 years beyond latest data point (2020–2040).
future_years = np.linspace(max(years), max(years) + 20, 1000)
future_eff_fit = logistic(future_years, E_a0, alpha_a, beta_a, t0)

# --- Plotting ---
# Visualize historical data, fitted trend, and 20-year projection.
plt.figure(figsize=(10, 6))
plt.scatter(years, flopw, color='red', label='Historical Data')  # Raw data points
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Fitted curve
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Forecast

# Annotation
plt.title('Hardware Efficiency Over Time')
plt.xlabel('Year')
plt.ylabel('Hardware Efficiency (TFLOPs/W)')
plt.legend()
plt.grid(True)
plt.show()

# --- Results ---
# Display the logistic model's best-fit parameters.
# These values are used in forecasting hardware efficiency for future years.
print(f"Fitted coefficients:\n E_a0 = {E_a0:.4f}, alpha_a = {alpha_a:.4f}, beta_a = {beta_a:.4f}, t0 = {t0:.4f}")

# Forecast hardware efficiency for 2024 using the fitted model.
projection_year = 2024
projected_efficiency = logistic(projection_year, E_a0, alpha_a, beta_a, t0)
print(f"Projected Hardware Efficiency in {projection_year}: {projected_efficiency:.4f} TFLOPs/W")
