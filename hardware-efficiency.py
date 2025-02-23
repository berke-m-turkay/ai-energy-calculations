import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
years = np.array([2012, 2017, 2020])
flopw = np.array([0.42, 0.78, 1.4])

# Logistic regression function
def logistic(years, E_a0, alpha_a, beta_a, t0):
    return E_a0 / (1 + np.exp(-alpha_a * (years - t0))) + beta_a

# Perform regression with bounds and initial guesses
initial_guess = [1.5, 0.1, 0.2, 2015]  # [E_a0, alpha_a, beta_a, t0]
bounds = ([0, 0, 0, 2000], [10, 5, 5, 2100])  # Bounds for parameters
par, covariance = curve_fit(logistic, years, flopw, p0=initial_guess, bounds=bounds)

# Extract fitted parameters
E_a0, alpha_a, beta_a, t0 = par

# Historical fit
years_fit = np.linspace(min(years), max(years), 1000)
eff_fit = logistic(years_fit, E_a0, alpha_a, beta_a, t0)

# Projection for 20 years into the future
future_years = np.linspace(max(years), max(years) + 20, 1000)
future_eff_fit = logistic(future_years, E_a0, alpha_a, beta_a, t0)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(years, flopw, color='red', label='Historical Data')  # Historical data points
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Historical trendline
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Future projection

# Add titles and labels
plt.title('Hardware Efficiency Over Time')
plt.xlabel('Year')
plt.ylabel('Hardware Efficiency (TFLOPs/W)')
plt.legend()
plt.grid(True)
plt.show()

# Print fitted coefficients
print(f"Fitted coefficients:\n E_a0 = {E_a0:.4f}, alpha_a = {alpha_a:.4f}, beta_a = {beta_a:.4f}, t0 = {t0:.4f}")

# Projection for the year 2024
projection_year = 2024
projected_efficiency = logistic(projection_year, E_a0, alpha_a, beta_a, t0)
print(f"Projected Hardware Efficiency in {projection_year}: {projected_efficiency:.4f} TFLOPs/W")
