import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
years = np.array([2009, 2010, 2011, 2012, 2013, 2015, 2017, 2018, 2020, 2022, 2023])
unsolved = np.array([1655, 1511, 1343, 1265, 1134, 939, 575, 385, 319, 202, 173])
eff_unscaled = (10000 - unsolved) / 10000

# Logistic regression function (flexible asymptotes)
def logistic(years, E_a0, E_a1, alpha_a, beta_a):
    return E_a0 + E_a1 / (1 + np.exp(-alpha_a * (years - beta_a)))

# Perform regression with new bounds
initial_guess = [1.0, 0.2, 0.5, 2015]  # Initial guesses: [E_a0, E_a1, alpha_a, beta_a]
bounds = ([0, 0, 0, 2000], [2, 1, 5, 2100])  # Bounds for parameters
par, covariance = curve_fit(logistic, years, eff_unscaled, p0=initial_guess, bounds=bounds)

# Extract fitted parameters
E_a0, E_a1, alpha_a, beta_a = par

# Historical fit
years_fit = np.linspace(min(years), max(years), 1000)
eff_fit = logistic(years_fit, E_a0, E_a1, alpha_a, beta_a)

# Projection for 20 years into the future
future_years = np.linspace(max(years), max(years) + 20, 1000)
future_eff_fit = logistic(future_years, E_a0, E_a1, alpha_a, beta_a)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(years, eff_unscaled, color='red', label='Historical Data')  # Historical data points
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Historical trendline
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Future projection

# Add titles and labels
plt.title('Algorithm Efficiency')
plt.xlabel('Year')
plt.ylabel('Algorithm Efficiency')
plt.legend()
plt.grid(True)
plt.show()

# Print fitted coefficients
print(f"Fitted coefficients:\n E_a0 = {E_a0:.4f}, E_a1 = {E_a1:.4f}, alpha_a = {alpha_a:.4f}, beta_a = {beta_a:.4f}")

# Display the effectiveness in 20 years
final_year = max(years) + 20
final_effectiveness = logistic(final_year, E_a0, E_a1, alpha_a, beta_a)
print(f"Projected Algorithm Usage Effectiveness in {final_year:.0f}: {final_effectiveness:.4f}")
