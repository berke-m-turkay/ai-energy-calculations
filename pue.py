import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# This dataset tracks improvements in Power Usage Effectiveness (PUE) from 2007 to 2022.
# PUE measures how efficiently a data center uses energy; lower values are better.
# Source: Supplementary Section 2.3 of the paper.
years = np.array([2007, 2008, 2009, 2010, 2011, 2013, 2016, 2018, 2020, 2022])
pue = np.array([2.5, 2.4, 2.2, 2.0, 1.8, 1.6, 1.5, 1.6, 1.6, 1.5])

# Convert PUE to efficiency values for modeling purposes.
# Efficiency is defined as the inverse of PUE: eff = 1 / PUE
eff = 1 / pue

# --- Model Definition ---
# A generalized logistic function with flexible asymptotes is used to fit the trend.
# Parameters:
#   E_a0    - lower asymptote (baseline efficiency)
#   E_a1    - range between upper and lower asymptotes
#   alpha_a - growth rate
#   beta_a  - midpoint year (inflection point)
def logistic(years, E_a0, E_a1, alpha_a, beta_a):
    return E_a0 + E_a1 / (1 + np.exp(-alpha_a * (years - beta_a)))

# --- Curve Fitting ---
# Perform nonlinear regression to fit the logistic model to historical efficiency data.
# Initial guesses and bounds are based on expected ranges from past PUE observations.
initial_guess = [0.4, 0.3, 0.1, 2015]  # [E_a0, E_a1, alpha_a, beta_a]
bounds = ([0, 0, 0, 2000], [1, 1, 5, 2100])  # Logical bounds for parameters
par, covariance = curve_fit(logistic, years, eff, p0=initial_guess, bounds=bounds)

# --- Extract Parameters ---
# Unpack fitted values for later use in plotting and forecasting.
E_a0, E_a1, alpha_a, beta_a = par

# --- Historical Trendline ---
# Generate a smooth curve for the historical years to visualize model fit.
years_fit = np.linspace(min(years), max(years), 1000)
eff_fit = logistic(years_fit, E_a0, E_a1, alpha_a, beta_a)

# --- Future Projection ---
# Forecast efficiency gains 20 years beyond the latest recorded year (2022–2042).
future_years = np.linspace(max(years), max(years) + 20, 1000)
future_eff_fit = logistic(future_years, E_a0, E_a1, alpha_a, beta_a)

# --- Plotting ---
# Plot both the empirical data and modeled trends (past and projected).
plt.figure(figsize=(10, 6))
plt.scatter(years, eff, color='red', label='Historical Data')  # Actual PUE-derived efficiency values
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Model fit for past data
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Forward projection

# Annotatation
plt.title('Hardware Efficiency Over Time')
plt.xlabel('Year')
plt.ylabel('Hardware Efficiency')
plt.legend()
plt.grid(True)
plt.show()

# --- Results ---
# Print the fitted logistic parameters, which define the historical + projected curve.
print(f"Fitted coefficients:\n E_a0 = {E_a0:.4f}, E_a1 = {E_a1:.4f}, alpha_a = {alpha_a:.4f}, beta_a = {beta_a:.4f}")

# Project hardware efficiency.
final_year = max(years) + 20
final_effectiveness = logistic(final_year, E_a0, E_a1, alpha_a, beta_a)
print(f"Projected Hardware Efficiency in {final_year:.0f}: {final_effectiveness:.4f}")
