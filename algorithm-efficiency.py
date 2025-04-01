import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# This dataset measures algorithmic efficiency improvements from 2009 to 2023.
# Raw data comes from the number of unsolved instances on the MATH benchmark (out of 10,000 total).
# As fewer problems remain unsolved over time, efficiency is inferred to be increasing.
# Source: Supplementary Section A.3 of the paper.
years = np.array([2009, 2010, 2011, 2012, 2013, 2015, 2017, 2018, 2020, 2022, 2023])
unsolved = np.array([1655, 1511, 1343, 1265, 1134, 939, 575, 385, 319, 202, 173])

# Normalize data to express efficiency on a 0–1 scale
# Formula: efficiency = (10,000 - unsolved) / 10,000
eff_unscaled = (10000 - unsolved) / 10000

# --- Model Definition ---
# A flexible logistic function is used to model the nonlinear improvement in efficiency.
# Parameters:
#   E_a0    - lower bound of the logistic function (baseline efficiency)
#   E_a1    - range (upper asymptote minus baseline)
#   alpha_a - slope or rate of growth
#   beta_a  - inflection point (year where growth is fastest)
def logistic(years, E_a0, E_a1, alpha_a, beta_a):
    return E_a0 + E_a1 / (1 + np.exp(-alpha_a * (years - beta_a)))

# --- Curve Fitting ---
# Fit the logistic function to the historical algorithm efficiency data.
# Initial guesses and bounds are empirically derived to reflect plausible efficiency limits.
initial_guess = [1.0, 0.2, 0.5, 2015]  # [E_a0, E_a1, alpha_a, beta_a]
bounds = ([0, 0, 0, 2000], [2, 1, 5, 2100])  # Physically/empirically reasonable ranges
par, covariance = curve_fit(logistic, years, eff_unscaled, p0=initial_guess, bounds=bounds)

# --- Extract Parameters ---
# The optimized parameters from curve fitting will be used for prediction and visualization.
E_a0, E_a1, alpha_a, beta_a = par

# --- Historical Trendline Generation ---
# Create a smooth curve using the fitted model over the observed time period.
years_fit = np.linspace(min(years), max(years), 1000)
eff_fit = logistic(years_fit, E_a0, E_a1, alpha_a, beta_a)

# --- Future Projection ---
# Forecast algorithm efficiency 20 years into the future (up to 2043).
future_years = np.linspace(max(years), max(years) + 20, 1000)
future_eff_fit = logistic(future_years, E_a0, E_a1, alpha_a, beta_a)

# --- Plotting ---
# Plot historical data, fitted logistic curve, and forward projection.
plt.figure(figsize=(10, 6))
plt.scatter(years, eff_unscaled, color='red', label='Historical Data')  # Raw normalized efficiency
plt.plot(years_fit, eff_fit, color='blue', label='Historical Trendline')  # Fitted model curve
plt.plot(future_years, future_eff_fit, color='orange', label='Future Projection')  # Forecast curve

# Annotation
plt.title('Algorithm Efficiency')
plt.xlabel('Year')
plt.ylabel('Algorithm Efficiency')
plt.legend()
plt.grid(True)
plt.show()

# --- Results ---
# Display the estimated parameters from logistic regression.
print(f"Fitted coefficients:\n E_a0 = {E_a0:.4f}, E_a1 = {E_a1:.4f}, alpha_a = {alpha_a:.4f}, beta_a = {beta_a:.4f}")

# --- Projection ---
final_year = max(years) + 20
final_effectiveness = logistic(final_year, E_a0, E_a1, alpha_a, beta_a)
print(f"Projected Algorithm Usage Effectiveness in {final_year:.0f}: {final_effectiveness:.4f}")
