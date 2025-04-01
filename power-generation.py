import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data Section ---
# Dataset: Total global electricity generation per year from 1990 to 2023.
# Units are in TWh (or equivalent), used to estimate the growth of global energy supply capacity.
# Source: Supplementary Section 4.1 of the paper.
years = np.array([
    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
    2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023
])
power = np.array([
    11961, 12222, 12336, 12600, 12924, 13382, 13797, 14129, 14511, 14926,
    15565, 15800, 16538, 16936, 17737, 18465, 19167, 20060, 20437, 20279,
    21591, 22279, 22834, 23469, 24076, 24315, 24984, 25738, 26783, 27147,
    27033, 28548, 29188, 29925
])

# --- Time Normalization ---
# Offset years so that t = 0 corresponds to 1990. This simplifies exponential modeling.
years_offset = years - 1990

# --- Model Definition ---
# Define a simple exponential growth function:
#   Power(t) = P0 * exp(e * t)
# Where:
#   P0 = initial value in 1990
#   e  = exponential growth rate (fractional annual increase)
def reg(years_offset, P0, e):
    return P0 * np.exp(e * years_offset)
initial_guess = [power[0], 0.01]

# --- Curve Fitting ---
# Use nonlinear least squares to fit the exponential model to historical data.
# maxfev is increased in case convergence takes longer due to data scale.
par, covariance = curve_fit(reg, years_offset, power, p0=initial_guess, maxfev=5000)

# --- Extract Parameters ---
# Fitted values for base power (P0) and exponential rate (e) are used for forecasting.
P0, e = par

# --- Trendline ---
# Create a dense time grid to plot smooth fitted curve over historical range.
years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
eff_fit = reg(years_fit_offset, P0, e)
years_fit = years_fit_offset + 1990  # Convert offset back to calendar years

# --- Visualization ---
# Plot the actual electricity generation data and the exponential trendline.
plt.scatter(years, power, color='red', label='Data')  # Observed data points
plt.plot(years_fit, eff_fit, color='blue', label='Trendline')  # Exponential fit
plt.title('Power per Year')
plt.xlabel('Year')
plt.ylabel('Power')
plt.legend()
plt.grid(True)
plt.show()

# --- Output ---
print(f"Fitted coefficients:\n P0 = {P0:.4f}, e = {e:.4f}")
