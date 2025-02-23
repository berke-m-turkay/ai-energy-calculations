import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data:
years = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,2023])
power = np.array([11961, 12222, 12336, 12600, 12924, 13382, 13797, 14129, 14511, 14926, 15565, 15800, 16538, 16936, 17737, 18465, 19167, 20060, 20437, 20279, 21591, 22279, 22834, 23469, 24076, 24315, 24984, 25738, 26783, 27147, 27033, 28548, 29188, 29925])

# Offset the years to start from t=0 for 1990 (or another starting year)
years_offset = years - 1990

# Regression function:
def reg(years_offset, P0, e):
    return P0 * np.exp(e * years_offset)

# Initial guess for the parameters
initial_guess = [power[0], 0.01]  # Start with the initial power and a reasonable growth rate

# Regression using curve_fit:
par, covariance = curve_fit(reg, years_offset, power, p0=initial_guess, maxfev=5000)

# Extract the fitted parameters:
P0, e = par

# Create a smooth curve for plotting the fitted regression line
years_fit_offset = np.linspace(min(years_offset), max(years_offset), 1000)
eff_fit = reg(years_fit_offset, P0, e)

# Convert years for plot (adding back the offset)
years_fit = years_fit_offset + 1990

# Plot:
plt.scatter(years, power, color='red', label='Data')
plt.plot(years_fit, eff_fit, color='blue', label='Trendline')
plt.title('Power per Year')
plt.xlabel('Year')
plt.ylabel('Power')
plt.legend()
plt.grid(True)
plt.show()

print(f"Fitted coefficients:\n P0 = {P0:.4f}, e = {e:.4f}")
