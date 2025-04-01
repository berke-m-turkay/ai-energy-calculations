import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Data Section ---
# Dataset: Population sizes for five demographic groups (A through E), from 2002 to 2022.
# Each group represents a subset of the global population by IHDI.
# Units are population.
# Source: Supplementary Section 3.3.2 of the paper.
data = {
    "Year": np.arange(2002, 2023),
    "Group_A": [837569360, 842532419, 847614322, 852850479, 858900743, 865577486,
                872584425, 878940668, 884539671, 887892949, 892884564, 897734097, 903013908, 908648115,
                914353415, 919231408, 923786147, 928037275, 932952804, 933286550, 937734327],
    "Group_B": [1948391620, 1958805480, 1969160443, 1979439974,
                1989405779, 1998915471, 2009051039, 2019008642, 2028134039, 2038663177, 2051049079,
                2063744561, 2075763616, 2086659191, 2097491973, 2108720515, 2117855717, 2124907276,
                2129667559, 2131477951, 2125963925],
    "Group_C": [1023267997, 1037655073, 1052359524, 1066980105, 1081810827,
                1096667684, 1111250050, 1125854782, 1140809526, 1156208556, 1171674903, 1187555341,
                1204174546, 1220868331, 1236759082, 1251743180, 1266212255, 1280056742, 1292945066,
                1304695569, 1315813017],
    "Group_D": [1735538963, 1767768353, 1799794728, 1831399666,
                1861557405, 1890896212, 1920702679, 1951041890, 1982104825, 2013876935, 2046208383,
                2078731764, 2110657556, 2142299710, 2173961461, 2205452087, 2236504878, 2266833639,
                2296842351, 2324887859, 2350986426],
    "Group_E": [532336124, 547520610, 562496754, 577802046, 593677094, 609569057,
                626078554, 643600904, 661372648, 679433003, 697359888, 714884959, 732516965, 750063275,
                767555456, 785664280, 804699945, 824491614, 845012629, 865790681, 886835085]
}

# --- Data Preparation ---
# Convert the dictionary to a Pandas DataFrame for easy manipulation and plotting.
df = pd.DataFrame(data)

# Offset years for modeling: t = 0 corresponds to 2022.
# This simplifies interpretation of exponential growth parameters.
df["Year_Offset"] = df["Year"] - 2022

# --- Model Definition ---
# Define the exponential growth model:
#   P(t) = P0 * exp(growth_rate * t)
# Where:
#   P0 = population at t=0 (i.e., 2022)
#   growth_rate = annual exponential growth rate
def exp_growth_model(t, P0, growth_rate):
    return P0 * np.exp(growth_rate * t)

# Define consistent color mapping for plotting each group
colors = {
    "Group_A": "blue",
    "Group_B": "green",
    "Group_C": "orange",
    "Group_D": "red",
    "Group_E": "brown"
}

# Dictionary to store fitted parameters and projections
results = {}

# --- Future Projection Setup ---
# Define future time span: project population growth from 2022 to 2052 (30 years)
future_years_offset = np.linspace(0, 30, 1000)
future_years = future_years_offset + 2022

# --- Model Fitting and Plotting ---
plt.figure(figsize=(12, 8))

# Iterate over each group to fit, predict, and plot
for group in ["Group_A", "Group_B", "Group_C", "Group_D", "Group_E"]:
    pop_data = df[group]

    # Fit exponential model to historical data
    params, _ = curve_fit(
        exp_growth_model,
        df["Year_Offset"],
        pop_data,
        p0=[pop_data.iloc[0], 0.01]  # Initial guess: current pop and 1% annual growth
    )
    P0, growth_rate = params

    # Create a smooth curve for historical data fit
    years_fit_offset = np.linspace(df["Year_Offset"].min(), 0, 1000)
    historical_fit = exp_growth_model(years_fit_offset, P0, growth_rate)
    years_fit = years_fit_offset + 2022

    # Project population for 2024 (Year_Offset = 2)
    pop_2024 = exp_growth_model(2, P0, growth_rate)

    # Generate full 30-year future population projection
    future_projection = exp_growth_model(future_years_offset, P0, growth_rate)

    # Store results in dictionary
    results[group] = {
        "P0": P0,
        "pop_growth": growth_rate,
        "Projected Population (2024)": pop_2024
    }

    # --- Plotting ---
    # Plot historical data, model fit, and future projection
    plt.scatter(df["Year"], pop_data, color=colors[group], label=f"Actual {group}", s=10)
    plt.plot(years_fit, historical_fit, color=colors[group], linestyle="--", label=f"Fitted {group} Trend")
    plt.plot(future_years, future_projection, color=colors[group], linestyle="-", label=f"Future Projection {group}")

# Plot metadata
plt.title("Population Growth and Future Projection by Group")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# --- Output ---
# Rename parameters for clarity when exporting results
renamed_results = {
    "P0_A": results["Group_A"]["P0"],
    "pop_growth_A": results["Group_A"]["pop_growth"],
    "P0_B": results["Group_B"]["P0"],
    "pop_growth_B": results["Group_B"]["pop_growth"],
    "P0_C": results["Group_C"]["P0"],
    "pop_growth_C": results["Group_C"]["pop_growth"],
    "P0_D": results["Group_D"]["P0"],
    "pop_growth_D": results["Group_D"]["pop_growth"],
    "P0_E": results["Group_E"]["P0"],
    "pop_growth_E": results["Group_E"]["pop_growth"]
}

renamed_results_df = pd.DataFrame(renamed_results, index=["Values"]).T
print(renamed_results_df)
