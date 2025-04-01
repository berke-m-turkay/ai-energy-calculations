# AI Energy Simulation Model (2024–2050)
# This code integrates all subcomponents to forecast global AI energy usage, emissions, and user impact.

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ------------------- Initialization -------------------

# Time horizon: years from 2024 to 2050
time_span = 2050 - 2024
t = np.linspace(0, time_span, 500)
years = 2024 + t

# --- PUE Parameters (from PUE script) ---
E_p0 = 1 / 1.5
alpha_p = 0.157
beta_p = 0.275

# --- Hardware Efficiency Parameters (from hardware efficiency script) ---
E_h0 = 1.8084e12
alpha_h = 0.6008
beta_h = 0.3945

# --- Algorithm Efficiency (from algorithm efficiency script) ---
E_a0 = 1 / ((10000 - 173) / 10000)
alpha_a = -0.0194
beta_a = 0.0709

# --- Model Complexity and Demand (from model complexity and model demand scripts) ---
C_c0 = 272846e18  # base FLOPs per model
eta_c1 = 1.5075   # growth exponent
lambda_c1 = 0.022

C_m0 = 58.8968    # base model count per year
eta_m = 1.3189
lambda_m = 0.3670

# --- Query Complexity (from query complexity script) ---
C_q0 = 251.2e14
eta_q = 1.7124
lambda_q = 0.5383

# --- Population & Growth Rates (from population growth script) ---
P0_A, pop_growth_A = 9.440795e+08, 5.796385e-03
P0_B, pop_growth_B = 2.150507e+09, 4.823640e-03
P0_C, pop_growth_C = 1.328686e+09, 1.279255e-02
P0_D, pop_growth_D = 2.371931e+09, 1.514939e-02
P0_E, pop_growth_E = 8.911701e+08, 2.524729e-02

# --- Adoption Parameters (see Supplementary Section 3.3.3) ---
A_A0, A_B0, A_C0, A_D0, A_E0 = 0.17, 0.13, 0.07, 0.04, 0.02

At20, At80 = 5, 15
gamma_A = 2 * math.log(4) / (At20 - At80)
t_0A = (At20 + At80) / 2

Bt20, Bt80 = 8, 21
gamma_B = 2 * math.log(4) / (Bt20 - Bt80)
t_0B = (Bt20 + Bt80) / 2

Ct10, Ct40 = 6, 19
gamma_C = -math.log(6) / (Ct40 - Ct10)
t_0C = Ct10 + (Ct40 - Ct10) * math.log(9) / math.log(6)

Dt10, Dt40 = 9, 20
gamma_D = -math.log(6) / (Dt40 - Dt10)
t_0D = Dt10 + (Dt40 - Dt10) * math.log(9) / math.log(6)

Et5, Et20 = 13, 36
gamma_E = -math.log(6) / (Et20 - Et5)
t_0E = Et5 + (Et20 - Et5) * math.log(9) / math.log(6)

# --- Global Energy Growth (from global power generation script) ---
initial_global_energy = 29188  # TWh in 2022
g_e = 0.0286

# ------------------- Simulation Parameters -------------------
n_simulations = 1000
n_years = len(t)

# Scenarios: baseline, fewer larger models, more smaller models
scenario_multipliers = {
    "baseline": (1, 1),
    "fewer_larger": (1.2, 0.6),
    "more_smaller": (0.8, 1.35)
}

scenario_results = {scenario: {} for scenario in scenario_multipliers}

# ------------------- Monte Carlo Simulations -------------------

for scenario, (complexity_multiplier, model_demand_multiplier) in scenario_multipliers.items():
    # Arrays to hold results per simulation
    C_c_scenario = np.zeros((n_simulations, n_years))
    C_m_scenario = np.zeros((n_simulations, n_years))
    C_t_scenario = np.zeros((n_simulations, n_years))
    P_t_scenario = np.zeros((n_simulations, n_years))
    P_tt_scenario = np.zeros((n_simulations, n_years))
    P_ut_scenario = np.zeros((n_simulations, n_years))

    for i in range(n_simulations):
        # Sample stochastic parameters from normal distributions
        eta_c1_sim = np.random.normal(eta_c1 * complexity_multiplier, 0.03)
        lambda_c1_sim = np.random.normal(lambda_c1, 0.02)
        eta_m_sim = np.random.normal(eta_m * model_demand_multiplier, 0.03)
        lambda_m_sim = np.random.normal(lambda_m, 0.02)

        # --- Efficiency curves ---
        E_p_sim = E_p0 * np.exp((alpha_p / beta_p) * (1 - np.exp(-beta_p * t)))
        E_h_sim = E_h0 * np.exp((alpha_h / beta_h) * (1 - np.exp(-beta_h * t)))
        E_a_sim = E_a0 * np.exp((alpha_a / beta_a) * (1 - np.exp(-beta_a * t)))

        # --- Computation ---
        C_c = C_c0 * (np.maximum(t + 2 + lambda_c1_sim, 1e-6)) ** eta_c1_sim
        C_m = C_m0 * (np.maximum(t + 2 + lambda_m_sim, 1e-6)) ** eta_m_sim
        C_q = C_q0 * (np.maximum(t + 2 + lambda_q, 1e-6)) ** eta_q

        # --- Population ---
        P_A = P0_A * np.exp(pop_growth_A * t)
        P_B = P0_B * np.exp(pop_growth_B * t)
        P_C = P0_C * np.exp(pop_growth_C * t)
        P_D = P0_D * np.exp(pop_growth_D * t)
        P_E = P0_E * np.exp(pop_growth_E * t)

        # --- AI Adoption ---
        A_A = A_A0 + (1 - A_A0) / (1 + np.exp(gamma_A * (t - t_0A)))
        A_B = A_B0 + (1 - A_B0) / (1 + np.exp(gamma_B * (t - t_0B)))
        A_C = A_C0 + (1 - A_C0) / (1 + np.exp(gamma_C * (t - t_0C)))
        A_D = A_D0 + (1 - A_D0) / (1 + np.exp(gamma_D * (t - t_0D)))
        A_E = A_E0 + (1 - A_E0) / (1 + np.exp(gamma_E * (t - t_0E)))

        # --- Compute demand and energy usage ---
        C_u = P_A * A_A + P_B * A_B + P_C * A_C + P_D * A_D + P_E * A_E
        C_t = C_c * C_m + C_u * C_q
        P_t = (C_t / (E_p_sim * E_h_sim / E_a_sim)) / 1e12  # TWh
        P_tt = (C_c * C_m / (E_p_sim * E_h_sim / E_a_sim)) / 1e12
        P_ut = (C_u * C_q / (E_p_sim * E_h_sim / E_a_sim)) / 1e12

        # Store results
        C_c_scenario[i] = C_c
        C_m_scenario[i] = C_m
        C_t_scenario[i] = C_t
        P_t_scenario[i] = P_t
        P_tt_scenario[i] = P_tt
        P_ut_scenario[i] = P_ut

    # --- Scenario summary statistics ---
    scenario_results[scenario]["C_c"] = np.percentile(C_c_scenario, [50, 5, 95], axis=0)
    scenario_results[scenario]["C_m"] = np.percentile(C_m_scenario, [50, 5, 95], axis=0)
    scenario_results[scenario]["C_t"] = np.percentile(C_t_scenario, [50, 5, 95], axis=0)
    scenario_results[scenario]["P_t"] = np.percentile(P_t_scenario, [50, 5, 95], axis=0)
    scenario_results[scenario]["P_tt"] = np.percentile(P_tt_scenario, [50, 5, 95], axis=0)
    scenario_results[scenario]["P_ut"] = np.percentile(P_ut_scenario, [50, 5, 95], axis=0)
    

# Plotting results for all scenarios
plt.figure(figsize=(12, 8))

# Computational Complexity
plt.subplot(2, 2, 1)
for scenario, results in scenario_results.items():
    plt.plot(years, results["C_c"][0], label=f'{scenario.capitalize()} (Median)')
    plt.fill_between(
        years, results["C_c"][1], results["C_c"][2],
        alpha=0.2, label=f'{scenario.capitalize()} (90% CI)'
    )
plt.title("Computational Complexity (FLOPs/model)")
plt.xlabel("Year")
plt.ylabel("Complexity")
plt.legend()
plt.grid()

# Model Demand
plt.subplot(2, 2, 2)
for scenario, results in scenario_results.items():
    plt.plot(years, results["C_m"][0], label=f'{scenario.capitalize()} (Median)')
    plt.fill_between(
        years, results["C_m"][1], results["C_m"][2],
        alpha=0.2, label=f'{scenario.capitalize()} (90% CI)'
    )
plt.title("Model Demand (Models/year)")
plt.xlabel("Year")
plt.ylabel("Demand")
plt.legend()
plt.grid()

# Total Computational Demand
plt.subplot(2, 2, 3)
for scenario, results in scenario_results.items():
    plt.plot(years, results["C_t"][0], label=f'{scenario.capitalize()} (Median)')
    plt.fill_between(
        years, results["C_t"][1], results["C_t"][2],
        alpha=0.2, label=f'{scenario.capitalize()} (90% CI)'
    )
plt.title("Total Computational Demand (FLOPs/year)")
plt.xlabel("Year")
plt.ylabel("Demand")
plt.legend()
plt.grid()

# Power Consumption
plt.subplot(2, 2, 4)
for scenario, results in scenario_results.items():
    plt.plot(years, results["P_t"][0], label=f'{scenario.capitalize()} (Median)')
    plt.fill_between(
        years, results["P_t"][1], results["P_t"][2],
        alpha=0.2, label=f'{scenario.capitalize()} (90% CI)'
    )
plt.title("Power Consumption (TWh/year)")
plt.xlabel("Year")
plt.ylabel("Power")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# AI Energy as % of Global Energy Consumption
plt.figure(figsize=(12, 8))

# Define colors for scenarios
scenario_colors = {
    "baseline": "#3b75b1",
    "more_smaller": "#519d37",
    "fewer_larger": "#ef872b"
}

for scenario, results in scenario_results.items():
    # Compute AI energy as % of global energy consumption
    AI_energy_percentage = (results["P_t"][0] / (initial_global_energy * np.exp(g_e * t))) * 100
    AI_energy_min = (results["P_t"][1] / (initial_global_energy * np.exp(g_e * t))) * 100
    AI_energy_max = (results["P_t"][2] / (initial_global_energy * np.exp(g_e * t))) * 100

    plt.plot(years, AI_energy_percentage, label=f'{scenario.capitalize()} (Median)')
    plt.fill_between(
        years, AI_energy_min, AI_energy_max,
        alpha=0.2, label=f'{scenario.capitalize()} (90% CI)'
    )

# Highlight specific years: 2024, 2030, 2040, 2050
highlight_years = [2024, 2030, 2040, 2050]
highlight_indices = [np.argmin(np.abs(years - year)) for year in highlight_years]

for scenario, results in scenario_results.items():
    AI_energy_percentage = (results["P_t"][0] / (initial_global_energy * np.exp(g_e * t))) * 100
    for year, idx in zip(highlight_years, highlight_indices):
        # Only show the baseline in 2024
        if year == 2024 and scenario != "baseline":
            continue
        # Scatter plot the points with specific colors
        plt.scatter(years[idx], AI_energy_percentage[idx], color=scenario_colors[scenario])
        # Annotate the points with their values
        plt.text(
            years[idx], AI_energy_percentage[idx], f"{AI_energy_percentage[idx]:.1f}%",
            color="black", fontsize=8, ha="center", va="bottom"
        )

plt.title("AI Energy Consumption as % of Global Energy Consumption")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid()
plt.show()


# Adoption Rates
plt.figure(figsize=(10, 7))
# Group A
plt.plot(years, A_A, label='Group A', color='blue')

# Group B
plt.plot(years, A_B, label='Group B', color='green')

# Group C
plt.plot(years, A_C, label='Group C', color='orange')

# Group D
plt.plot(years, A_D, label='Group D', color='red')

# Group E
plt.plot(years, A_E, label='Group E', color='brown')

plt.title('Adoption Rate (%)')
plt.xlabel('Year')
plt.ylabel('Adoption Rate (%)')
plt.legend(loc='upper left', ncol=2)  # Adjust legend layout for readability
plt.grid(True)
plt.xticks(np.arange(2024, 2024+time_span, 2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.show()


# Population
plt.figure(figsize=(10, 7))
# Group A
plt.plot(years, P_A, label='Group A', color='blue')

# Group B
plt.plot(years, P_B, label='Group B', color='green')

# Group C
plt.plot(years, P_C, label='Group C', color='orange')

# Group D
plt.plot(years, P_D, label='Group D', color='red')

# Group E
plt.plot(years, P_E, label='Group E', color='brown')

plt.title('Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(loc='upper left', ncol=2)
plt.grid(True)
plt.xticks(np.arange(2024, 2024+time_span, 2))
plt.tight_layout()
plt.show()

# AI Users
plt.figure(figsize=(10, 7))
# Group A
plt.plot(years, P_A*A_A, label='Group A', color='blue')

# Group B
plt.plot(years, P_B*A_B, label='Group B', color='green')

# Group C
plt.plot(years, P_C*A_C, label='Group C', color='orange')

# Group D
plt.plot(years, P_D*A_D, label='Group D', color='red')

# Group E
plt.plot(years, P_E*A_E, label='Group E', color='brown')

plt.title('AI Users')
plt.xlabel('Year')
plt.ylabel('Users')
plt.legend(loc='upper left', ncol=2)
plt.grid(True)
plt.xticks(np.arange(2024, 2024+time_span, 2))
plt.tight_layout()
plt.show()

# Extract yearly total energy demand for all scenarios
energy_demand_data = {"Year": np.arange(2024, 2051)}
training_demand_data = {"Year": np.arange(2024, 2051)}
usage_demand_data = {"Year": np.arange(2024, 2051)}

for scenario, results in scenario_results.items():
    yearly_energy_demand = results["P_t"][0]
    energy_demand_data[scenario.capitalize()] = yearly_energy_demand[::len(t) // (2050 - 2024)]

    yearly_training_demand = results["P_tt"][0]
    training_demand_data[scenario.capitalize()] = yearly_training_demand[::len(t) // (2050 - 2024)]

    yearly_usage_demand = results["P_ut"][0]
    usage_demand_data[scenario.capitalize()] = yearly_usage_demand[::len(t) // (2050 - 2024)]

# Create a DataFrame
energy_demand_df = pd.DataFrame(energy_demand_data)
training_demand_df = pd.DataFrame(training_demand_data)
usage_demand_df = pd.DataFrame(usage_demand_data)

# Save the DataFrame as a CSV file
energy_demand_df.to_csv("ai_total_energy_demand.csv", index=False)
training_demand_df.to_csv("ai_training_energy_demand.csv", index=False)
usage_demand_df.to_csv("ai_usage_energy_demand.csv", index=False)


# Percentage of computational demand due to training vs usage (without confidence interval)
plt.figure(figsize=(12, 8))

for scenario, results in scenario_results.items():
    # Calculate the percentage of training vs total computational demand
    training_percentage = (C_q*C_u / results["C_t"][0]) * 100

    plt.plot(years, training_percentage, label=f'{scenario.capitalize()} (Median)')

plt.title("Percentage of Computational Demand Due to Training vs Usage")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid()
plt.show()


# Calculate each group's share of C_n_total over time
C_A = P_A * A_A
C_B = P_B * A_B
C_C = P_C * A_C
C_D = P_D * A_D
C_E = P_E * A_E

# Compute the shares of C_n_total for each group
C_A_share = C_A / C_u
C_B_share = C_B / C_u
C_C_share = C_C / C_u
C_D_share = C_D / C_u
C_E_share = C_E / C_u

# Plot the shares over time
plt.figure(figsize=(12, 8))

plt.plot(years, C_A_share, label='Group A', color='blue')
plt.plot(years, C_B_share, label='Group B', color='green')
plt.plot(years, C_C_share, label='Group C', color='orange')
plt.plot(years, C_D_share, label='Group D', color='red')
plt.plot(years, C_E_share, label='Group E', color='brown')

plt.title('Share of AI Energy Demand by Country Group')
plt.xlabel('Year')
plt.ylabel('Share of Total Computational Demand (%)')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.xticks(np.arange(2024, 2024 + time_span, 2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # Convert to percentage format
plt.tight_layout()
plt.show()

# GPUs
gpu_power_kw = 0.7  # GPU power in kilowatts
gpu_energy_per_year_kwh = gpu_power_kw * 24 * 365  # Energy usage per GPU per year
gpu_co2_per_year_tons = 1308.8767 / 1000  # CO2 emissions per GPU per year in tons
total_atmospheric_mass_tons = 5.15e15  # Total mass of Earth's atmosphere in tons
util_rate = 0.8  # GPU utilization rate

# Calculate the number of GPUs required and CO2 emissions for each scenario
gpu_demand_data = {"Year": np.arange(2024, 2051)}
co2_ppm_change = {"Year": np.arange(2024, 2051)}
cumulative_co2_ppm = {"Year": np.arange(2024, 2051)}

for scenario, results in scenario_results.items():
    yearly_energy_demand_kwh = results["P_t"][0] * 1e9  # Convert TWh to kWh
    yearly_gpu_demand = yearly_energy_demand_kwh / gpu_energy_per_year_kwh / util_rate
    yearly_co2_emissions = yearly_gpu_demand * gpu_co2_per_year_tons  # Total CO2 emissions in tons
    yearly_co2_ppm = (yearly_co2_emissions / total_atmospheric_mass_tons) * 1e6  # Convert to ppm

    # Calculate cumulative CO2 ppm
    cumulative_ppm = np.cumsum(yearly_co2_ppm)  # Cumulative sum across years

    gpu_demand_data[scenario.capitalize()] = yearly_gpu_demand[::len(t) // (2050 - 2024)]
    co2_ppm_change[scenario.capitalize()] = yearly_co2_ppm[::len(t) // (2050 - 2024)]
    cumulative_co2_ppm[scenario.capitalize()] = cumulative_ppm[::len(t) // (2050 - 2024)]

# Create DataFrames
gpu_demand_df = pd.DataFrame(gpu_demand_data)
co2_ppm_df = pd.DataFrame(co2_ppm_change)
cumulative_ppm_df = pd.DataFrame(cumulative_co2_ppm)

# Save the data
gpu_demand_df.to_csv("ai_gpu_demand.csv", index=False)
co2_ppm_df.to_csv("gpu_co2_ppm_change.csv", index=False)
cumulative_ppm_df.to_csv("gpu_cumulative_co2_ppm.csv", index=False)

# Plot both graphs in a single figure
fig, axs = plt.subplots(2, 1, figsize=(7, 10))
