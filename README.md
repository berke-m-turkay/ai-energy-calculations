# ai-energy-carbon-calculations

This repository models and projects the energy consumption, computational demand, user adoption, and environmental impact of Artificial Intelligence systems from 2024 to 2050. It integrates a suite of empirically grounded submodels based on historical trends and Monte Carlo simulation techniques.



**Repository Structure**

ai_energy_simulation.py

Core simulation engine for AI energy demand forecasting under different deployment scenarios.

Incorporates results from all component models (PUE, hardware, algorithm efficiency, etc.)

Runs 1000 stochastic simulations per scenario

Computes training, inference, and total energy consumption



**Component Scripts**

These scripts generate input parameters used by the main simulation:


_pue.py_

Models the Power Usage Effectiveness of data centers over time

Output: PUE efficiency trajectory from 2007 to 2022


_hardware_efficiency.py_

Fits a logistic curve to historical hardware performance (TFLOPs/Watt)

Output: Hardware efficiency projection to 2050


_algorithm_efficiency.py_

Uses unsolved MATH problems as a proxy for algorithmic improvement

Output: Algorithmic efficiency trend and projection


_model_demand.py_

Models the number of large models deployed each year

Uses a power-law fit on published model releases (e.g. GPT, PaLM)


_model_complexity.py_

Projects computational complexity (FLOPs/model) based on model scaling trends


_query_complexity.py_

Projects per-query FLOPs (based on user interaction with systems like ChatGPT)


_population_growth.py_

Divides the global population into 5 demographic groups (A–E)

Fits exponential models to each group's historic population

Outputs parameters for future AI user growth projections


_global_power_generation.py_

Exponential fit of global electricity generation from 1990 to 2023

Used to normalize AI's share of total global energy



**Output Files**

Generated by the simulation engine:

ai_total_energy_demand.csv – Yearly total AI power consumption by scenario

ai_training_energy_demand.csv – Energy used for model training only

ai_usage_energy_demand.csv – Energy used for inference (queries)

ai_gpu_demand.csv – Projected number of GPUs needed



**Visual Outputs**

AI energy as % of global energy supply

Adoption rates by region

Population vs. AI users

Training vs. usage share of computation

Scenario comparisons for emissions, power demand, and compute trends



**Dependencies**

Python 3.8+

NumPy

SciPy

Pandas

Matplotlib



**Citation**

If using this codebase for academic or policy work, please cite:

"AI’s Energy Challenge: Racing Towards Machine Intelligence or Climate Crisis?" (2024)



**License**

MIT License — open for use, distribution, and adaptation with attribution.

For questions or collaborations, contact the project maintainer: Berke Türkay (berke_turkay@alumni.brown.edu)
