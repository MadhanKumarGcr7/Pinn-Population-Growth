import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PARAMETERS
# ----------------------------

np.random.seed(42)

num_countries = 500
years_per_country = 50

data = []

# ----------------------------
# GENERATE SYNTHETIC DATA
# ----------------------------

for country_id in range(num_countries):

    # Random initial population
    N0 = np.random.uniform(1e6, 5e7)

    # Random carrying capacity
    K = np.random.uniform(5e7, 2e8)

    # Random growth rate
    r = np.random.uniform(0.01, 0.05)

    years = np.arange(years_per_country)

    for t in years:

        # Logistic Growth Formula
        population = K / (
            1 + ((K - N0) / N0) * np.exp(-r * t)
        )

        # Add realistic noise
        noise = np.random.normal(
            0,
            population * 0.01
        )

        population_noisy = population + noise

        data.append([
            country_id,
            t,
            N0,
            K,
            r,
            population_noisy
        ])

# ----------------------------
# CREATE DATAFRAME
# ----------------------------

df = pd.DataFrame(
    data,
    columns=[
        "country_id",
        "year",
        "initial_population",
        "carrying_capacity",
        "growth_rate",
        "population"
    ]
)

# ----------------------------
# SAVE DATASET
# ----------------------------

df.to_csv(
    "synthetic_population_growth.csv",
    index=False
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------
# VISUALIZE SAMPLE COUNTRIES
# ----------------------------

plt.figure(figsize=(12,6))

for cid in range(5):

    temp = df[df["country_id"] == cid]

    plt.plot(
        temp["year"],
        temp["population"],
        label=f"Country {cid}"
    )

plt.title(
    "Synthetic Logistic Population Growth"
)

plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()