# ============================================================
# PINN for World Population Dataset
# Actual Population vs PINN vs Logistic Growth Equation
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("world_population.csv")

country = "India"      # Change if needed

row = df[df["Country/Territory"] == country]

if len(row) == 0:
    raise ValueError(f"{country} not found in dataset")

# ============================================================
# 2. EXTRACT POPULATION DATA
# ============================================================

years = np.array([
    1970,
    1980,
    1990,
    2000,
    2010,
    2015,
    2020,
    2022
]).reshape(-1, 1)

population = np.array([
    row["1970 Population"].values[0],
    row["1980 Population"].values[0],
    row["1990 Population"].values[0],
    row["2000 Population"].values[0],
    row["2010 Population"].values[0],
    row["2015 Population"].values[0],
    row["2020 Population"].values[0],
    row["2022 Population"].values[0]
]).reshape(-1, 1)

# ============================================================
# 3. NORMALIZATION
# ============================================================

year_scaler = MinMaxScaler()
pop_scaler = MinMaxScaler()

X = year_scaler.fit_transform(years)
y = pop_scaler.fit_transform(population)

X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

# ============================================================
# 4. PINN MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# ============================================================
# 5. LOGISTIC GROWTH PARAMETERS
# ============================================================

r = 1.0
K = 1.0

# ============================================================
# 6. TRAINING
# ============================================================

epochs = 5000

total_losses = []
data_losses = []
physics_losses = []

for epoch in range(epochs):

    with tf.GradientTape() as outer_tape:

        with tf.GradientTape() as inner_tape:

            inner_tape.watch(X_tf)

            N_pred = model(X_tf)

        dN_dt = inner_tape.gradient(N_pred, X_tf)

        residual = dN_dt - r * N_pred * (1 - N_pred / K)

        data_loss = tf.reduce_mean(
            tf.square(y_tf - N_pred)
        )

        physics_loss = tf.reduce_mean(
            tf.square(residual)
        )

        total_loss = data_loss + physics_loss

    gradients = outer_tape.gradient(
        total_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    total_losses.append(total_loss.numpy())
    data_losses.append(data_loss.numpy())
    physics_losses.append(physics_loss.numpy())

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total={total_loss.numpy():.6f} | "
            f"Data={data_loss.numpy():.6f} | "
            f"Physics={physics_loss.numpy():.6f}"
        )

# ============================================================
# 7. PREDICTIONS
# ============================================================

pred_scaled = model.predict(X)

pred_population = pop_scaler.inverse_transform(
    pred_scaled
)

# ============================================================
# 8. METRICS
# ============================================================

mae = mean_absolute_error(
    population,
    pred_population
)

rmse = np.sqrt(
    mean_squared_error(
        population,
        pred_population
    )
)

r2 = r2_score(
    population,
    pred_population
)

print("\n==============================")
print("PINN PERFORMANCE")
print("==============================")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ============================================================
# 9. ACTUAL VS PINN COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "Year": years.flatten(),
    "Actual Population": population.flatten(),
    "PINN Prediction": pred_population.flatten()
})

comparison_df["Error"] = (
    comparison_df["Actual Population"]
    - comparison_df["PINN Prediction"]
)

comparison_df["Absolute Error"] = (
    np.abs(comparison_df["Error"])
)

print("\n==============================")
print("ACTUAL VS PINN")
print("==============================")
print(comparison_df)

# ============================================================
# 10. DERIVATIVE COMPUTATION
# ============================================================

with tf.GradientTape() as tape:

    tape.watch(X_tf)

    pinn_output = model(X_tf)

dN_dt = tape.gradient(
    pinn_output,
    X_tf
)

lhs = dN_dt.numpy().flatten()

rhs = (
    r *
    pinn_output.numpy().flatten()
    *
    (
        1
        - pinn_output.numpy().flatten()/K
    )
)

residuals = lhs - rhs

# ============================================================
# 11. LOGISTIC EQUATION CHECK
# ============================================================

equation_df = pd.DataFrame({
    "Year": years.flatten(),
    "dN/dt (PINN)": lhs,
    "rN(1-N/K)": rhs,
    "Residual": residuals
})

print("\n==============================")
print("LOGISTIC EQUATION CHECK")
print("==============================")
print(equation_df)

# ============================================================
# 12. PHYSICS METRICS
# ============================================================

mean_residual = np.mean(
    np.abs(residuals)
)

max_residual = np.max(
    np.abs(residuals)
)

print("\n==============================")
print("PHYSICS METRICS")
print("==============================")
print("Mean Residual :", mean_residual)
print("Max Residual  :", max_residual)

# ============================================================
# 13. PLOT 1
# ACTUAL VS PINN
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    years,
    population,
    'o-',
    linewidth=3,
    label="Actual Population"
)

plt.plot(
    years,
    pred_population,
    's--',
    linewidth=3,
    label="PINN Prediction"
)

plt.title(
    f"{country}: Actual vs PINN Population"
)

plt.xlabel("Year")
plt.ylabel("Population")

plt.grid(True)
plt.legend()

plt.show()

# ============================================================
# 14. PLOT 2
# TRAINING LOSSES
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    total_losses,
    label="Total Loss"
)

plt.plot(
    data_losses,
    label="Data Loss"
)

plt.plot(
    physics_losses,
    label="Physics Loss"
)

plt.title("PINN Training Losses")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()

# ============================================================
# 15. PLOT 3
# DERIVATIVE VS LOGISTIC EQUATION
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    years,
    lhs,
    'o-',
    linewidth=3,
    label="PINN dN/dt"
)

plt.plot(
    years,
    rhs,
    's--',
    linewidth=3,
    label="rN(1-N/K)"
)

plt.title(
    "PINN Derivative vs Logistic Equation"
)

plt.xlabel("Year")
plt.ylabel("Growth Rate")

plt.grid(True)
plt.legend()

plt.show()

# ============================================================
# 16. PLOT 4
# RESIDUALS
# ============================================================

plt.figure(figsize=(10,6))

plt.bar(
    years.flatten(),
    residuals
)

plt.title(
    "Physics Residuals"
)

plt.xlabel("Year")
plt.ylabel("Residual")

plt.grid(True)

plt.show()

print("\nAnalysis Completed Successfully.")