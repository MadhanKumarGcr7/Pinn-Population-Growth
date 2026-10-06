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

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("world_population.csv")

country = "Japan"

row = df[df["Country/Territory"] == country]

if len(row) == 0:
    raise ValueError(f"{country} not found")

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

# ======================================================
# NORMALIZATION
# ======================================================

year_scaler = MinMaxScaler()
pop_scaler = MinMaxScaler()

X = year_scaler.fit_transform(years)
y = pop_scaler.fit_transform(population)

X_tf = tf.convert_to_tensor(
    X,
    dtype=tf.float32
)

y_tf = tf.convert_to_tensor(
    y,
    dtype=tf.float32
)

# ======================================================
# PINN MODEL
# ======================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(
        64,
        activation='tanh'
    ),
    tf.keras.layers.Dense(
        64,
        activation='tanh'
    ),
    tf.keras.layers.Dense(
        64,
        activation='tanh'
    ),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

# ======================================================
# LOGISTIC PARAMETERS
# ======================================================

r = 1.0
K = 1.0

lambda_physics = 0.01

# ======================================================
# TRAINING
# ======================================================

epochs = 5000

total_losses = []
data_losses = []
physics_losses = []

for epoch in range(epochs):

    with tf.GradientTape() as outer_tape:

        with tf.GradientTape() as inner_tape:

            inner_tape.watch(X_tf)

            N_pred = model(X_tf)

        dN_dt = inner_tape.gradient(
            N_pred,
            X_tf
        )

        residual = (
            dN_dt
            - r * N_pred * (
                1 - N_pred / K
            )
        )

        data_loss = tf.reduce_mean(
            tf.square(
                y_tf - N_pred
            )
        )

        physics_loss = tf.reduce_mean(
            tf.square(
                residual
            )
        )

        total_loss = (
            data_loss
            + lambda_physics * physics_loss
        )

    gradients = outer_tape.gradient(
        total_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    total_losses.append(
        total_loss.numpy()
    )

    data_losses.append(
        data_loss.numpy()
    )

    physics_losses.append(
        physics_loss.numpy()
    )

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch} | "
            f"Total={total_loss.numpy():.6f} | "
            f"Data={data_loss.numpy():.6f} | "
            f"Physics={physics_loss.numpy():.6f}"
        )

# ======================================================
# PREDICTIONS
# ======================================================

pred_scaled = model.predict(X)

pred_population = (
    pop_scaler.inverse_transform(
        pred_scaled
    )
)

# ======================================================
# METRICS
# ======================================================

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

print("\n====================")
print("PINN RESULTS")
print("====================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ======================================================
# COMPARISON TABLE
# ======================================================

comparison_df = pd.DataFrame({
    "Year": years.flatten(),
    "Actual": population.flatten(),
    "PINN": pred_population.flatten()
})

comparison_df["Error"] = (
    comparison_df["Actual"]
    - comparison_df["PINN"]
)

print("\nACTUAL VS PINN")
print(comparison_df)

# ======================================================
# PHYSICS VALIDATION
# ======================================================

with tf.GradientTape() as tape:

    tape.watch(X_tf)

    output = model(X_tf)

dN_dt = tape.gradient(
    output,
    X_tf
)

lhs = dN_dt.numpy().flatten()

rhs = (
    r
    * output.numpy().flatten()
    * (
        1 -
        output.numpy().flatten()/K
    )
)

residuals = lhs - rhs

print("\nMean Residual:",
      np.mean(np.abs(residuals)))

print("Max Residual:",
      np.max(np.abs(residuals)))

# ======================================================
# GRAPH 1
# ======================================================

plt.figure(figsize=(10,6))

plt.plot(
    years,
    population,
    'o-',
    linewidth=3,
    label='Actual Population'
)

plt.plot(
    years,
    pred_population,
    's--',
    linewidth=3,
    label='PINN Prediction'
)

plt.title(
    "Japan Population Prediction (PINN)"
)

plt.xlabel("Year")
plt.ylabel("Population")
plt.grid(True)
plt.legend()

plt.show()

# ======================================================
# GRAPH 2
# ======================================================

plt.figure(figsize=(10,6))

plt.plot(
    total_losses,
    label='Total Loss'
)

plt.plot(
    data_losses,
    label='Data Loss'
)

plt.plot(
    physics_losses,
    label='Physics Loss'
)

plt.title("PINN Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()

# ======================================================
# GRAPH 3
# ======================================================

plt.figure(figsize=(10,6))

plt.plot(
    years,
    lhs,
    'o-',
    label='PINN dN/dt'
)

plt.plot(
    years,
    rhs,
    's--',
    label='rN(1-N/K)'
)

plt.title(
    "Derivative vs Logistic Equation"
)

plt.xlabel("Year")
plt.ylabel("Growth Rate")

plt.legend()
plt.grid(True)

plt.show()

# ======================================================
# GRAPH 4
# ======================================================

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

print("\nFinished Successfully.")