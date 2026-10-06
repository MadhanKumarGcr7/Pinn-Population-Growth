import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("world_population.csv")

country = "India"      # Change if needed

row = df[df["Country/Territory"] == country]

if len(row) == 0:
    raise ValueError(f"{country} not found in dataset")

years = np.array([
    1970,
    1980,
    1990,
    2000,
    2010,
    2015,
    2020,
    2022
], dtype=np.float32)

population = np.array([
    row["1970 Population"].values[0],
    row["1980 Population"].values[0],
    row["1990 Population"].values[0],
    row["2000 Population"].values[0],
    row["2010 Population"].values[0],
    row["2015 Population"].values[0],
    row["2020 Population"].values[0],
    row["2022 Population"].values[0]
], dtype=np.float32)

# =====================================================
# NORMALIZE DATA
# =====================================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(years.reshape(-1, 1))
Y = y_scaler.fit_transform(population.reshape(-1, 1))

X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
Y_tf = tf.convert_to_tensor(Y, dtype=tf.float32)

# =====================================================
# LOGISTIC PARAMETERS
# =====================================================

r = 1.0
K = 1.0

# =====================================================
# PINN MODEL
# =====================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# =====================================================
# TRAINING
# =====================================================

epochs = 5000

loss_history = []
data_loss_history = []
physics_loss_history = []

for epoch in range(epochs):

    with tf.GradientTape() as outer_tape:

        with tf.GradientTape() as inner_tape:

            inner_tape.watch(X_tf)

            N_pred = model(X_tf)

        dN_dt = inner_tape.gradient(N_pred, X_tf)

        residual = dN_dt - r * N_pred * (1 - N_pred / K)

        data_loss = tf.reduce_mean(
            tf.square(Y_tf - N_pred)
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

    loss_history.append(total_loss.numpy())
    data_loss_history.append(data_loss.numpy())
    physics_loss_history.append(physics_loss.numpy())

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total Loss = {total_loss.numpy():.6f} | "
            f"Data Loss = {data_loss.numpy():.6f} | "
            f"Physics Loss = {physics_loss.numpy():.6f}"
        )

# =====================================================
# PREDICTIONS
# =====================================================

pred_norm = model(X_tf).numpy()

pred_population = y_scaler.inverse_transform(pred_norm)

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    population,
    pred_population.flatten()
)

rmse = np.sqrt(
    mean_squared_error(
        population,
        pred_population.flatten()
    )
)

r2 = r2_score(
    population,
    pred_population.flatten()
)

print("\n==============================")
print("PINN RESULTS")
print("==============================")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# =====================================================
# RESIDUAL
# =====================================================

with tf.GradientTape() as tape:

    tape.watch(X_tf)

    pred = model(X_tf)

dN_dt = tape.gradient(pred, X_tf)

residual = (
    dN_dt -
    r * pred * (1 - pred / K)
).numpy()

# =====================================================
# PLOT 1
# ACTUAL VS PINN
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    years,
    population,
    'o-',
    linewidth=2,
    label='Actual Population'
)

plt.plot(
    years,
    pred_population,
    's--',
    linewidth=2,
    label='PINN Prediction'
)

plt.xlabel("Year")
plt.ylabel("Population")
plt.title(f"PINN Population Prediction ({country})")
plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# PLOT 2
# TRAINING LOSS
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    loss_history,
    label='Total Loss'
)

plt.plot(
    data_loss_history,
    label='Data Loss'
)

plt.plot(
    physics_loss_history,
    label='Physics Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training Loss")
plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# PLOT 3
# PHYSICS RESIDUAL
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    years,
    residual,
    'o-'
)

plt.axhline(
    0,
    linestyle='--'
)

plt.xlabel("Year")
plt.ylabel("Residual")
plt.title("Logistic Equation Residual")
plt.grid(True)

plt.show()