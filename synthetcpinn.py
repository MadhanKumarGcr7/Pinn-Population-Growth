import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("synthetic_population_growth.csv")

# ==========================================================
# FEATURES
# ==========================================================

X = df[
    [
        "year",
        "initial_population",
        "carrying_capacity",
        "growth_rate"
    ]
]

y = df["population"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# SCALE X
# ==========================================================

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# ==========================================================
# SCALE Y
# ==========================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.values.reshape(-1,1)
)

y_test_scaled = y_scaler.transform(
    y_test.values.reshape(-1,1)
)

# ==========================================================
# TENSORFLOW TENSORS
# ==========================================================

X_train_tf = tf.convert_to_tensor(
    X_train_scaled,
    dtype=tf.float32
)

y_train_tf = tf.convert_to_tensor(
    y_train_scaled,
    dtype=tf.float32
)

# ==========================================================
# PINN MODEL
# ==========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

# ==========================================================
# PHYSICS WEIGHT
# ==========================================================

lambda_physics = 1e-6

# ==========================================================
# TRAINING
# ==========================================================

epochs = 2000

history_data = []
history_physics = []

for epoch in range(epochs):

    with tf.GradientTape() as outer_tape:

        with tf.GradientTape() as inner_tape:

            inner_tape.watch(X_train_tf)

            y_pred = model(
                X_train_tf,
                training=True
            )

        grads = inner_tape.gradient(
            y_pred,
            X_train_tf
        )

        # derivative wrt scaled year
        dN_dt = grads[:, 0:1]

        N = y_pred

        r = tf.reshape(
            X_train_tf[:, 3],
            (-1,1)
        )

        K = tf.reshape(
            X_train_tf[:, 2],
            (-1,1)
        )

        physics_term = r * N * (
            1 - N / (tf.abs(K) + 1e-6)
        )

        physics_loss = tf.reduce_mean(
            tf.square(
                dN_dt - physics_term
            )
        )

        data_loss = tf.reduce_mean(
            tf.square(
                y_train_tf - y_pred
            )
        )

        total_loss = (
            data_loss +
            lambda_physics * physics_loss
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

    history_data.append(
        data_loss.numpy()
    )

    history_physics.append(
        physics_loss.numpy()
    )

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch} | "
            f"Data Loss={data_loss.numpy():.6f} | "
            f"Physics Loss={physics_loss.numpy():.6f} | "
            f"Total={total_loss.numpy():.6f}"
        )

# ==========================================================
# PREDICTIONS
# ==========================================================

pinn_pred_scaled = model.predict(
    X_test_scaled
)

pinn_pred = y_scaler.inverse_transform(
    pinn_pred_scaled
).flatten()

# ==========================================================
# METRICS
# ==========================================================

mae = mean_absolute_error(
    y_test,
    pinn_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pinn_pred
    )
)

r2 = r2_score(
    y_test,
    pinn_pred
)

print("\n==============================")
print("PINN RESULTS")
print("==============================")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# LOSS CURVES
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(
    history_data,
    label="Data Loss"
)

plt.plot(
    history_physics,
    label="Physics Loss"
)

plt.yscale("log")

plt.title("PINN Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    pinn_pred,
    alpha=0.5
)

plt.plot(
    [
        y_test.min(),
        y_test.max()
    ],
    [
        y_test.min(),
        y_test.max()
    ],
    'r--'
)

plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")

plt.title("PINN: Actual vs Predicted")

plt.grid(True)

plt.show()

# ==========================================================
# SAMPLE COMPARISON
# ==========================================================

sample = 300

plt.figure(figsize=(14,6))

plt.plot(
    y_test.values[:sample],
    label="Actual",
    linewidth=3
)

plt.plot(
    pinn_pred[:sample],
    label="PINN"
)

plt.title(
    "Actual vs Predicted Population"
)

plt.xlabel("Sample")
plt.ylabel("Population")

plt.legend()
plt.grid(True)

plt.show()