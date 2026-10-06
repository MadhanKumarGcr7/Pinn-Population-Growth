import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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
# SCALING
# ==========================================================

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# ==========================================================
# METRIC FUNCTION
# ==========================================================

def evaluate_model(name, y_true, y_pred, train_time):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Training_Time(s)": train_time
    }

# ==========================================================
# SVM
# ==========================================================

print("\nTraining SVM...")

start = time.time()

svm = SVR(
    kernel="rbf",
    C=100,
    gamma="scale"
)

svm.fit(
    X_train_scaled,
    y_train
)

svm_time = time.time() - start

svm_pred = svm.predict(
    X_test_scaled
)

svm_results = evaluate_model(
    "SVM",
    y_test,
    svm_pred,
    svm_time
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

print("Training Random Forest...")

start = time.time()

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_time = time.time() - start

rf_pred = rf.predict(
    X_test
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_time
)

# ==========================================================
# PINN PREPARATION
# ==========================================================

print("Training PINN...")

X_train_tf = tf.convert_to_tensor(
    X_train.values,
    dtype=tf.float32
)

y_train_tf = tf.convert_to_tensor(
    y_train.values.reshape(-1,1),
    dtype=tf.float32
)

# Scale output for stable training

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.values.reshape(-1,1)
)

y_train_scaled_tf = tf.convert_to_tensor(
    y_train_scaled,
    dtype=tf.float32
)

# ==========================================================
# PINN MODEL
# ==========================================================

pinn = tf.keras.Sequential([
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

epochs = 2000

lambda_physics = 1e-8

data_losses = []
physics_losses = []

start = time.time()

# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(epochs):

    with tf.GradientTape() as outer_tape:

        with tf.GradientTape() as inner_tape:

            inner_tape.watch(X_train_tf)

            pred_scaled = pinn(
                X_train_tf,
                training=True
            )

        grads = inner_tape.gradient(
            pred_scaled,
            X_train_tf
        )

        dN_dt = grads[:, 0:1]

        pred_population = (
            pred_scaled *
            y_scaler.scale_[0]
            +
            y_scaler.mean_[0]
        )

        r = tf.reshape(
            X_train_tf[:,3],
            (-1,1)
        )

        K = tf.reshape(
            X_train_tf[:,2],
            (-1,1)
        )

        physics_rhs = (
            r *
            pred_population *
            (
                1 -
                pred_population /
                (K + 1e-8)
            )
        )

        physics_loss = tf.reduce_mean(
            tf.square(
                dN_dt -
                physics_rhs * 1e-8
            )
        )

        data_loss = tf.reduce_mean(
            tf.square(
                y_train_scaled_tf -
                pred_scaled
            )
        )

        total_loss = (
            data_loss +
            lambda_physics *
            physics_loss
        )

    gradients = outer_tape.gradient(
        total_loss,
        pinn.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            pinn.trainable_variables
        )
    )

    data_losses.append(
        data_loss.numpy()
    )

    physics_losses.append(
        physics_loss.numpy()
    )

    if epoch % 200 == 0:

        print(
            f"Epoch {epoch} | "
            f"Data={data_loss.numpy():.6f} | "
            f"Physics={physics_loss.numpy():.6f}"
        )

pinn_time = time.time() - start

# ==========================================================
# PINN PREDICTIONS
# ==========================================================

pinn_scaled_pred = pinn.predict(
    X_test.values,
    verbose=0
)

pinn_pred = y_scaler.inverse_transform(
    pinn_scaled_pred
).flatten()

pinn_results = evaluate_model(
    "PINN",
    y_test,
    pinn_pred,
    pinn_time
)

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame([
    svm_results,
    rf_results,
    pinn_results
])

print("\n")
print("="*80)
print(results_df)
print("="*80)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

# ==========================================================
# R2 COMPARISON
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("R² Comparison")

plt.ylabel("R²")

plt.grid(True)

plt.show()

# ==========================================================
# RMSE COMPARISON
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("RMSE Comparison")

plt.ylabel("RMSE")

plt.grid(True)

plt.show()

# ==========================================================
# MAPE COMPARISON
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["MAPE"]
)

plt.title("MAPE Comparison")

plt.ylabel("MAPE (%)")

plt.grid(True)

plt.show()

# ==========================================================
# PINN LOSS CURVES
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(
    data_losses,
    label="Data Loss"
)

plt.plot(
    physics_losses,
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
    svm_pred,
    alpha=0.4,
    label="SVM"
)

plt.scatter(
    y_test,
    rf_pred,
    alpha=0.4,
    label="RF"
)

plt.scatter(
    y_test,
    pinn_pred,
    alpha=0.4,
    label="PINN"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title(
    "Actual vs Predicted"
)

plt.legend()

plt.grid(True)

plt.show()

# ==========================================================
# SAMPLE COMPARISON
# ==========================================================

sample = 300

plt.figure(figsize=(15,6))

plt.plot(
    y_test.values[:sample],
    label="Actual",
    linewidth=3
)

plt.plot(
    svm_pred[:sample],
    label="SVM"
)

plt.plot(
    rf_pred[:sample],
    label="RF"
)

plt.plot(
    pinn_pred[:sample],
    label="PINN"
)

plt.legend()

plt.title(
    "Population Prediction Comparison"
)

plt.grid(True)

plt.show()