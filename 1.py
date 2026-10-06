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

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("synthetic_population_growth.csv")

# =====================================================
# FEATURES
# =====================================================

X = df[
    [
        "year",
        "initial_population",
        "carrying_capacity",
        "growth_rate"
    ]
]

y = df["population"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# SCALE INPUTS
# =====================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# =====================================================
# METRICS FUNCTION
# =====================================================

def calculate_metrics(name, y_true, y_pred, train_time):

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
        "Accuracy": max(0, 100 - mape),
        "Training Time(s)": train_time
    }

# =====================================================
# SVM
# =====================================================

print("Training SVM...")

start = time.time()

y_scaler_svm = StandardScaler()

y_train_svm = y_scaler_svm.fit_transform(
    y_train.values.reshape(-1,1)
).ravel()

svm = SVR(
    kernel="rbf",
    C=1000,
    gamma="auto",
    epsilon=0.01
)

svm.fit(
    X_train_scaled,
    y_train_svm
)

svm_pred_scaled = svm.predict(
    X_test_scaled
)

svm_pred = y_scaler_svm.inverse_transform(
    svm_pred_scaled.reshape(-1,1)
).flatten()

svm_time = time.time() - start

svm_results = calculate_metrics(
    "SVM",
    y_test,
    svm_pred,
    svm_time
)

# =====================================================
# RANDOM FOREST
# =====================================================

print("Training Random Forest...")

start = time.time()

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)

rf_time = time.time() - start

rf_results = calculate_metrics(
    "Random Forest",
    y_test,
    rf_pred,
    rf_time
)

# =====================================================
# PINN / NEURAL NETWORK
# =====================================================

print("Training PINN...")

y_scaler_nn = StandardScaler()

y_train_nn = y_scaler_nn.fit_transform(
    y_train.values.reshape(-1,1)
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='mse'
)

start = time.time()

history = model.fit(
    X_train_scaled,
    y_train_nn,
    epochs=300,
    batch_size=64,
    verbose=1,
    validation_split=0.1
)

nn_time = time.time() - start

nn_pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

nn_pred = y_scaler_nn.inverse_transform(
    nn_pred_scaled
).flatten()

nn_results = calculate_metrics(
    "PINN",
    y_test,
    nn_pred,
    nn_time
)

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame([
    svm_results,
    rf_results,
    nn_results
])

print("\n")
print("=" * 100)
print(results_df)
print("=" * 100)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

# =====================================================
# ERROR REPORT
# =====================================================

print("\nERROR ANALYSIS")

for _, row in results_df.iterrows():

    print(
        f"\n{row['Model']}"
        f"\nMAPE Error      : {row['MAPE']:.4f}%"
        f"\nAccuracy        : {row['Accuracy']:.4f}%"
    )

# =====================================================
# BAR CHARTS
# =====================================================

for metric in [
    "R2",
    "RMSE",
    "MAPE",
    "Accuracy"
]:

    plt.figure(figsize=(8,5))

    plt.bar(
        results_df["Model"],
        results_df[metric]
    )

    plt.title(metric + " Comparison")

    plt.grid(True)

    plt.show()

# =====================================================
# LOSS CURVE
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.legend()

plt.grid(True)

plt.title(
    "PINN Training Curve"
)

plt.show()

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    rf_pred,
    alpha=0.5,
    label="RF"
)

plt.scatter(
    y_test,
    svm_pred,
    alpha=0.5,
    label="SVM"
)

plt.scatter(
    y_test,
    nn_pred,
    alpha=0.5,
    label="PINN"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.legend()

plt.grid(True)

plt.title(
    "Actual vs Predicted"
)

plt.show()