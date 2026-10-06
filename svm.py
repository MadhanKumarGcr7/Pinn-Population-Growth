import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

# ---------------------------------
# LOAD SYNTHETIC DATASET
# ---------------------------------

df = pd.read_csv(
    "synthetic_population_growth.csv"
)

print("Dataset Shape:")
print(df.shape)

# ---------------------------------
# FEATURES AND TARGET
# ---------------------------------

X = df[
    [
        "year",
        "initial_population",
        "carrying_capacity",
        "growth_rate"
    ]
]

y = df["population"]

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# FEATURE SCALING
# ---------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ---------------------------------
# SVM REGRESSOR
# ---------------------------------

svm_model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

print("\nTraining SVM...")

svm_model.fit(
    X_train_scaled,
    y_train
)

print("Training Complete!")

# ---------------------------------
# PREDICTIONS
# ---------------------------------

y_pred = svm_model.predict(
    X_test_scaled
)

# ---------------------------------
# METRICS
# ---------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nSVM RESULTS")
print("-" * 30)

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ---------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r'
)

plt.title(
    "SVM: Actual vs Predicted Population"
)

plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")

plt.grid(True)
plt.show()

# ---------------------------------
# ERROR DISTRIBUTION
# ---------------------------------

errors = y_test - y_pred

plt.figure(figsize=(10,6))

plt.hist(
    errors,
    bins=50
)

plt.title(
    "Prediction Error Distribution"
)

plt.xlabel("Error")
plt.ylabel("Frequency")

plt.grid(True)
plt.show()

# ---------------------------------
# SAMPLE PREDICTIONS
# ---------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_test.values - y_pred
})

print("\nSample Predictions:")
print(results.head(20))