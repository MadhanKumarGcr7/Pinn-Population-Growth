import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv(
    "synthetic_population_growth.csv"
)

print("Dataset Shape:")
print(df.shape)

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = df[
    [
        "year",
        "initial_population",
        "carrying_capacity",
        "growth_rate"
    ]
]

y = df["population"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# RANDOM FOREST REGRESSOR
# -----------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

print("Training Complete!")

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = rf_model.predict(
    X_test
)

# -----------------------------------
# METRICS
# -----------------------------------

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

print("\nRANDOM FOREST RESULTS")
print("-" * 35)

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# -----------------------------------
# ACTUAL VS PREDICTED
# -----------------------------------

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
    "Random Forest: Actual vs Predicted Population"
)

plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")

plt.grid(True)
plt.show()

# -----------------------------------
# ERROR DISTRIBUTION
# -----------------------------------

errors = y_test - y_pred

plt.figure(figsize=(10,6))

plt.hist(
    errors,
    bins=50
)

plt.title(
    "Random Forest Error Distribution"
)

plt.xlabel("Error")
plt.ylabel("Frequency")

plt.grid(True)
plt.show()

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_importance)

# -----------------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------------

plt.figure(figsize=(10,6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.grid(True)

plt.show()

# -----------------------------------
# SAMPLE PREDICTIONS
# -----------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_test.values - y_pred
})

print("\nSample Predictions")
print(results.head(20))