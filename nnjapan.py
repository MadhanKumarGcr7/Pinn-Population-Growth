import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("world_population.csv")

country = "Japan"

row = df[df["Country/Territory"] == country]

years = np.array([
1970,1980,1990,2000,
2010,2015,2020,2022
]).reshape(-1,1)

population = np.array([
row["1970 Population"].values[0],
row["1980 Population"].values[0],
row["1990 Population"].values[0],
row["2000 Population"].values[0],
row["2010 Population"].values[0],
row["2015 Population"].values[0],
row["2020 Population"].values[0],
row["2022 Population"].values[0]
]).reshape(-1,1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(years)
y = y_scaler.fit_transform(population)

model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
tf.keras.layers.Dense(64, activation='tanh'),
tf.keras.layers.Dense(64, activation='tanh'),
tf.keras.layers.Dense(1)
])

model.compile(
optimizer='adam',
loss='mse'
)

history = model.fit(
X,
y,
epochs=5000,
verbose=0
)

pred = model.predict(X)

pred_population = y_scaler.inverse_transform(pred)

mae = mean_absolute_error(population, pred_population)
rmse = np.sqrt(mean_squared_error(population, pred_population))
r2 = r2_score(population, pred_population)

print("\nNN RESULTS")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

plt.figure(figsize=(10,6))
plt.plot(years, population, 'o-', label='Actual')
plt.plot(years, pred_population, 's--', label='NN Prediction')
plt.title("Japan Population Prediction (Traditional NN)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()
