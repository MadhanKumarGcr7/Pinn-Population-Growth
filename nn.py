import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =====================================

# LOAD DATA

# =====================================

df = pd.read_csv("world_population.csv")

country = "India"

row = df[df["Country/Territory"] == country]

years = np.array([1970,1980,1990,2000,2010,2015,2020,2022])

population = np.array([
row["1970 Population"].values[0],
row["1980 Population"].values[0],
row["1990 Population"].values[0],
row["2000 Population"].values[0],
row["2010 Population"].values[0],
row["2015 Population"].values[0],
row["2020 Population"].values[0],
row["2022 Population"].values[0]
])

# =====================================

# NORMALIZATION

# =====================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(years.reshape(-1,1))
y = y_scaler.fit_transform(population.reshape(-1,1))

# =====================================

# MODEL

# =====================================

model = Sequential([
Dense(64, activation='tanh', input_shape=(1,)),
Dense(64, activation='tanh'),
Dense(64, activation='tanh'),
Dense(1)
])

model.compile(
optimizer='adam',
loss='mse'
)

history = model.fit(
X,
y,
epochs=1000,
verbose=0
)

# =====================================

# PREDICTION

# =====================================

pred = model.predict(X)

pred_population = y_scaler.inverse_transform(pred)

# =====================================

# METRICS

# =====================================

mae = mean_absolute_error(population,pred_population)

rmse = np.sqrt(
mean_squared_error(population,pred_population)
)

r2 = r2_score(
population,
pred_population
)

print("MAE:",mae)
print("RMSE:",rmse)
print("R2:",r2)

# =====================================

# GRAPH

# =====================================

plt.figure(figsize=(10,6))

plt.plot(
years,
population,
'o-',
label='Actual'
)

plt.plot(
years,
pred_population,
's--',
label='NN Prediction'
)

plt.legend()
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Population")

plt.title("Traditional Neural Network")

plt.show()
