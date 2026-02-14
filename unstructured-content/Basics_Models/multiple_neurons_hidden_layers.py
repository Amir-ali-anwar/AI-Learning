from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

# print(X)
model= keras.Sequential([
    keras.layers.Dense(units=3, activation='relu'),
    keras.layers.Dense(units=2, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X,Y, epochs=10,verbose=0)

print(model.predict(np.array([[2]]))) 
