from tensorflow.keras import keras
from tensorflow import layers
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)


modek= keras.Sequential([
    keras.Input(shape=(1,)),
    keras.Dense(units=3, activation='relu'),
    layers.Dense(unit=1)
])

model.compile(optimizer='adam' loss='mean_squared_error')