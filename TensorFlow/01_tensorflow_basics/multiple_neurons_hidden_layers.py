from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)


model= keras.Sequential([
    keras.Input(shape=(1,)),
    keras.Dense(units=3, activation='relu'),
    layers.Dense(unit=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X,Y, epochs=10,verbose=0)

print(model.predict(np.array([[2]]))) 