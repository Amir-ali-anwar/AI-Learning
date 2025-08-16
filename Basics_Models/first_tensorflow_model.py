from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

model= keras.Sequential([
    keras.Input(shape=(1,)),
    layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd' , loss='mean_squared_error')

model.fit(X,Y, epochs=100, verbose=0)

print(model.predict(np.array([[2]])))  