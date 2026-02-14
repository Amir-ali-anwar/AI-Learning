from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
X = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.9], dtype=float)
Y = np.array([0, 0, 1, 1, 0, 1], dtype=float)

model = keras.Sequential([
    layers.Dense(3, activation='relu', input_shape=[1]),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X,Y,  epochs=100, verbose=0 )
loss, accuracy =model.evaluate(X, Y)
print(f'Accuracy:{accuracy*100:.2f}%')

predictions= model.predict( model.predict(np.array([[0.2], [0.55], [0.75]])))
print("Predictions (probabilities):", predictions)