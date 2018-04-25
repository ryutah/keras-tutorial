import numpy as np
import keras

from keras.layers import Input, Dense
from keras.models import Model

# Tensorを作る
inputs = Input(shape=(784,))

x = Dense(64, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

x_train = np.random.random((1000, 784))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 784))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model.fit(x_train, y_train, epochs=20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
