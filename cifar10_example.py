import keras.utils as kutils

from keras.datasets import cifar10
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = kutils.to_categorical(y_train, num_classes=10)
y_test = kutils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
x_train = x_train.astype("float32")
x_test = x_train.astype("float32")

x_train /= 255
x_test /= 255

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    shuffle=True,
)

scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)
