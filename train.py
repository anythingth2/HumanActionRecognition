from keras.layers import LSTM, Dense
from keras.models import Sequential, Model
import DataGenerator
import Visualize
import numpy as np


window_size = 1
category_count = DataGenerator.category_count
xs, ys = DataGenerator.load_dataset(window_size, number_sample=10000)
x_train, x_test = xs.reshape((2, -1, window_size, 50))
y_train, y_test = ys.reshape((2, -1, category_count))

model = Sequential()
model.add(LSTM(48, return_sequences=True, input_shape=(window_size, 50,)))
model.add(LSTM(48, ))
model.add(Dense(category_count))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=8, epochs=10)

model.evaluate(x_test, y_test)

model.save('model.h5')
