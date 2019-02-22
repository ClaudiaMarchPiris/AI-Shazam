# This model is an LSTM trained to predict the following fft given the sequence. 
# The idea is that people usually try to sing along a bit before remembering what the song is called.
# The long-memeory layer can then hopefully be used to predict the song.

import numpy as np
import pickle as pk
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import backend
import datainfo # imports songs, x and y
np.random.seed(7)

x, y,, = datainfo.getData()
datagen = sequence.TimeseriesGenerator(x,y,x.shape[0])
print(x)
print(x.shape, y.shape)

backend.clear_session() #maybe not working
model = Sequential()
model.add(Dense(1000))
# model.add(LSTM(30, dropout=0.05, input_shape=(100,1)))
model.add(Dense(songs, activation="softmax")) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=100, epochs=20)
print(model.summary())

folder = "models\\"
model.save(folder+"simplest.h5")
