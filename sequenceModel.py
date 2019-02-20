# This model is an LSTM trained to predict the following fft given the sequence. 
# The idea is that people usually try to sing along a bit before remembering what the song is called.
# The long-memeory layer can then hopefully be used to predict the song.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import datainfo # imports songs, x and y
np.random.seed(7)

x, y, songs = datainfo.getData()
x = np.expand_dims(x, 2)
print(x.shape, y.shape)



model = Sequential()
model.add(Dense(100))
model.add(LSTM(100, input_shape=(100,1), dropout=0.2))
model.add(Dense(songs)) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=100, epochs=10)
