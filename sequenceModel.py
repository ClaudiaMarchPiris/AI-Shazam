# This model is an LSTM trained to predict the following fft given the sequence. 
# The idea is that people usually try to sing along a bit before remembering what the song is called.
# The long-memeory layer can then hopefully be used to predict the song.

import numpy as np
import pickle as pk
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, TimeDistributed
from keras.preprocessing import sequence
from keras import backend
from datainfo import * # imports songs, x and y
np.random.seed(7) 

def train(name):
	x, y,_ = getData() 
	batch_size = 20
	num_steps = 20 # This is weird. For some reason Keras LSTM is meant to erase the 
	# hidden memory state after num_steps time steps unless you set stateful=true.
	epochs = 20
	print(x) 
	print(x.shape, y.shape) 
	gen = KerasBatchGenerator(x, y, batch_size, num_steps)

	backend.clear_session() #maybe not working
	model = Sequential() 
	model.add(LSTM(256, return_sequences=True, stateful=True, batch_input_shape=(batch_size, num_steps, x.shape[1]))) 
	model.add(LSTM(256, dropout=0.05, return_sequences=True, stateful=True)) 
	model.add(TimeDistributed(Dense(songs, activation="softmax"))) 
	model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy']) 
	model.fit_generator(gen.generate(), x.shape[0]//(batch_size*num_steps), epochs) 
	print(model.summary()) 

	folder = "models\\"
	model.save(folder+name+".h5")

train("2lstm256b1n20e20")
