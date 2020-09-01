# This model is an LSTM trained to predict the following fft given the sequence. 
# The idea is that people usually try to sing along a bit before remembering what the song is called.
# The long-memeory layer can then hopefully be used to predict the song.

import numpy as np
import pickle as pk
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, TimeDistributed
from keras.preprocessing import sequence
from keras import backend
from datainfo import * # imports songs, x and y
np.random.seed(7) 
# data and parameters
x, y, _ = getData()
batch_size = 20
num_steps = 20
# This is weird. For some reason Keras LSTM is meant to erase the 
# hidden memory state after num_steps time steps unless you set stateful=true.
gen = Generator2(x, y, batch_size, num_steps)

def train(name, epochs):
	############# model
	backend.clear_session() #maybe not working
	model = Sequential()
	model.add(LSTM(256, dropout=0.15, return_sequences=False, stateful= True, batch_input_shape=(batch_size, num_steps, x.shape[1]), activation='sigmoid')) 
	# model.add(LSTM(256, dropout=0.25, return_sequences=True, stateful= True, activation='relu')) 
	model.add(Dense(x.shape[1], activation="softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
	model.fit_generator(gen.generate(), x.shape[0]//(batch_size*num_steps), epochs, shuffle=False) 
	print(model.summary()) 
	############# save
	folder = "models\\"
	model.save(folder+name+".h5")

def trainMore(name, epochs):
	############# model
	folder = "models\\"
	model = load_model(folder+name+".h5")
	model.fit_generator(gen.generate(), x.shape[0]//(batch_size*num_steps), epochs)
	print(model.summary())
	############## save 
	model.save(folder+name+".h5")

def trainForever(name, epochsasave):
	############# model
	folder = "models\\"
	model = load_model(folder+name+".h5")
	times = 0
	while(True):
		print("Times:", times)
		model.fit_generator(gen.generate(), x.shape[0]//(batch_size*num_steps), epochsasave, shuffle=False)
		print(model.summary())
		############## save 
		model.save(folder+name+".h5")
		times += 1


train("newest", 10)
