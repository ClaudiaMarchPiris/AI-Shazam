experiment


import numpy as np
import pickle as pk
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, TimeDistributed, Input
from keras.preprocessing import sequence
from keras import backend
from datainfo import * # imports songs, x and y
np.random.seed(7) 

class Gen(object):
	def __init__(self, data, labs):
		self.data = data
		self.labs = labs
		self.num_steps = num_steps
		self.batch_size = batch_size
		# this will track the progress of the batches sequentially through the
		# data set - once the data reaches the end of the data set it will reset
		# back to zero
		self.current_idx = 0
	def generate(self):
		x = np.zeros((self.batch_size, self.num_steps, self.data.shape[1]))
		y = np.zeros((self.batch_size, songs))
		while True:
			for i in range(self.batch_size):
				n = randint(0,self.num_steps)
				# skip to beginning of next song if we reach the end within sample
				if not argmax(self.labs[self.current_idx])==argmax(self.labs[self.current_idx+n]):
					self.current_idx += n
				# reset the index back to the start of the data set if we reach the end
				if self.current_idx + n > len(self.data):
					self.current_idx = 0
				x[i, 0:n] = self.data[self.current_idx:self.current_idx + n]
				y[i] = self.labs[self.current_idx:self.current_idx + n]
				self.current_idx += n
			yield x, y



def train(name, epochs):
	############# model
	backend.clear_session() #maybe not working
	model = Sequential()
	input = Input(shape=(None, x.shape[1]))
	model.add(LSTM(256, return_sequences=True, batch_input_shape=(batch_size, None, x.shape[1]), activation='tanh')) 
	model.add(LSTM(256, dropout=0.05, return_state=True, activation='tanh')) 
	model.add(Dense(songs, activation="softmax")) 
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
	model.fit_generator(gen.generate(), x.shape[0]//(batch_size*num_steps), epochs, shuffle=true) 
	print(model.summary()) 
	############# save
	folder = "models\\"
	model.save(folder+name+".h5")

train("experiment")