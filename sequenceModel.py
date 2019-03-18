# This model is an LSTM trained to predict the following fft given the sequence. 
# The idea is that people usually try to sing along a bit before remembering what the song is called.
# The long-memeory layer can then hopefully be used to predict the song.

import numpy as np					# Library imports.
import pickle as pk					#
from keras.models import Sequential, load_model		#
from keras.layers import Dense, Dropout, Activation	#
from keras.layers import LSTM, TimeDistributed		#
from keras.preprocessing import sequence		#
from keras import backend				#

from datainfo import * 					# Imports songs, x and y.



np.random.seed(7) 			# Sets numpy seed.		


x, y, _ = getData()			# Retrieves data to train network on.

batch_size = 1				# Constants for model specifications.
num_steps = 200				#
folder = "models\\"			#

gen = KerasBatchGenerator(x, y, batch_size, num_steps)	# Calls KerasBatchGeneraor with specified constants to recive generator

####################################################################################################################################

def train(name, epochs):					# Creates a model of with name=name and trains for ammount of epochs=epochs
	
	backend.clear_session() 				# Clears session if there is one active.
	
	model = Sequential()					# Creates a sequential model.
	
	model.add(LSTM(256,					# Adds a LSTM layer and sets all parameters.
		       return_sequences=True,			#
		       batch_input_shape=(batch_size,		#
					  num_steps,		#
					  x.shape[1]),		#
		       activation='sigmoid')) 			#
	
	#model.add(LSTM(256,					# Potential sequenced LSTM layer and its parameters.
	#		dropout=0.05,				#
	#		return_state=True,			#
	#		activation='tanh')) 			#
	
	model.add(Dense(songs,					# Adds a dense layer as output layer.
			activation="softmax")) 			#
	
	model.compile(loss='categorical_crossentropy',		# Sets general parameters for the model.
		      optimizer='sgd',				#
		      metrics=['accuracy']) 			#
	
	model.fit_generator(gen.generate(),			# Sets parameters for training.
			    x.shape[0]//(batch_size*num_steps),	#
			    epochs,				#
			    shuffle=true) 			#
	
	print(model.summary()) 					# Prints a summary of the model.
						
	model.save(folder+name+".h5")				# Saves the model.
	
####################################################################################################################################

def trainMore(name, epochs):					# Allows for further training of existing model.
	
	
	model = load_model(folder+name+".h5")			# Loads model.
	
	model.fit_generator(gen.generate(),			# Sets parameters for training.
			    x.shape[0]//(batch_size*num_steps),	#
			    epochs)				#
	
	print(model.summary())					# Prints a summary of the model.			
	
	model.save(folder+name+".h5")				# Saves the model.

####################################################################################################################################

def trainForever(name, epochsasave):				# Allows for indefinite training of existing model.
	
	times = 0						# times = counter.
	
	model = load_model(folder+name+".h5")			# Loads model.
	
	while(True):
		print("Times:", times)					# prints ammount of times the model has trained through its epochs
		
		model.fit_generator(gen.generate(),			#Sets parameters for training.
				    x.shape[0]//(batch_size*num_steps),	#
				    epochsasave)			#
		
		print(model.summary())					# Prints a summary of the model.
		
		model.save(folder+name+".h5")				# Saves the model.
		
		times += 1						# increment times to account for number of iterations.

####################################################################################################################################

train("special", 10)
