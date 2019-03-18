import pickle as pk
import numpy as np
from random import randint
songs = 30

###################################################################################################################################

class KerasBatchGenerator(object): 				# Creates an object as an interface between data and keras fit_generator
	def __init__(self, data, labs, batch_size, num_steps):
		self.data = data
		self.labs = labs
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.current_idx = 0				# current_idx will track the progress of the batches sequentially through the
								# data set - once the data reaches the end of the data set it will reset
								# back to zero
		
	def generate(self): 					# Returns a batch_size long list of sequences of length num_steps
		x = np.zeros((self.batch_size, 
			      self.num_steps, 
			      self.data.shape[1]))
		y = np.zeros((self.batch_size, self.num_steps, songs))
		while True:
			for i in range(self.batch_size):
								# reset the index back to the start of the data set if we reach the end
				if self.current_idx + self.num_steps > len(self.data):
					self.current_idx = 0
				x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
				y[i, :] = self.labs[self.current_idx:self.current_idx + self.num_steps]
				self.current_idx += self.num_steps
			yield x, y

###################################################################################################################################

def getData(): 						# Load data into numpy array and return
	with open("data\\xs.pkl", "rb") as file:
		x = pk.load(file)
	with open("data\\ys.pkl", "rb") as file:
		y = pk.load(file)
	indices = []
	i = 0
	indices.append(i)
	for j in range(songs):
		while i<y.shape[0] and y[i,j]:
			i+=1
	return x, y, indices

##################################################################################################################################
	
def getRandomChunks(n, l): 				# create random sequence of tests through n sequences of length l*batch_size
	x, y, _ = getData()				# concatenated into single matrix.
	gen = KerasBatchGenerator(x, y, batch_size, num_steps)
	tests = []
	facit = []
	for i in range(n):
		for j in range(randint(0,500)):
			next(gen.generate()) #skip random amount of batches
		for j in range(l):
			x,y = next(gen.generate())
			tests.append(x)
			facit.append(y)
	facit = np.concatenate([np.argmax(y.reshape((20,20,30)), axis = 2) for y in facit])
	print(facit.shape)
	return tests, facit
