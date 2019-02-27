import pickle as pk
import numpy as np
songs = 30
batch_size = 20
num_steps = 20

class KerasBatchGenerator(object):
	def __init__(self, data, labs, batch_size, num_steps):
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
		y = np.zeros((self.batch_size, self.num_steps, songs))
		while True:
			for i in range(self.batch_size):
				if self.current_idx + self.num_steps >= len(self.data):
					# reset the index back to the start of the data set
					self.current_idx = 0
				x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
				y[i, :] = self.labs[self.current_idx:self.current_idx + self.num_steps]
				self.current_idx += self.num_steps
			yield x, y

def getData():
	with open("data\\xs.pkl", "rb") as file:
		x = pk.load(file)
	with open("data\\ys.pkl", "rb") as file:
		y = pk.load(file)
	indices = []
	i = 0
	for j in range(songs):
		indices.append(i)
		while i<y.shape[0] and y[i,j]:
			i+=1
	return x, y, indices

from random import randint
def getRandomChunks(n, l):
	x, y, _ = getData()
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
