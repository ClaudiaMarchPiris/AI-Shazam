import pickle as pk
import numpy as np
from random import randint
songs = 30

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
				# n = randint(0,self.num_steps)
				# skip to beginning of next song if we reach the end within sample
				# if not argmax(self.labs[self.current_idx])==argmax(self.labs[self.current_idx+self.num_steps]):
				# 	self.current_idx += self.num_steps
				# reset the index back to the start of the data set if we reach the end
				if self.current_idx + self.num_steps > len(self.data):
					self.current_idx = randint(0,self.num_steps)
				x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
				y[i, :] = self.labs[self.current_idx:self.current_idx + self.num_steps]
				self.current_idx += self.num_steps
			yield x, y

class Generator2(object):
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
				# n = randint(0,self.num_steps)
				# skip to beginning of next song if we reach the end within sample
				# if not argmax(self.labs[self.current_idx])==argmax(self.labs[self.current_idx+self.num_steps]):
				# 	self.current_idx += self.num_steps
				# reset the index back to the start of the data set if we reach the end
				if self.current_idx + self.num_steps > len(self.data):
					self.current_idx = randint(0,self.num_steps)
				x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
				y[i, :] = self.labs[self.current_idx + self.num_steps]
				self.current_idx += self.num_steps
			yield x, y


class SingAlongGen(object):
	def __init__(self, data, labs, batch_size, num_steps, indices):
		self.data = data
		self.labs = labs
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.indices = indices
		# this will track the progress of the batches sequentially through the
		# data set - once the data reaches the end of the data set it will reset
		# back to zero
		self.song = randint(0, len(indices))
		self.current_idx = indices[self.song]
	def generate(self):
		x = np.zeros((self.batch_size, self.num_steps, self.data.shape[1]))
		y = np.zeros((self.batch_size, self.num_steps, self.data.shape[1]))
		while True:
			for i in range(self.batch_size):
				# skip to beginning of next song if we reach the end during sample
				if self.indices[self.song+1]<self.current_idx+self.num_steps+1 or self.data.shape[0]<self.current_idx+self.num_steps+1:
					self.song = randint(0, len(indices))
					self.current_idx = indices[song]
				x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
				y[i, :] = self.data[self.current_idx+1:self.current_idx + self.num_steps+1]
				self.current_idx += self.num_steps
			yield x, y

def getData():
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
		indices.append(i)
	return x, y, indices
	
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
