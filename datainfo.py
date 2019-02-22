import pickle as pk
import numpy as np

def getData():
	with open("data\\xs.pkl", "rb") as file:
		x = pk.load(file)
	with open("data\\ys.pkl", "rb") as file:
		y = pk.load(file)
	songs = 30
	indices = []
	i = 0
	for j in range(songs):
		indices.append(i)
		while i<y.shape[0] and y[i,j]:
			i+=1
	return x, y, songs, indices
