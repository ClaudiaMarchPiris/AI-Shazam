import pickle as pk
import numpy as np

def getData():
	songs = 30
	with open("data\\xs.pkl", "rb") as file:
		x = pk.load(file)
	with open("data\\ys.pkl", "rb") as file:
		y = pk.load(file)
	return x, y, songs