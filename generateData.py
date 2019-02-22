import pickle as pk
import numpy as np
import os
from mp3ToDigital import *
from audioToData import *

folder = "sounds\\"
out = "data\\"
songs = 30
names = []
xs = []
x1s = []
ys = []
for subdir, dirs, files in os.walk(folder):
	i = 0
	for file in files:
		waveform = fileToArray(file)
		x1, x = convert(waveform)
		with open(out+str(i)+".pkl", 'wb') as outfile:
			pk.dump(x, outfile, protocol=pk.HIGHEST_PROTOCOL)
		xs.append(x)
		x1s.append(x1)
		y = np.zeros((x.shape[0], songs))
		y[:,i]=1
		ys.append(y)
		names.append(os.path.splitext(file)[0])
		i+=1
xs = np.concatenate(xs)
x1s = np.concatenate(x1s)
ys = np.concatenate(ys)
with open(out+"xs.pkl", 'wb') as outfile:
	pk.dump(xs, outfile, protocol=pk.HIGHEST_PROTOCOL)
with open(out+"x1s.pkl", 'wb') as outfile:
	pk.dump(x1s, outfile, protocol=pk.HIGHEST_PROTOCOL)
with open(out+"ys.pkl", 'wb') as outfile:
	pk.dump(ys, outfile, protocol=pk.HIGHEST_PROTOCOL)
with open(out+"names.pkl", 'wb') as outfile:
	pk.dump(names, outfile, protocol=pk.HIGHEST_PROTOCOL)

print(xs.shape, ys.shape)