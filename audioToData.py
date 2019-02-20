import numpy as np 
from makeSongs import *
from mp3ToDigital import *

# constants
aFreq = 44100
sampleSize = 4096
parts = [0,20,40,80,160,320]
npart = 20
fftfreqs = np.fft.rfftfreq(sampleSize, 1/aFreq)

# utils
def partition(rfft):
	x = []
	for i in range(len(parts)-1):
		boxsize = (parts[i+1]-parts[i])//npart
		x.append(sum((rfft[parts[i]+j:parts[i+1]:boxsize] for j in range(boxsize))))
	return np.concatenate(x)
def partitions():
	x = []
	for i in range(len(parts)-1):
		boxsize = (parts[i+1]-parts[i])//npart
		x.append(fftfreqs[parts[i]:parts[i+1]:boxsize])
	return np.concatenate(x)


#generate rfft and partitions
def convert(audio):
	rffts = []
	x = []
	for i in range(audio.size//sampleSize):
		data = audio[sampleSize*i:sampleSize*(i+1)]
		rfft = np.array(np.abs(np.fft.rfft(data)))
		# rfft[j] is the amplitude of f_j=j*(aFreq/sampleSize)Hz ~= j*10. 
		# f_last = aFreq = 44100 Hz, which is 2-3 times higher than the highest note we can hear.
		rffts.append(rfft)
		x.append(partition(rfft))
	return np.array(rffts), np.array(x)


