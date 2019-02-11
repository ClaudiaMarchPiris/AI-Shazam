import numpy as np 
import matplotlib.pyplot as plt

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

#generate toy sound
T = 10 #seconds
t = np.arange(T*aFreq)/aFreq
chord1 = [210, 100, 500, 120, 2100, 1200]
chord2 = [150, 200, 300, 140]
chord3 = [500, 1600, 1400]
chord4 = [742, 761, 1245, 612]
audio = sum(((i+1)*np.sin(2*np.pi*f*t) for i, f in enumerate(chord4)))
print(audio.shape)

#plot first sample of sound
plt.subplot(1, 3, 1)
plt.plot(t[0:sampleSize], audio[0:sampleSize])

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

#convert to matrix
rffts, x = convert(audio)
print(rffts.shape)
print(x.shape)

#plot first freq distribution
plt.subplot(1, 3, 2)
plt.plot(np.fft.rfftfreq(sampleSize, 1/aFreq), rffts[0])

#plot first data vector
plt.subplot(1, 3, 3)
plt.plot(partitions(), x[0])

plt.show()