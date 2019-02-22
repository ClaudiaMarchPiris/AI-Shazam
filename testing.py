import numpy as np 
import matplotlib.pyplot as plt
# from audioToData import *





#######################################
# TESTING mp3 -> waveform 
#######################################
#plot first sample of sound
# T=10
# t = np.arange(T*aFreq)/aFreq
# # audio = generateSound(t, 2) #Second argument is index of chord in list of chords, were each chord is a list of frequencies.
# audio = fileToArray("A_Quiet_Thought.mp3")
# print(audio.shape)
# plt.subplot(1, 3, 1)
# plt.plot(t[0:sampleSize], audio[0:sampleSize])


#######################################
# TESTING waveform -> fftsequence x modified-fft-sequence 
#######################################
# rffts, x = convert(audio)
# print(rffts.shape)
# print(x.shape)

# #plot first freq distribution
# plt.subplot(1, 3, 2)
# plt.plot(np.fft.rfftfreq(sampleSize, 1/aFreq), rffts[0])

# #plot first data vector
# plt.subplot(1, 3, 3)
# plt.plot(partitions(), x[0])

# plt.show()

#######################################
# TESTING xs and ys
#######################################
import datainfo
x, y, songs, ind = datainfo.getData()
def visualdata():
	print(x.shape, y.shape)
	plt.plot(np.arange(x.shape[0]), x)
	for i in ind:
		plt.axvline(x=i)
	plt.show()


#######################################
# TESTING sequence model
#######################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

def visualeval():
	# x = np.expand_dims(x, 2)
	folder = "models\\"
	model = load_model(folder+"simplest.h5")
	out = model.predict(x)
	print(out)
	print(out.shape)
	plt.plot(np.arange(out.shape[0]), out)
	plt.legend([str(i) for i in range(songs)])
	plt.show()

visualdata()
