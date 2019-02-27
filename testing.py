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
from datainfo import *
x, y, ind = getData()
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
from keras.layers import LSTM, TimeDistributed

def visualEval(name):
	folder = "models\\"
	model = load_model(folder+name+".h5")
	x, y, _ = getData()
	batch_size = 20
	num_steps = 20
	gen = KerasBatchGenerator(x, y, batch_size, num_steps)
	tests = []
	facit = []
	out = []
	for i in range(x.shape[0]//(10*batch_size*num_steps)):
		x, y = next(gen.generate())
		out.append(model.predict(x))
		facit.append(y)
	outgen = [o.reshape((batch_size*num_steps, 30)) for o in out]
	out = np.concatenate(outgen, axis = 0)
	print(out.shape)
	facit = np.concatenate([y.reshape((batch_size*num_steps, 30)) for y in facit], axis = 0)
	print(facit.shape)
	correct = np.equal(out,facit)
	print(np.sum(correct)/correct.size)
	plt.subplot(1,2,1)
	plt.plot(np.arange(out.shape[0]), out)
	plt.subplot(1,2,2) 
	plt.plot(np.arange(facit.shape[0]), facit)
	plt.legend([str(i) for i in range(songs)])
	plt.show()

def visualEvalArgmax(name):
	folder = "models\\"
	model = load_model(folder+name+".h5")
	x, y, _ = getData()
	batch_size = 20
	num_steps = 20
	gen = KerasBatchGenerator(x, y, batch_size, num_steps)
	tests = []
	facit = []
	out = []
	for i in range(x.shape[0]//(batch_size*num_steps)):
		x, y = next(gen.generate())
		out.append(model.predict(x))
		facit.append(y)
	outgen = [np.argmax(o.reshape((batch_size*num_steps, 30)), axis=1) for o in out]
	out = np.concatenate(outgen, axis = 0)
	print(out.shape)
	facit = np.concatenate([np.argmax(y.reshape((batch_size*num_steps, 30)), axis=1) for y in facit], axis = 0)
	print(facit.shape)
	correct = np.equal(out,facit)
	print(np.sum(correct)/correct.size)
	plt.subplot(1,2,1)
	plt.plot(np.arange(out.shape[0]), out)
	plt.subplot(1,2,2) 
	plt.plot(np.arange(facit.shape[0]), facit)
	plt.legend([str(i) for i in range(songs)])
	plt.show()

visualEvalArgmax("wednesday2+300")
