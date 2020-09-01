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
	plt.plot(np.arange(200), x[0:200])
	for i in ind:
		plt.axvline(x=i)
	plt.show()
	plt.plot(np.arange(x.shape[0]), np.argmax(y, axis = 1))
	plt.show()


#######################################
# TESTING sequence model
#######################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, TimeDistributed

from mp3ToDigital import record
from audioToData import *

folder = "models\\"	
x, y, _ = getData()
batch_size = 20
num_steps = 20
gen = KerasBatchGenerator(x, y, batch_size, num_steps)

def visualEval(name):
	model = load_model(folder+name+".h5")
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
	plt.plot(np.arange(out.shape[0]), out, '.')
	plt.subplot(1,2,2) 
	plt.plot(np.arange(facit.shape[0]), facit)
	plt.legend([str(i) for i in range(songs)])
	plt.show()

def visualEvalArgmax(name):
	model = load_model(folder+name+".h5")
	tests = []
	facit = []
	out = []
	for i in range(x.shape[0]//(batch_size*num_steps)):
		t, y = next(gen.generate())
		out.append(model.predict(t))
		facit.append(y)
	outgen = [np.argmax(o.reshape((batch_size*num_steps, 30)), axis=1) for o in out]
	out = np.concatenate(outgen, axis = 0)
	print(out.shape)
	facit = np.concatenate([np.argmax(y.reshape((batch_size*num_steps, 30)), axis=1) for y in facit], axis = 0)
	print(facit.shape)
	correct = np.equal(out,facit)
	print(np.sum(correct)/correct.size)
	plt.subplot(1,2,1)
	plt.plot(np.arange(out.shape[0]), out, '.')
	plt.subplot(1,2,2) 
	plt.plot(np.arange(facit.shape[0]), facit)
	plt.legend([str(i) for i in range(songs)])
	plt.show()
import time
def testSong(name, i):
	model = load_model(folder+name+".h5")
	t, y = next(gen.generate())
	while (np.argmax(y.reshape(400, 30), axis=1)<i).any():
		t, y = next(gen.generate())
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.ylim(0,1)
	xax = np.arange(songs)
	song = plt.axvline(x=i)
	outs = np.zeros((1,songs))
	line1, = ax.plot(xax, outs.T, 'r-')
	while True:
		outs = np.vstack((outs, model.predict(t).reshape(400,30)))
		line1.set_ydata(np.average(outs, axis = 0).T)
		fig.canvas.draw()
		fig.canvas.flush_events()
		t, y = next(gen.generate())
		time.sleep(0.1)
		if (np.argmax(y.reshape(400, 30), axis = 1)!=i).any():
			i = (i+1)%songs
			outs = np.zeros((1,songs))
			song.set_xdata(i)

def testRecord(name):
	model = load_model(folder+name+".h5")
	newmodel = Sequential()
	newmodel.add(LSTM(256, return_sequences=True, batch_input_shape=(1, num_steps, x.shape[1]), activation='sigmoid')) 
	# newmodel.add(LSTM(256, dropout=0.05, return_sequences=True, activation='sigmoid')) 
	newmodel.add(Dense(songs, activation="softmax"))
	newmodel.set_weights(model.get_weights())
	newmodel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.ylim(0,1)
	xax = np.arange(songs)
	outs = np.zeros((1,songs))
	line1, = ax.plot(xax, np.average(outs, axis = 0).T, 'r-')
	i = 0
	while True:
		print(i)
		waveform = record(2)
		print(np.isnan(waveform).any())
		_, t = convert(waveform)
		print(np.isnan(t).any())
		out = newmodel.predict(t[:20].reshape(1,20,100), batch_size=1).reshape(20,30)
		print(np.isnan(out).any())
		outs = np.vstack((outs, out))
		line1.set_ydata(np.average(outs, axis = 0).T)
		fig.canvas.draw()
		fig.canvas.flush_events()
		i += 1
		if outs.shape[0]>500:
			outs = np.zeros((1,songs))



model = load_model("models\\wednesday2+8000.h5")
print("\n")
print(model.get_config())
# testSong("wednesday2+8000", 3)
