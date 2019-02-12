import numpy as np
#generate toy sound
aFreq = 44100
chords = [[210, 100, 500, 120, 2100, 1200],
	[150, 200, 300, 140],
	[500, 1600, 1400],
	[742, 761, 1245, 612]]
def generateSound(t, chord):
	audio = sum(((i+1)*np.sin(2*np.pi*f*t) for i, f in enumerate(chords[chord])))
	return audio
