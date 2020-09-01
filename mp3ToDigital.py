# Analogic to Digital

from pydub import AudioSegment
import numpy as np

folder = "sounds\\"
def fileToArray(file):
	sound = AudioSegment.from_mp3(folder+file)
	return np.array(sound.get_array_of_samples())
#five_seconds = sound[:5000] # get the first five seconds of an mp3

import sounddevice as sd
sd.default.samplerate=44100
fs = 44100
def record(time):
	data = sd.rec(int(time*44100), channels=1).flatten()
	sd.wait()
	return data