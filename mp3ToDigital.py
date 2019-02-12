# Analogic to Digital

from pydub import AudioSegment
import numpy as np

folder = "sounds\\"
def fileToArray(file):
	sound = AudioSegment.from_mp3(folder+file)
	return np.array(sound.get_array_of_samples())
#five_seconds = sound[:5000] # get the first five seconds of an mp3

#raw_data = sound._data
#raw_data = five_seconds._data

# I want to see how is the raw data 
# I want to get the frequencies
#print("Raw data:"+raw_data)