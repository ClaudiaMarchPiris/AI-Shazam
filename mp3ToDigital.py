# Analogic to Digital

from pydub import AudioSegment
import numpy as np

# Using the method fileToArray we convert the mp3. file formated songs into 
# an array of amplitude over time, later to be converted to frequency domain.

folder = "sounds\\"					# Assign constants.

def fileToArray(file):
	sound = AudioSegment.from_mp3(folder+file)      # Converts file in folder to discrete amplitude over time using the pydub library.
	return np.array(sound.get_array_of_samples())   #
