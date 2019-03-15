import pickle as pk
import numpy as np
import os
from mp3ToDigital import *
from audioToData import *

# Using the method generateData(), we preprocess the songs into our "finger print" chunks
# and generate files for each song as well as a large file containing the whole dataset to
# train and test our model on.

folder = "sounds\\"                #  Assign constants.
out = "data\\"                     #  
songs = 30                         #

def generateData():
	names = []                 #  Create array-holders for our data while processing.
	xs = []                    #  
	ys = []                    #
	for subdir, dirs, files in os.walk(folder):
		i = 0
		for file in files:
			waveform = fileToArray(file)                           # Calls mp3ToDigital.fileToArray() with current song converting it to a array of amplitudes. 
			_, x = convert(waveform)                               # Calls the method audioToData.convert() finalizing preprocessing the the songs entirety.
			
			with open(out+str(i)+".pkl", 'wb') as outfile:             # Saves the processed song into a .pkl file
				pk.dump(x, outfile, protocol=pk.HIGHEST_PROTOCOL)  # 
			xs.append(x)                                               #
			
			y = np.zeros((x.shape[0], songs))                          # Creates and appends the one-hot encoded "correct answers" to the song
			y[:,i]=1						   # 
			ys.append(y)						   #
			
			names.append(os.path.splitext(file)[0])                    # Appenends the name of the song to the name array
			
			i+=1                                               
			
	xs = np.concatenate(xs)							   # Concatenates all the songs fingerprints aswell as their " correct answers" 
	ys = np.concatenate(ys)							   #
	
	with open(out+"xs.pkl", 'wb') as outfile:                                  # Saves the final .pkl files for songs, answers and names.
		pk.dump(xs, outfile, protocol=pk.HIGHEST_PROTOCOL)		   #
	with open(out+"ys.pkl", 'wb') as outfile:				   #
		pk.dump(ys, outfile, protocol=pk.HIGHEST_PROTOCOL)	           # 
	with open(out+"names.pkl", 'wb') as outfile:				   # 
		pk.dump(names, outfile, protocol=pk.HIGHEST_PROTOCOL)     	   # 
