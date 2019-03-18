import numpy as np 
from makeSongs import *
from mp3ToDigital import *


aFreq = 44100						# Assign  constants.
sampleSize = 4096					#
parts = [0,20,40,80,160,320]				# 
npart = 20						#
fftfreqs = np.fft.rfftfreq(sampleSize, 1/aFreq)		#


def partition(rfft):										
	x = []										     # Create array-holders for the partitioning  of 
	
	for i in range(len(parts)-1):							     # Goes though the different frequency ranges
		boxsize = (parts[i+1]-parts[i])//npart					     # and assigns them to equaly sized ranges of 20
		x.append(sum((rfft[parts[i]+j:parts[i+1]:boxsize] for j in range(boxsize)))) #
	
	return np.concatenate(x)							     # concatenate the results to create array of length 100

def partitions():									    
	
	x = []										     # Create array-holders for the partitioning.
	
	for i in range(len(parts)-1):							     #
		boxsize = (parts[i+1]-parts[i])//npart					     #
		x.append(fftfreqs[parts[i]:parts[i+1]:boxsize])				     #
		
	return np.concatenate(x)  							     #


def convert(audio):
	
	rffts = []									     # Create array-holders for the partitioning.
	x = []										     # 
	
	for i in range(audio.size//sampleSize):						     
		
		data = audio[sampleSize*i:sampleSize*(i+1)]				     # Split the data up in chunks
		
		rfft = np.array(np.abs(np.fft.rfft(data)))				     # Use the numpy library to calculate the fast fourier transform
		
		rffts.append(rfft)							     # Appends the rfft to rffts
		x.append(partition(rfft))						     # Appends the partitioned rfft to the total representation of the song x
		
	return np.array(rffts), np.array(x)


