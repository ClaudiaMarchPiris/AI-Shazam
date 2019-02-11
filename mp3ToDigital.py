# Analogic to Digital

from pydub import AudioSegment

AudioSegment.ffmpeg = "C:\\ffmpeg\\bin\\ffmpeg.exe"
sound = AudioSegment.from_mp3("song1.mp3")

five_seconds = sound[:5000] # get the first five seconds of an mp3

#raw_data = sound._data
raw_data = five_seconds._data

# I want to see how is the raw data 
# I want to get the frequencies
#print("Raw data:"+raw_data)

