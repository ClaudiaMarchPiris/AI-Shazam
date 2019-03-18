# AI-Shazam
AI project

A Keras sequence model for song prediction. 

audioToData.py 
Function convert(audio) converts 1d-waveform "audio" into a sequence of features. Each feature consists of values from the FFT of "sampleSize" waveform data points. Each interval seperated by values in "parts" makes 20 features by summing (interval length)/20 values for each feature. 

generateData.py
Reads files from folder sounds and writes files into folder data. The waveform is converted into a sequence of feature with audioData.py and written to its own file. The sequence is also concatenated into a large matrix xs.pkl containing one long sequence of all the songs. 



Songs: https://drive.google.com/drive/folders/1BM7sDBg2geB4_XXMUHs2Mdu4LawNkVUu
