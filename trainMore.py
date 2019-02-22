
import numpy as np
import pickle as pk
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras import backend
import datainfo # imports songs, x and y

x, y, songs = datainfo.getData()
x = np.expand_dims(x, 2)

folder = "models\\"
model = load_model(folder+"sequenceSoftmax.h5")
model.fit(x, y, batch_size=64, epochs=10)
print(model.summary())
model.save(folder+"sequenceSoftmax.h5")