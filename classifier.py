import numpy as np
import subprocess #to launch linux commands
import os.path
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib.pyplot as plt
import sys


import tensorflow as tf
from tensorflow import keras
import datetime

from shared import *


sizedata=2

############ load data

train_data=np.zeros((Ndata*sizedata,ib,1))
train_labels=np.zeros((Ndata*sizedata,1))

for i in range(0,Ndata):
    train_labels[i]=1
print(train_labels)

'''for i in range(Ndata,Ndata*2):
    train_labels[i]=0
print(train_labels)'''

signalMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_data.bin"
noiseMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_transient.bin"
i=0
with open(signalMLfilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(ib)
        train_data[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1
with open(noiseMLfilename,'rb') as fd:
    while i<Ndata*sizedata:
        content=fd.read(ib)
        train_data[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1

'''for i in range(0,Ndata*2):
    train_labels[i]=round(random())
train_data=train_labels*2
ib=1
'''
ind_list = [i for i in range(Ndata*sizedata)]
shuffle(ind_list)
print(ind_list)
new_data  = train_data[ind_list[0:Ndata], :,:]
new_labels = train_labels[ind_list[0:Ndata],0]
test_data=train_data[ind_list[Ndata:Ndata*sizedata], :,:]
test_labels=train_labels[ind_list[Ndata:Ndata*sizedata],0]


print(sum(new_labels))

############ build model

print('modelbeg')

'''model = keras.Sequential([
  keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(ib,1)),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(32, activation=tf.nn.relu),
  keras.layers.Dense(8, activation=tf.nn.relu),
  keras.layers.Flatten(),
  keras.layers.Dense(2),
  keras.layers.Softmax()
])'''
	      
model = keras.Sequential([
  keras.layers.Conv1D(32, 3, activation='relu', input_shape=(ib,1), padding='valid'), #input shape = x,channel(cannot be none or nothing) ....  timestep,features
  keras.layers.MaxPooling1D(2),
  keras.layers.Conv1D(64, 3, activation='relu'),
  keras.layers.MaxPooling1D(2),
  keras.layers.Conv1D(64, 3, activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(2),
  keras.layers.Softmax()
])
'''optimizer = tf.keras.optimizers.Adam()
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])'''
optimizer = tf.keras.optimizers.Adam()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'])
	      
model.summary()

print('modelend')
print(np.shape(train_data))
print(np.shape(train_labels))

EPOCHS = 50
strt_time = datetime.datetime.now()
history = model.fit(new_data, new_labels, epochs=EPOCHS, verbose=1,batch_size=32)
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
#print(history.history['val_loss'])
#plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Val loss')
#plt.show()


'''for i in range(0,Ndata*2):
    train_labels[i]=round(random())
print(train_labels)
train_data=train_labels*2
train_labels[50]=abs(train_labels[50]-1)'''

    
test_loss, test_met = model.evaluate(test_data, test_labels, verbose=2)

print('\nMetric:', test_met)

test_loss, test_met = model.evaluate(train_data, train_labels, verbose=2)
############ save model


#model.save('path/to/location')
#model = keras.models.load_model('path/to/location')



############ predictions


predictions = model.predict(train_data)
#maximum=np.argmax(predictions,1)
#maximum=predictions
'''for i in range(0,Ndata*2):
    if maximum[i]> 0.5:
        maximum[i]=1
    else:
        maximum[i]=0
print(maximum)'''
#print(predictions)
#keras_err = abs(train_labels.flatten() - maximum)
#print(np.sum(keras_err)/(Ndata*2))

plt.plot(predictions)
plt.plot(train_labels,'r', linewidth=2)
plt.show()








