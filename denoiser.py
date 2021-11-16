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




############ load data

train_data=np.zeros((Ndata,ib,1))
train_labels=np.zeros((Ndata,ib,1))


inputMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_data.bin"
targetMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_simu.bin"
i=0
with open(inputMLfilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(ib)
        train_data[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1
i=0
with open(targetMLfilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(ib)
        train_labels[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1
	
noise=train_data-train_labels


std_noise=np.std(noise,1).flatten()
mean_noise=train_labels[:,0,0]
print(std_noise)
print(mean_noise)

maxi_data=np.max(abs(train_data),1).flatten()
print(maxi_data)
ind_data=np.argmax(abs(train_data),1).flatten()
print(ind_data)
maxi_labels=np.max(abs(train_labels),1).flatten()
print(maxi_labels)
ind_labels=np.argmax(abs(train_labels),1).flatten()
print(ind_labels)
dt_data=ind_data-ind_labels
damp_data=(maxi_data-maxi_labels)/maxi_labels
print(damp_data)

maxinstd=(maxi_data-mean_noise)/std_noise
print(maxinstd)



ind_list = [i for i in range(Ndata)]
shuffle(ind_list)
print(ind_list)
new_data  = train_data[ind_list[0:100], :,:]
new_labels = train_labels[ind_list[0:100],:,0]
test_data=train_data[ind_list[100:Ndata], :,:]
test_labels=train_labels[ind_list[100:Ndata],:,0]



'''print(maxi_data[ind_data>512])
plt.plot(ind_data)
plt.plot(ind_labels)
plt.show()

plt.plot(maxi_data)
plt.show()
plt.plot(std_noise)
plt.show()
plt.plot(maxinstd)
plt.show()'''

############ build model

print('modelbeg')

'''model = keras.Sequential([
  keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(ib,1)),
#  keras.layers.Dense(128, activation=tf.nn.relu),
#  keras.layers.Dense(32, activation=tf.nn.relu),
#  keras.layers.Dense(8, activation=tf.nn.relu),
  keras.layers.Flatten(),  
  keras.layers.Dense(1024)

])'''
	      
model = keras.Sequential([
  keras.layers.Conv1D(32, 3, activation='relu', input_shape=(ib,1), padding='valid'), #input shape = x,channel(cannot be none or nothing) ....  timestep,features
  keras.layers.MaxPooling1D(2),
  keras.layers.Conv1D(64, 3, activation='relu'),
  keras.layers.MaxPooling1D(2),
  keras.layers.Conv1D(64, 3, activation='relu'),
  keras.layers.MaxPooling1D(2),
  keras.layers.Conv1D(64, 3, activation='relu'),  
  keras.layers.Flatten(),
  keras.layers.Dense(1024),
  #keras.layers.Dense(2),
  #keras.layers.Softmax()
])
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])
'''optimizer = tf.keras.optimizers.Adam()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'])'''
	      
model.summary()

print('modelend')
print(np.shape(train_data))
print(np.shape(train_labels))

EPOCHS = 100
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


predictions = np.round(model.predict(train_data))
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

maxi_predictions=np.max(abs(predictions),1)
print(maxi_predictions)
ind_predictions=np.argmax(abs(predictions),1)
dt_predictions=ind_predictions-ind_labels
damp_predictions=(maxi_predictions-maxi_labels)/maxi_labels
print(damp_predictions)

'''plt.plot(maxi_data)
plt.plot(maxi_predictions)
plt.plot(maxi_labels)
plt.show()
plt.plot(maxi_data,maxi_predictions,'.')
plt.show()'''
plt.plot(maxinstd,dt_data,'.')
plt.plot(maxinstd,dt_predictions,'.')
plt.show()
plt.plot(maxinstd,damp_data,'.')
plt.plot(maxinstd,damp_predictions,'.')
plt.show()


print(np.shape(predictions[19]))
plt.plot(train_data[19])
plt.show()
plt.plot(train_labels[19])
plt.show()
plt.plot(train_labels[19])
plt.plot(np.round(predictions[19]))
plt.show()











