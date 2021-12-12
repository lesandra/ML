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



shiftscale=1 # shift and scale of dataset
showstat=0
showtrain=0
basic=0 #train on std,max,maxinstd
stupidtest=0 #shuffle labels

sizedata=2

#training on simu
#signalMLfilename=SPS_PATH+'/ML'+run+'/'+'ML'+run+"_A0"+ID+"_data.bin"
#noiseMLfilename=SPS_PATH+'/ML'+run+'/'+'ML'+run+"_A0"+ID+"_transient.bin"
Ndata=3555
signalMLfilename=MLP6SIM_DATA_PATH+'MLP6SIM_simunoise.bin'
noiseMLfilename=MLP6SIM_DATA_PATH+'MLP6SIM_transient.bin'

#training on P6 signals
if trainingP6:
    Ndata=nP6
    signalMLfilename=MLP6_DATA_PATH+'MLP6_selected'+suffix+'.bin'
    noiseMLfilename=MLP6_DATA_PATH+'MLP6_transient'+suffix+'.bin'
    

if testonP6simutrain:
    ratiotrain=1
    testsignalMLfilename=MLP6_DATA_PATH+'MLP6_selected'+suffix+'.bin'
    testnoiseMLfilename=MLP6_DATA_PATH+'MLP6_transient'+suffix+'.bin'
    


############ load data

input_data=np.zeros((Ndata*sizedata,ib,1))
input_labels=np.zeros((Ndata*sizedata,1))

for i in range(Ndata):
    input_labels[i]=1
    


i=0
with open(signalMLfilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(ib)
        input_data[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1
with open(noiseMLfilename,'rb') as fd:
    while i<Ndata*sizedata:
        content=fd.read(ib)
        input_data[i,:,0]=struct.unpack('B'*ib,content)
        i=i+1
        


        
############ shift and scale of data to perform better

if shiftscale:
    plt.plot(input_data[0])
    plt.savefig('trace.png')
    plt.close()
    for i in range(Ndata*sizedata):
        input_data[i,:,0]=input_data[i,:,0]-np.mean(input_data[i,:,0])
        input_data[i,:,0]=input_data[i,:,0]/quantization
        ind=np.argmax(abs(input_data[i,:,0]))
	
        #if ind > 520:
         #   plt.plot(input_data[i])
          #  plt.show()
    '''fig,ax=plt.subplots(3, 3)

    ax[0,0].plot(input_data[1])
    ax[0,1].plot(input_data[50])
    #ax[0,1].sharey(ax[0, 0])
    ax[0,2].plot(input_data[100])
    #ax[0,1].sharex(ax[0, 0])
    ax[1,0].plot(input_data[150])
    #ax[0,1].sharex(ax[0, 0])
    ax[1,1].plot(input_data[200])
    ax[1,2].plot(input_data[250])
    ax[2,0].plot(input_data[300])
    ax[2,1].plot(input_data[350])
    ax[2,2].plot(input_data[400])
    plt.show()
    
    fig,ax=plt.subplots(3, 3)

    
    ax[0,0].plot(input_data[-1])
    ax[0,1].plot(input_data[-50])
    ax[0,2].plot(input_data[-100])
    ax[1,0].plot(input_data[-150])
    ax[1,1].plot(input_data[-200])
    ax[1,2].plot(input_data[-250])
    ax[2,0].plot(input_data[-300])
    ax[2,1].plot(input_data[-350])
    ax[2,2].plot(input_data[-400])
    plt.show()'''
    
    #plt.savefig()
    #plt.savefig('shiftscale.png')
    #plt.close()




############ histo of dataset        



nbins=20

std=np.std(input_data[:,:,:],1)
ministd=np.min(std)
maxistd=np.max(std)
plt.hist(np.std(input_data[Ndata:Ndata*sizedata,:,0],1), bins=nbins, range=(ministd,maxistd), histtype='step', linewidth=2)
plt.hist(np.std(input_data[0:Ndata,:,0],1), bins=nbins, range=(ministd,maxistd), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Std')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig('std.png')
if showstat:
    plt.show()
plt.close()

maxi=np.max(abs(input_data[:,:,:]),1)
minimaxi=np.min(maxi)
maximaxi=np.max(maxi)
plt.hist(np.max(abs(input_data[Ndata:Ndata*sizedata,:,0]),1), bins=nbins, range=(minimaxi,maximaxi), histtype='step', linewidth=2)
plt.hist(np.max(abs(input_data[0:Ndata,:,0]),1), bins=nbins, range=(minimaxi,maximaxi), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Maximum')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig('max.png')
if showstat:
    plt.show()
plt.close()


argmaxi=np.argmax(abs(input_data[:,:,:]),1)
miniargmaxi=np.min(argmaxi)
maxiargmaxi=np.max(argmaxi)
plt.hist(np.argmax(abs(input_data[Ndata:Ndata*sizedata,:,0]),1), bins=1024, range=(0,1024), histtype='step', linewidth=2)
plt.hist(np.argmax(abs(input_data[0:Ndata,:,0]),1), bins=1024, range=(0,1024), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Position of maximum')
plt.yscale('log')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig('argmax.png')
if showstat:
    plt.show()
plt.close()

maxstd=maxi/std
minimaxstd=np.min(maxstd)
maximaxstd=np.max(maxstd)
plt.hist(np.max(abs(input_data[Ndata:Ndata*sizedata,:,0]),1)/np.std(input_data[Ndata:Ndata*sizedata,:,0],1), bins=nbins, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.hist(np.max(abs(input_data[0:Ndata,:,0]),1)/np.std(input_data[0:Ndata,:,0],1), bins=nbins, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Maximum [std unit]')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig('maxinstd')
if showstat:
    plt.show()
plt.close()


#try basic NN with max and std
if basic:
    ib=3
    basicinput_data=np.zeros((Ndata*sizedata,ib,1))
    basicinput_data[:,0,:]=maxi
    basicinput_data[:,1,:]=std
    basicinput_data[:,2,:]=maxstd
    input_data=basicinput_data

   
        
############ shuffle dataset

sizetrain=int(Ndata*ratiotrain)*sizedata
sizetest=Ndata*sizedata-sizetrain

sizetrainsignal=int(Ndata*ratiotrain)
sizetestsignal=Ndata-sizetrainsignal

train_data=np.zeros((sizetrain,ib,1))
train_labels=np.zeros((sizetrain))
test_data=np.zeros((sizetest,ib,1))
test_labels=np.zeros((sizetest))

ind_list_signal = [i for i in range(Ndata)]
shuffle(ind_list_signal)
print(ind_list_signal)
ind_list_noise = [i for i in range(Ndata,Ndata*sizedata)]
shuffle(ind_list_noise)
print(ind_list_noise)

train_data[0:sizetrainsignal,:,:]  = input_data[ind_list_signal[0:sizetrainsignal],:,:]
train_data[sizetrainsignal:,:,:]  = input_data[ind_list_noise[0:sizetrainsignal], :,:]
train_labels[0:sizetrainsignal] = input_labels[ind_list_signal[0:sizetrainsignal],0]
train_labels[sizetrainsignal:] = input_labels[ind_list_noise[0:sizetrainsignal],0]

test_data[0:sizetestsignal,:,:]  = input_data[ind_list_signal[sizetrainsignal:],:,:]
test_data[sizetestsignal:,:,:]  = input_data[ind_list_noise[sizetrainsignal:], :,:]
test_labels[0:sizetestsignal] = input_labels[ind_list_signal[sizetrainsignal:],0]
test_labels[sizetestsignal:] = input_labels[ind_list_noise[sizetrainsignal:],0]


ind_list = [i for i in range(sizetrain)]
shuffle(ind_list)
train_data=train_data[ind_list]
train_labels=train_labels[ind_list]
ind_list = [i for i in range(sizetest)]
shuffle(ind_list)
test_data=test_data[ind_list]
test_labels=test_labels[ind_list]



#test a total random label
if stupidtest:
    for i in range(sizetrain):
        train_labels[i]=round(random())



print(sum(train_labels))


if testonP6simutrain:

    test_data=np.zeros((nP6*sizedata,ib,1))
    test_labels=np.zeros((nP6*sizedata))

    for i in range(nP6):
        test_labels[i]=1

    i=0
    with open(testsignalMLfilename,'rb') as fd:
        while i<nP6:
            content=fd.read(ib)
            test_data[i,:,0]=struct.unpack('B'*ib,content)
            i=i+1
    with open(testnoiseMLfilename,'rb') as fd:
        while i<nP6*sizedata:
            content=fd.read(ib)
            test_data[i,:,0]=struct.unpack('B'*ib,content)
            i=i+1
	    
	    
    if shiftscale:
        for i in range(nP6*sizedata):
            test_data[i,:,0]=test_data[i,:,0]-np.mean(test_data[i,:,0])
            test_data[i,:,0]=test_data[i,:,0]/quantization

    #plt.plot(test_data[0])
    #plt.show()


'''if basic:
    train_data=train_data[:,:,0]
    test_data=test_data[:,:,0]'''

############ build model

print('modelbeg')
trainmet=[]
valmet=[]
testmet=[]
#kernels=[ (3,),  (9,), (15,), (21,), (27,), (33,), (39,), (45,), (51,), (57,), (63,), (69,), (75,), (81,), (87,)]
#filters=[8,16,32]
kernels=[(9,)]
filters=[8]
for f in filters:
    for k in kernels:

        print(k,f)

        if basic:
            model = keras.Sequential([
              #keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(ib,1)),
              #keras.layers.Dense(128, activation=tf.nn.relu),
              keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(ib,1)),
              keras.layers.Dense(8, activation=tf.nn.relu),
              keras.layers.Flatten(),
              keras.layers.Dense(2),
              keras.layers.Softmax()
            ])

        else:
            regul=0.002
            drop=0.3
            #regul=0
            #drop=0    
            model = keras.Sequential([
              keras.layers.Conv1D(f, k, activation='relu', input_shape=(ib,1), padding='valid',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)), #input shape = x,channel(cannot be none or nothing) ....  timestep,features	      
              keras.layers.MaxPooling1D(2),
	      keras.layers.Dropout(drop),
              keras.layers.Conv1D(2*f, k, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
              keras.layers.MaxPooling1D(2),
	      keras.layers.Dropout(drop),
              keras.layers.Conv1D(2*f, k, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
              keras.layers.Flatten(),
	      keras.layers.Dropout(drop),
              keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
	      keras.layers.Dropout(drop),
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
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=3)

        model.summary()

        print('modelend')
        print(np.shape(train_data))
        print(np.shape(train_labels))

        EPOCHS = 20
        strt_time = datetime.datetime.now()
        history = model.fit(train_data, train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)#, callbacks=[callback])
        #history = model.fit(basictrain_data, train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)
        curr_time = datetime.datetime.now()
        timedelta = curr_time - strt_time
        dnn_train_time = timedelta.total_seconds()
        print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
        #print(history.history['val_loss'])

        test_loss, test_met = model.evaluate(test_data, test_labels, verbose=2)
        print('\nMetric:', test_met)
        trainmet.append(history.history['accuracy'][-1])
        valmet.append(history.history['val_accuracy'][-1])
        testmet.append(test_met)


plt.plot(history.epoch, np.array(history.history['loss']),label = 'loss')
plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (cross entropy)')
plt.legend(labels=['Training set (80%)', 'Validation set (20%)'])
#plt.title('[OVERFIT]')
plt.grid()
plt.savefig('loss.png')
if showtrain:
    plt.show()
plt.close()
plt.plot(history.epoch, np.array(history.history['accuracy']),label = 'accuracy')
plt.plot(history.epoch, np.array(history.history['val_accuracy']),label = 'Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(labels=['Training set (80%)', 'Validation set (20%)'])
#plt.title('[OVERFIT]')
plt.grid()
plt.savefig('accuracy')
if showtrain:
    plt.show()
plt.close()



'''for i in range(0,Ndata*2):
    train_labels[i]=round(random())
print(train_labels)
train_data=train_labels*2
train_labels[50]=abs(train_labels[50]-1)'''

print(trainmet,valmet,testmet)   

############ save model


#model.save('path/to/location')
#model = keras.models.load_model('path/to/location')



############ predictions


predictions = model.predict(test_data)
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


predlabelsignal=predictions[test_labels==1,1]
predlabelnoise=predictions[test_labels==0,1]


keepsignal=np.zeros(100)
keepnoise=np.zeros(100)
threshold=np.zeros(100)
for i in range(0,100):
    threshold[i]=i/100
    keepsignal[i]=len(predlabelsignal[predlabelsignal>=threshold[i]])/len(predlabelsignal)
    keepnoise[i]=len(predlabelnoise[predlabelnoise>=threshold[i]])/len(predlabelnoise)
    
plt.plot(threshold,keepnoise)
plt.plot(threshold,keepsignal)
plt.plot(threshold,(keepsignal+1-keepnoise)/2,'r',linewidth=2, linestyle='dotted')
plt.xlabel('Decision threshold')
plt.ylabel('Selection rate (test set)')
plt.legend(labels=['Noise', 'Signal', 'Accuracy'])
plt.grid()
plt.savefig('threshold.png')
if showtrain:
    plt.show()
plt.close()

print(len(predlabelsignal[predlabelsignal>=0.7])/len(predlabelsignal))
print(len(predlabelnoise[predlabelnoise>=0.7])/len(predlabelnoise))



siz=np.arange(0,len(test_labels[test_labels==0]))
plt.scatter(siz,predictions[test_labels==0,1],c='tab:orange',alpha=0.8)

plt.plot(siz,test_labels[test_labels==0],c='r', linewidth=2)
plt.plot(siz,np.zeros(len(siz))+np.mean(predictions[test_labels==0,1]),c='tab:orange', linewidth=2, linestyle='dotted')
siz=np.arange(len(test_labels[test_labels==0]),len(test_labels))
plt.scatter(siz,predictions[test_labels==1,1],c='tab:orange',alpha=0.8)
plt.plot(siz,test_labels[test_labels==1],c='r', linewidth=2)
plt.plot(siz,np.zeros(len(siz))+np.mean(predictions[test_labels==1,1]),c='tab:orange', linewidth=2, linestyle='dotted')
plt.xlabel('Label-sorted data # (noise=0, signal=1) ')
plt.ylabel('Prediction probability to be a signal (dots)')
plt.legend(labels=['Label', 'Averaged prediction'])
plt.title('Accuracy (test set) = '+str(round(test_met*100))+'%')
plt.grid()
plt.savefig('prediction.png')
if showtrain:
    plt.show()
plt.close()










