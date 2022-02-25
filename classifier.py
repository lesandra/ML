import numpy as np
import subprocess #to launch linux commands
import os.path
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow import keras
import datetime

from shared import *


shiftscale=1 # shift and scale of dataset
FFT=0
FFTri=0
FFTalone=1
if FFT or FFTalone:
    from scipy.fftpack import rfft, irfft, rfftfreq
if FFT or FFTalone or FFTri:
    shiftscale=1   
showstat=0
showtrain=0
basic=0 #train on std,max,maxinstd
stupidtest=0 #shuffle labels
sizedata=2
unbalanced=0


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
    load=np.loadtxt(SPS_PATH+'runcoincantevt_space.txt',dtype='int')    
    #load=np.loadtxt(PBS_PATH+'P6all.txt',dtype='int')    
    #run=load[:,0]
    coinc=load[:,1]
    ant=load[:,2]
    #evt=load[:,3]
    
#if trainingP6byant:    
    #Ndata=112 #ant 109
    #Ndata=104 #ant110
    #Ndata=106 #ant114
    #Ndata=80 #ant109 tpsel
    #Ndata=76 #ant110 tpsel
    #signalMLfilename=MLP6_DATA_PATH+'MLP6_109_selected'+suffix+'.bin'
    #noiseMLfilename=MLP6_DATA_PATH+'MLP6_109_transient'+suffix+'.bin'


#training on hybrid
if traininghyb:
    Ndata=nhyb
    signalMLfilename=MLHYB_DATA_PATH+'MLhybrid_selected'+suffix+'.bin'
    noiseMLfilename=MLHYB_DATA_PATH+'MLhybrid_transient'+suffix+'.bin'
    if unbalanced:
        signalMLfilename=MLHYB_DATA_PATH+'MLhybrid_selected_unbal20'+suffix+'.bin'
        noiseMLfilename=MLHYB_DATA_PATH+'MLhybrid_transient_unbal20'+suffix+'.bin'
        sizedata=21


#training on P6 and hybrid
if trainingP6hyb:
    Ndata=nhyb+nP6
    signalMLfilename=MLP6HYB_DATA_PATH+'MLP6hybrid_selected'+suffix+'.bin'
    noiseMLfilename=MLP6HYB_DATA_PATH+'MLP6hybrid_transient'+suffix+'.bin'



if testonP6:
    Ntest=nP6
    ratiotrain=1
    testsignalMLfilename=MLP6_DATA_PATH+'MLP6_selected'+suffix+'.bin'
    testnoiseMLfilename=MLP6_DATA_PATH+'MLP6_transient'+suffix+'.bin'
    
if testonhybrid:
    ratiotrain=1
    Ntest=nhyb
    testsignalMLfilename=MLHYB_DATA_PATH+'MLhybrid_selected'+suffix+'.bin'
    testnoiseMLfilename=MLHYB_DATA_PATH+'MLhybrid_transient'+suffix+'.bin'


#train/test sep already done 0.7/0.3
if trainingP6byevt:
    Ndata=979
    ratiotrain=1
    signalMLfilename=MLP6_DATA_PATH+'MLP6_trainbyevt_selected'+suffix+'.bin'
    noiseMLfilename=MLP6_DATA_PATH+'MLP6_trainbyevt_transient'+suffix+'.bin'
    Ntest=434
    testsignalMLfilename=MLP6_DATA_PATH+'MLP6_testbyevt_selected'+suffix+'.bin'
    testnoiseMLfilename=MLP6_DATA_PATH+'MLP6_testbyevt_transient'+suffix+'.bin'
    load=np.loadtxt(PBS_PATH+'P6testbyevt.txt',dtype='int')    
    test_ant=np.zeros((Ntest*sizedata))
    test_coinc=np.zeros((Ntest*sizedata)) 
    #run=load[:,0]
    test_coinc[0:Ntest]=load[:,1]
    test_ant[0:Ntest]=load[:,2]
    #evt=load[:,3]



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
    #plt.plot(input_data[0])
    #plt.savefig(PBS_PATH+'trace.png')
    #plt.close()
    for i in range(Ndata*sizedata):
        input_data[i,:,0]=input_data[i,:,0]-np.mean(input_data[i,:,0])
        input_data[i,:,0]=input_data[i,:,0]/quantization
        #ind=np.argmax(abs(input_data[i,:,0]))
        
        



    
'''fig,ax=plt.subplots(3, 3)

ax[0,0].plot(input_data[0])
ax[0,1].plot(input_data[1])
ax[0,2].plot(input_data[2])
ax[1,0].plot(input_data[3])
ax[1,1].plot(input_data[4])
ax[1,2].plot(input_data[5])
ax[2,0].plot(input_data[6])
ax[2,1].plot(input_data[7])
ax[2,2].plot(input_data[8])
plt.show()

fig,ax=plt.subplots(3, 3)

ax[0,0].plot(input_data[0+Ndata])
ax[0,1].plot(input_data[1+Ndata])
ax[0,2].plot(input_data[2+Ndata])
ax[1,0].plot(input_data[3+Ndata])
ax[1,1].plot(input_data[4+Ndata])
ax[1,2].plot(input_data[5+Ndata])
ax[2,0].plot(input_data[6+Ndata])
ax[2,1].plot(input_data[7+Ndata])
ax[2,2].plot(input_data[8+Ndata])
plt.show()'''


############ histo of dataset        



#simple separations
signal_data=input_data[0:Ndata,:,0]
noise_data=input_data[Ndata:Ndata*sizedata,:,0]
signal_labels=input_labels[0:Ndata,0]
noise_labels=input_labels[Ndata:Ndata*sizedata,0]
print(sum(signal_labels[np.max(abs(signal_data),1)>0.3]))
print(sum(noise_labels[np.max(abs(noise_data),1)>0.3]+1))

print(sum(signal_labels[(np.max(abs(signal_data),1)/np.std(signal_data,1))>10]))
print(sum(noise_labels[(np.max(abs(noise_data),1)/np.std(noise_data,1))>10]+1))


w=1
if unbalanced:
    w=sizedata-1
nbins=15

std=np.std(input_data[:,:,:],1)
ministd=np.min(std)
maxistd=np.max(std)
plt.hist(np.std(input_data[Ndata:Ndata*sizedata,:,0],1), bins=nbins, range=(ministd,maxistd), histtype='step', linewidth=2)
plt.hist(np.std(input_data[0:Ndata,:,0],1), bins=nbins, range=(ministd,maxistd), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Std')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig(PBS_PATH+'std.png')
if showstat:
    plt.show()
plt.close()

maxi=np.max(abs(input_data[:,:,:]),1)
minimaxi=np.min(maxi)
maximaxi=np.max(maxi)
plt.hist(np.max(abs(input_data[Ndata:Ndata*sizedata,:,0]),1), bins=nbins, range=(minimaxi,maximaxi), histtype='step', linewidth=2)
plt.hist(np.max(abs(input_data[0:Ndata,:,0]),1), bins=nbins, weights=np.zeros((Ndata))+w, range=(minimaxi,maximaxi), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Maximum')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig(PBS_PATH+'max.png')
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
plt.savefig(PBS_PATH+'argmax.png')
if showstat:
    plt.show()
plt.close()

maxstd=maxi/std
minimaxstd=np.min(maxstd)
maximaxstd=np.max(maxstd)
plt.hist(np.max(abs(input_data[Ndata:Ndata*sizedata,:,0]),1)/np.std(input_data[Ndata:Ndata*sizedata,:,0],1), bins=nbins, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.hist(np.max(abs(input_data[0:Ndata,:,0]),1)/np.std(input_data[0:Ndata,:,0],1), bins=nbins, weights=np.zeros((Ndata))+w, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Maximum [std unit]')
plt.legend(labels=['Noise', 'Signal'])
plt.savefig(PBS_PATH+'maxinstd')
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
sizetrainnoise=sizetrain-sizetrainsignal
sizetestnoise=sizetest-sizetestsignal


train_data=np.zeros((sizetrain,ib,1))
train_labels=np.zeros((sizetrain))
test_data=np.zeros((sizetest,ib,1))
test_labels=np.zeros((sizetest))

ind_list_signal = [i for i in range(Ndata)]
shuffle(ind_list_signal)
#print(ind_list_signal)
ind_list_noise = [i for i in range(Ndata,Ndata*sizedata)]
shuffle(ind_list_noise)
#print(ind_list_noise)


train_data[0:sizetrainsignal,:,:]  = input_data[ind_list_signal[0:sizetrainsignal],:,:]
train_data[sizetrainsignal:,:,:]  = input_data[ind_list_noise[0:sizetrainnoise], :,:]
train_labels[0:sizetrainsignal] = input_labels[ind_list_signal[0:sizetrainsignal],0]
train_labels[sizetrainsignal:] = input_labels[ind_list_noise[0:sizetrainnoise],0]

test_data[0:sizetestsignal,:,:]  = input_data[ind_list_signal[sizetrainsignal:],:,:]
test_data[sizetestsignal:,:,:]  = input_data[ind_list_noise[sizetrainnoise:], :,:]
test_labels[0:sizetestsignal] = input_labels[ind_list_signal[sizetrainsignal:],0]
test_labels[sizetestsignal:] = input_labels[ind_list_noise[sizetrainnoise:],0]



if trainingP6:
    train_ant=np.zeros((sizetrain))
    train_coinc=np.zeros((sizetrain))
    test_ant=np.zeros((sizetest))
    test_coinc=np.zeros((sizetest)) 
    
    train_ant[0:sizetrainsignal]=ant[ind_list_signal[0:sizetrainsignal]]
    train_coinc[0:sizetrainsignal]=coinc[ind_list_signal[0:sizetrainsignal]]    
    test_ant[0:sizetestsignal]=ant[ind_list_signal[sizetrainsignal:]]
    test_coinc[0:sizetestsignal]=coinc[ind_list_signal[sizetrainsignal:]]  
    

ind_list = [i for i in range(sizetrain)]
shuffle(ind_list)
train_data=train_data[ind_list]
train_labels=train_labels[ind_list]
if trainingP6:
    train_ant=train_ant[ind_list]
    train_coinc=train_coinc[ind_list]

ind_list = [i for i in range(sizetest)]
shuffle(ind_list)
test_data=test_data[ind_list]
test_labels=test_labels[ind_list]
if trainingP6:
    test_ant=test_ant[ind_list]
    test_coinc=test_coinc[ind_list]

    print(test_ant,test_coinc,test_labels)




#test a total random label
if stupidtest:
    for i in range(sizetrain):
        train_labels[i]=round(random())



print(sum(train_labels))


if testonP6 or testonhybrid or trainingP6byevt:
    
    test_data=np.zeros((Ntest*sizedata,ib,1))
    test_labels=np.zeros((Ntest*sizedata))

    for i in range(Ntest):
        test_labels[i]=1

    i=0
    with open(testsignalMLfilename,'rb') as fd:
        while i<Ntest:
            content=fd.read(ib)
            test_data[i,:,0]=struct.unpack('B'*ib,content)
            i=i+1
    with open(testnoiseMLfilename,'rb') as fd:
        while i<Ntest*sizedata:
            content=fd.read(ib)
            test_data[i,:,0]=struct.unpack('B'*ib,content)
            i=i+1
            
            
    if shiftscale:
        for i in range(Ntest*sizedata):
            test_data[i,:,0]=test_data[i,:,0]-np.mean(test_data[i,:,0])
            test_data[i,:,0]=test_data[i,:,0]/quantization
            


    
    '''fig,ax=plt.subplots(3, 3)

    ax[0,0].plot(test_data[0])
    ax[0,1].plot(test_data[1])
    ax[0,2].plot(test_data[2])
    ax[1,0].plot(test_data[3])
    ax[1,1].plot(test_data[4])
    ax[1,2].plot(test_data[5])
    ax[2,0].plot(test_data[6])
    ax[2,1].plot(test_data[7])
    ax[2,2].plot(test_data[8])
    plt.show()'''


'''if basic:
    train_data=train_data[:,:,0]
    test_data=test_data[:,:,0]'''
    
    
    
    
#make unbalanced balanced
if unbalanced:

    s_train_data=train_data[train_labels==1]
    n_train_data=train_data[train_labels==0]
    print(len(s_train_data),len(n_train_data))

    train_data=np.zeros(((sizedata-1)*sizetrainsignal+sizetrainnoise,ib,1))
    train_data[0:sizetrainsignal]=s_train_data
    train_data[sizetrainsignal:2*sizetrainsignal]=s_train_data
    train_data[2*sizetrainsignal:3*sizetrainsignal]=s_train_data
    train_data[3*sizetrainsignal:4*sizetrainsignal]=s_train_data
    train_data[3*sizetrainsignal:4*sizetrainsignal]=s_train_data
    train_data[4*sizetrainsignal:5*sizetrainsignal]=s_train_data
    train_data[5*sizetrainsignal:6*sizetrainsignal]=s_train_data
    train_data[6*sizetrainsignal:7*sizetrainsignal]=s_train_data
    train_data[7*sizetrainsignal:8*sizetrainsignal]=s_train_data
    train_data[8*sizetrainsignal:9*sizetrainsignal]=s_train_data
    train_data[9*sizetrainsignal:10*sizetrainsignal]=s_train_data
    train_data[10*sizetrainsignal:11*sizetrainsignal]=s_train_data
    train_data[11*sizetrainsignal:12*sizetrainsignal]=s_train_data
    train_data[12*sizetrainsignal:13*sizetrainsignal]=s_train_data
    train_data[13*sizetrainsignal:14*sizetrainsignal]=s_train_data
    train_data[14*sizetrainsignal:15*sizetrainsignal]=s_train_data
    train_data[15*sizetrainsignal:16*sizetrainsignal]=s_train_data
    train_data[16*sizetrainsignal:17*sizetrainsignal]=s_train_data
    train_data[17*sizetrainsignal:18*sizetrainsignal]=s_train_data
    train_data[18*sizetrainsignal:19*sizetrainsignal]=s_train_data
    train_data[19*sizetrainsignal:20*sizetrainsignal]=s_train_data    
    train_data[20*sizetrainsignal:]=n_train_data
    
    train_labels=np.zeros(((sizedata-1)*sizetrainsignal+sizetrainnoise))
    train_labels[0:(sizedata-1)*sizetrainsignal]=1
         
    s_test_data=test_data[test_labels==1]
    n_test_data=test_data[test_labels==0]

    test_data=np.zeros(((sizedata-1)*sizetestsignal+sizetestnoise,ib,1))
    test_data[0:sizetestsignal]=s_test_data
    test_data[sizetestsignal:2*sizetestsignal]=s_test_data
    test_data[2*sizetestsignal:3*sizetestsignal]=s_test_data
    test_data[3*sizetestsignal:4*sizetestsignal]=s_test_data
    test_data[3*sizetestsignal:4*sizetestsignal]=s_test_data
    test_data[4*sizetestsignal:5*sizetestsignal]=s_test_data
    test_data[5*sizetestsignal:6*sizetestsignal]=s_test_data
    test_data[6*sizetestsignal:7*sizetestsignal]=s_test_data
    test_data[7*sizetestsignal:8*sizetestsignal]=s_test_data    
    test_data[8*sizetestsignal:9*sizetestsignal]=s_test_data 
    test_data[9*sizetestsignal:10*sizetestsignal]=s_test_data
    test_data[10*sizetestsignal:11*sizetestsignal]=s_test_data
    test_data[11*sizetestsignal:12*sizetestsignal]=s_test_data
    test_data[12*sizetestsignal:13*sizetestsignal]=s_test_data
    test_data[13*sizetestsignal:14*sizetestsignal]=s_test_data
    test_data[14*sizetestsignal:15*sizetestsignal]=s_test_data
    test_data[15*sizetestsignal:16*sizetestsignal]=s_test_data
    test_data[16*sizetestsignal:17*sizetestsignal]=s_test_data
    test_data[17*sizetestsignal:18*sizetestsignal]=s_test_data
    test_data[18*sizetestsignal:19*sizetestsignal]=s_test_data
    test_data[19*sizetestsignal:20*sizetestsignal]=s_test_data   
    test_data[20*sizetestsignal:]=n_test_data
     
    test_labels=np.zeros(((sizedata-1)*sizetestsignal+sizetestnoise))
    test_labels[0:(sizedata-1)*sizetestsignal]=1    
        
    print(sizedata,sizetestsignal,sizetestnoise,sum(test_labels))
            
    ind_list = [i for i in range((sizedata-1)*sizetrainsignal+sizetrainnoise)]
    shuffle(ind_list)
    train_data=train_data[ind_list]
    train_labels=train_labels[ind_list]
    ind_list = [i for i in range((sizedata-1)*sizetestsignal+sizetestnoise)]
    shuffle(ind_list)
    test_data=test_data[ind_list]
    test_labels=test_labels[ind_list]



############ FFT options

#with FFT alone
if FFTalone:
    #ib=int(ib/2+1)
    fourier=np.zeros((Ndata*sizedata,ib,1))
    for i in range(Ndata*sizedata):
        fourier[i,:,0]=rfft(train_data[i,:,0])
        #fourier[i,:,0]=np.real(np.fft.rfft(train_data[i,:,0]))        
    train_data=fourier
    fourier=np.zeros((Ntest*sizedata,ib,1))
    for i in range(Ntest*sizedata):
        fourier[i,:,0]=rfft(test_data[i,:,0])
        #fourier[i,:,0]=np.real(np.fft.rfft(test_data[i,:,0]))
    test_data=fourier    

#with FFT and trace
if FFT:
    ibfft=ib
    ftrain_data=np.zeros((len(train_data),ib,1))
    for i in range(len(ftrain_data)):
        ftrain_data[i,:,0]=rfft(train_data[i,:,0])
    ftest_data=np.zeros((len(test_data),ib,1))
    for i in range(len(ftest_data)):
        ftest_data[i,:,0]=rfft(test_data[i,:,0])

#with FFTreal FFTimag and temporal trace        
if FFTri: 
    ibfft=int(ib/2+1)
    rftrain_data=np.zeros((len(train_data),ibfft,1))
    iftrain_data=np.zeros((len(train_data),ibfft,1))    
    for i in range(len(train_data)):
        rftrain_data[i,:,0]=np.real(np.fft.rfft(train_data[i,:,0]))
        iftrain_data[i,:,0]=np.imag(np.fft.rfft(train_data[i,:,0]))        
    rftest_data=np.zeros((len(test_data),ibfft,1))
    iftest_data=np.zeros((len(test_data),ibfft,1))    
    for i in range(len(test_data)):
        rftest_data[i,:,0]=np.real(np.fft.rfft(test_data[i,:,0]))
        iftest_data[i,:,0]=np.imag(np.fft.rfft(test_data[i,:,0]))        



############ build model

print('modelbeg')
trainmet=[]
valmet=[]
testmet=[]
#kernels=[ (3,),  (9,), (15,), (21,), (27,), (33,), (39,), (45,), (51,), (57,), (63,), (69,), (75,), (81,), (87,)]
#filters=[8,16,32]
kernels=[(51,)]
fkernel=(21,)
#kernels=[(27,),(31,),(35,),(39,),(43,),(47,),(51,)]
filters=[16]
ntry=1

for f in filters:
    for k in kernels:
        for t in range(0,ntry):
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
                drop=0.5
                #pad='valid'
                pad='same'
                #regul=0
                #drop=0
                
                input1=keras.layers.Input(shape=(ib,1))
                conv1=keras.layers.Conv1D(f, k, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(input1)
                maxpool1=keras.layers.MaxPooling1D(2)(conv1)
                drop1=keras.layers.Dropout(drop)(maxpool1)
                conv2=keras.layers.Conv1D(2*f, k, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(drop1)
                maxpool2=keras.layers.MaxPooling1D(2)(conv2)
                drop2=keras.layers.Dropout(drop)(maxpool2)
                conv3=keras.layers.Conv1D(2*f, k, activation='relu',padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(drop2)
                flat=keras.layers.Flatten()(conv3)
                drop3=keras.layers.Dropout(drop)(flat)
                dense1=keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(drop3)

                
                if FFT:
                    finput1=keras.layers.Input(shape=(ibfft,1))
                    fconv1=keras.layers.Conv1D(f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(finput1)
                    fmaxpool1=keras.layers.MaxPooling1D(2)(fconv1)
                    fdrop1=keras.layers.Dropout(drop)(fmaxpool1)
                    fconv2=keras.layers.Conv1D(2*f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(fdrop1)
                    fmaxpool2=keras.layers.MaxPooling1D(2)(fconv2)
                    fdrop2=keras.layers.Dropout(drop)(fmaxpool2)
                    fconv3=keras.layers.Conv1D(2*f, fkernel, activation='relu',padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(fdrop2)
                    fflat=keras.layers.Flatten()(fconv3)
                    fdrop3=keras.layers.Dropout(drop)(fflat)
                    fdense1=keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(fdrop3)
                                
                    concat=keras.layers.concatenate([dense1, fdense1])
                    drop4=keras.layers.Dropout(drop)(concat)
                    dense2=keras.layers.Dense(2, activation='softmax')(drop4)
                    model=keras.models.Model(inputs=[input1,finput1], outputs=[dense2])
                    
                if FFTri:
                    rfinput1=keras.layers.Input(shape=(ibfft,1))
                    rfconv1=keras.layers.Conv1D(f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(rfinput1)
                    rfmaxpool1=keras.layers.MaxPooling1D(2)(rfconv1)
                    rfdrop1=keras.layers.Dropout(drop)(rfmaxpool1)
                    rfconv2=keras.layers.Conv1D(2*f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(rfdrop1)
                    rfmaxpool2=keras.layers.MaxPooling1D(2)(rfconv2)
                    rfdrop2=keras.layers.Dropout(drop)(rfmaxpool2)
                    rfconv3=keras.layers.Conv1D(2*f, fkernel, activation='relu',padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(rfdrop2)
                    rfflat=keras.layers.Flatten()(rfconv3)
                    rfdrop3=keras.layers.Dropout(drop)(rfflat)
                    rfdense1=keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(rfdrop3)
                    
                    ifinput1=keras.layers.Input(shape=(ibfft,1))
                    ifconv1=keras.layers.Conv1D(f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(ifinput1)
                    ifmaxpool1=keras.layers.MaxPooling1D(2)(ifconv1)
                    ifdrop1=keras.layers.Dropout(drop)(ifmaxpool1)
                    ifconv2=keras.layers.Conv1D(2*f, fkernel, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(ifdrop1)
                    ifmaxpool2=keras.layers.MaxPooling1D(2)(ifconv2)
                    ifdrop2=keras.layers.Dropout(drop)(ifmaxpool2)
                    ifconv3=keras.layers.Conv1D(2*f, fkernel, activation='relu',padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(ifdrop2)
                    ifflat=keras.layers.Flatten()(ifconv3)
                    ifdrop3=keras.layers.Dropout(drop)(ifflat)
                    ifdense1=keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul))(ifdrop3)                    
                               
                    concat=keras.layers.concatenate([dense1, rfdense1, ifdense1])
                    drop4=keras.layers.Dropout(drop)(concat)
                    dense2=keras.layers.Dense(2, activation='softmax')(drop4)
                    model=keras.models.Model(inputs=[input1,rfinput1,ifinput1], outputs=[dense2])
                    
                if FFT==0 and FFTri==0:
                
                    drop4=keras.layers.Dropout(drop)(dense1)
                    dense2=keras.layers.Dense(2, activation='softmax')(drop4)
                    model=keras.models.Model(inputs=[input1], outputs=[dense2])
                   
                '''model = keras.Sequential([
                  keras.layers.Conv1D(f, k, activation='relu', input_shape=(ib,1), padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)), #input shape = x,channel(cannot be none or nothing) ....  timestep,features              
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.MaxPooling1D(2),
                  keras.layers.Dropout(drop),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Conv1D(2*f, k, activation='relu', padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.MaxPooling1D(2),
                  keras.layers.Dropout(drop),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Conv1D(2*f, k, activation='relu',padding=pad, kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Flatten(),
                  keras.layers.Dropout(drop),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Dense(2*f, activation='relu',kernel_regularizer=keras.regularizers.l2(regul), bias_regularizer=keras.regularizers.l2(regul)),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Dropout(drop),
                  #keras.layers.BatchNormalization(), #BATCHNORM
                  keras.layers.Dense(2),#,activation='softmax'),
                  keras.layers.Softmax()
                ])'''



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

            EPOCHS = 80
            strt_time = datetime.datetime.now()
            if FFT:
                history = model.fit((train_data, ftrain_data), train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)#, callbacks=[callback])
            if FFTri:
                history = model.fit((train_data, rftrain_data, iftrain_data), train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)#, callbacks=[callback])
            if FFT==0 and FFTri==0:
                history = model.fit(train_data, train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)#, callbacks=[callback])
            #history = model.fit(basictrain_data, train_labels, epochs=EPOCHS, verbose=1, batch_size=32, validation_split=validintrain)
            curr_time = datetime.datetime.now()
            timedelta = curr_time - strt_time
            dnn_train_time = timedelta.total_seconds()
            print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
            #print(history.history['val_loss'])

            if FFT:
                test_loss, test_met = model.evaluate((test_data,ftest_data), test_labels, verbose=2)
            if FFTri:
                test_loss, test_met = model.evaluate((test_data,rftest_data,iftest_data), test_labels, verbose=2)                
            if FFT==0 and FFTri==0:
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
plt.savefig(PBS_PATH+'loss.png')
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
plt.savefig(PBS_PATH+'accuracy')
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


#model.save(SPS_PATH+'model')
#model = keras.models.load_model('path/to/location')


############ predictions

if FFT:
    predictions = model.predict((test_data,ftest_data))
if FFTri:
    predictions = model.predict((test_data,rftest_data,iftest_data))
if FFT==0 and FFTri==0:
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
    if len(predlabelnoise)!=0:
        keepnoise[i]=len(predlabelnoise[predlabelnoise>=threshold[i]])/len(predlabelnoise)
    
plt.plot(threshold,keepnoise)
plt.plot(threshold,keepsignal)
plt.plot(threshold,(keepsignal+1-keepnoise)/2,'r',linewidth=2, linestyle='dotted')
plt.xlabel('Decision threshold')
plt.ylabel('Selection rate (test set)')
#plt.title('Trace level')
plt.legend(labels=['Noise', 'Signal', 'Accuracy'])
plt.grid()
plt.savefig(PBS_PATH+'threshold.png')
if showtrain:
    plt.show()
plt.close()

'''
plt.plot(test_data[predictions[:,1]>0.9][0])
plt.show()
plt.plot(test_data[predictions[:,1]>0.9][1])
plt.show()
plt.plot(test_data[predictions[:,1]>0.9][2])
plt.show()
'''


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
plt.savefig(PBS_PATH+'prediction.png')
if showtrain:
    plt.show()
plt.close()



#stat of predictions
std=np.std(test_data[:,:,:],1)
maxi=np.max(abs(test_data[:,:,:]),1)

predsignal_test_data=test_data[predictions[:,1]>0.5]
prednoise_test_data=test_data[predictions[:,1]<=0.5]

maxstd=maxi/std
minimaxstd=np.min(maxstd)
maximaxstd=np.max(maxstd)
plt.hist(np.max(abs(prednoise_test_data),1)/np.std(prednoise_test_data,1), bins=nbins, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.hist(np.max(abs(predsignal_test_data),1)/np.std(predsignal_test_data,1), bins=nbins, range=(minimaxstd,maximaxstd), histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Maximum [std unit]')
plt.legend(labels=['Pred as noise', 'Pred as signal'])
plt.savefig(PBS_PATH+'maxinstd_pred')
plt.close()


if trainingP6byevt:

    coincunique=np.unique(test_coinc)
    coincid=np.zeros(len(test_coinc))
    for i in range(len(coincunique)):
        for j in range(len(test_coinc)):
            if test_coinc[j]==coincunique[i]:
                coincid[j]=i
                
    antunique=np.unique(test_ant)
    antid=np.zeros(len(test_ant))
    antmin=np.min(antunique[antunique!=0])
    for i in range(len(antunique)):
        for j in range(len(test_ant)):
            if test_ant[j]==antunique[i]:
                antid[j]=i+antmin-1                
                

    
    antidunique=np.unique(antid)
    meanpred=np.zeros(len(antidunique))
    stdpred=np.zeros(len(antidunique))
    for i in range(len(antidunique)):
        meanpred[i]=np.mean(predictions[:,1][antid==antidunique[i]])
        stdpred[i]=np.std(predictions[:,1][antid==antidunique[i]])
    print(antidunique,meanpred)
    plt.scatter(antid[coincid!=0],predictions[:,1][coincid!=0],c='tab:orange',alpha=0.8)
    #plt.errorbar(antunique[1:],meanpred[1:],yerr=stdpred[1:],c='r',label='Averaged prediction')
    plt.plot(antidunique[1:],meanpred[1:],c='r',label='Averaged prediction')
    plt.fill_between(antidunique[1:],meanpred[1:]-stdpred[1:],meanpred[1:]+stdpred[1:],color='tab:red',label='Standard deviation',alpha=0.3)

    plt.grid()
    plt.xlabel('Unbroken antenna ID')
    plt.ylabel('Prediction probability to be a signal (dots)')
    plt.legend()
    plt.title('Signal labels only')
    plt.savefig(PBS_PATH+'antennas')
    plt.close()
    
    
    
    from scipy.stats import binom
    #binom.pmf(x, n, p)
    #proba de x+ succes sur n essais avec proba de p par essai=
    #1-binom.cdf()
    indep=np.zeros(100)
    for i in range(100):
        indep[i]=1-(binom.cdf(5,8,keepsignal[i]))
    
    keepevent=np.zeros(100)
    threshold=np.zeros(100)
    coincidunique=np.unique(coincid)
    meanpred=np.zeros(len(coincidunique))
    stdpred=np.zeros(len(coincidunique))
    for i in range(len(coincidunique[1:])):
        pred=predictions[:,1][coincid==coincidunique[i]]
        for j in range(100):
            threshold[j]=j/100
            if len(pred[pred>=threshold[j]]) >=5:
                keepevent[j]=keepevent[j]+1
            
        meanpred[i]=np.mean(pred)
        stdpred[i]=np.std(pred)
        
    plt.plot(threshold,keepevent/len(coincidunique[1:]),label='Event level' )
    plt.plot(threshold,keepsignal,linestyle='dotted',label='Trace level (dep)', linewidth=2)
    plt.plot(threshold,indep,linestyle='dotted',c='r',label='1-binom.cdf(5,8,trace level) (indep)', linewidth=2)
    plt.grid()
    plt.title('5+ traces (signal labels only)')
    plt.xlabel('Decision threshold')
    plt.ylabel('Selection rate (test set)')
    plt.legend()
    plt.savefig(PBS_PATH+'thresholdevts')
    plt.close()
    

    print(coincidunique,meanpred,keepevent)
    plt.scatter(coincid[coincid!=0],predictions[:,1][coincid!=0],c='tab:orange',alpha=0.8)
    #plt.errorbar(coincidunique[1:],meanpred[1:],yerr=stdpred[1:],c='r',label='Averaged prediction')
    plt.plot(coincidunique[1:],meanpred[1:],c='r',label='Averaged prediction')
    plt.fill_between(coincidunique[1:],meanpred[1:]-stdpred[1:],meanpred[1:]+stdpred[1:],color='tab:red',label='Standard deviation', alpha=0.3)
    plt.grid()
    plt.xlabel('Coinc ID')
    plt.ylabel('Prediction probability to be a signal (dots)')
    plt.legend()
    plt.title('Signal labels only')
    plt.savefig(PBS_PATH+'coincs')
    plt.close()




############ true/false signal/noise



#from matplotlib.pyplot import figure
plt.figsize=(20, 12)

falsesignals=test_data[(test_labels==0) & (predictions[:,1]>=0.5)]
#misclassified traces
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(falsesignals[0])
ax[0,1].plot(falsesignals[1])
ax[0,2].plot(falsesignals[2])
ax[1,0].plot(falsesignals[3])
ax[1,1].plot(falsesignals[4])
ax[1,2].plot(falsesignals[5])
ax[2,0].plot(falsesignals[6])
ax[2,1].plot(falsesignals[7])
ax[2,2].plot(falsesignals[8])
plt.savefig(PBS_PATH+'falsesignals.png')
#plt.show()
plt.close()


falsenoises=test_data[(test_labels==1) & (predictions[:,1]<0.5)]
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(falsenoises[0])
ax[0,1].plot(falsenoises[1])
ax[0,2].plot(falsenoises[2])
ax[1,0].plot(falsenoises[3])
ax[1,1].plot(falsenoises[4])
if len(falsenoises)>5:
    ax[1,2].plot(falsenoises[5])
if len(falsenoises)>6:
    ax[2,0].plot(falsenoises[6])
if len(falsenoises)>7:
    ax[2,1].plot(falsenoises[7])
if len(falsenoises)>8: 
    ax[2,2].plot(falsenoises[8])
plt.savefig(PBS_PATH+'falsenoises.png')
#plt.show()
plt.close()


truesignals=test_data[(test_labels==1) & (predictions[:,1]>=0.8)]
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(truesignals[0])
ax[0,1].plot(truesignals[1])
ax[0,2].plot(truesignals[2])
ax[1,0].plot(truesignals[3])
ax[1,1].plot(truesignals[4])
ax[1,2].plot(truesignals[5])
ax[2,0].plot(truesignals[6])
ax[2,1].plot(truesignals[7])
ax[2,2].plot(truesignals[8])
plt.savefig(PBS_PATH+'truesignals.png')
#plt.show()
plt.close()

truenoises=test_data[(test_labels==0) & (predictions[:,1]<0.2)]
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(truenoises[0])
ax[0,1].plot(truenoises[1])
ax[0,2].plot(truenoises[2])
ax[1,0].plot(truenoises[3])
ax[1,1].plot(truenoises[4])
if len(truenoises)>5:
    ax[1,2].plot(truenoises[5])
if len(truenoises)>6:
    ax[2,0].plot(truenoises[6])
if len(truenoises)>7:
    ax[2,1].plot(truenoises[7])
if len(truenoises)>8:
    ax[2,2].plot(truenoises[8])
plt.savefig(PBS_PATH+'truenoises.png')
#plt.show()
plt.close()








############ clustering


'''model = keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
predictions = model.predict(input_data[:Ndata])
plt.plot(predictions)
plt.show()

from sklearn.cluster import KMeans
print("on last hidden")
kmeans = KMeans(n_clusters=2)
kmeans.fit(predictions)
print(np.shape(predictions))
#plt.scatter(np.arange(0,Ndata*2),kmeans.labels_,c='tab:orange',alpha=0.8)
#plt.savefig(PBS_PATH+'kmeans.png')
print(sum(kmeans.labels_[:Ndata]))
#print(sum(kmeans.labels_[Ndata:]))



cluster=(input_data[:Ndata])[kmeans.labels_==0]
print(np.shape(cluster))
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(cluster[0])
ax[0,1].plot(cluster[1])
ax[0,2].plot(cluster[2])
ax[1,0].plot(cluster[3])
ax[1,1].plot(cluster[4])
ax[1,2].plot(cluster[5])
ax[2,0].plot(cluster[6])
ax[2,1].plot(cluster[7])
ax[2,2].plot(cluster[8])
plt.savefig(PBS_PATH+'kmeans0.png')
plt.show()
plt.close()


cluster=(input_data[:Ndata])[kmeans.labels_==1]
print(np.shape(cluster))
fig,ax=plt.subplots(3, 3)

ax[0,0].plot(cluster[0])
ax[0,1].plot(cluster[1])
ax[0,2].plot(cluster[2])
ax[1,0].plot(cluster[3])
ax[1,1].plot(cluster[4])
ax[1,2].plot(cluster[5])
ax[2,0].plot(cluster[6])
ax[2,1].plot(cluster[7])
ax[2,2].plot(cluster[8])
plt.savefig(PBS_PATH+'kmeans1.png')
plt.show()
plt.close()



print("on input")
kmeans = KMeans(n_clusters=2)
kmeans.fit(input_data[:Ndata,:,0])
print(np.shape(input_data[:Ndata,:,0]))
print(sum(kmeans.labels_[:Ndata]))
#print(sum(kmeans.labels_[Ndata:]))



'''


















