import numpy as np
import subprocess #to launch linux commands
from os import listdir
import os.path
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import sys


import tensorflow as tf
from tensorflow import keras
import datetime

from shared import *


run=sys.argv[1]

job=1
if job:
    TMP_PATH=sys.argv[2]
else:
    TMP_PATH=SPS_PATH

#pdir=subprocess.Popen('iget -vfr '+RAW_DATA_PATH+run+' '+TMP_PATH, shell=True) 
#pdir.wait()


############ load model

#model = keras.models.load_model(SPS_PATH+'model')
model = keras.models.load_model(SPS_PATH+'modelFFT')


############ load data to make prediction on

shiftscale=1
FFT=1
if FFT:
    from scipy.fftpack import rfft
#run='R003564'

if os.path.isdir(SPS_PATH+'log_classpredict/'+run)!=1:
    pdir=subprocess.Popen('mkdir '+SPS_PATH+'log_classpredict/'+run, shell=True) 
    pdir.wait()

binfolders  = listdir(TMP_PATH+run)
print(binfolders)


for b in range(len(binfolders)):

    if binfolders[b][14:] != 'data.bin':
        continue

    
    datafilename=TMP_PATH+run+'/'+binfolders[b]
    filesize=os.path.getsize(datafilename)

    print(binfolders[b], filesize)

    ID=binfolders[b][10:13]

    with open(SPS_PATH+'log_classpredict/'+run+'/'+'log_classpredict_'+run+'_'+ID+'.txt','w') as f:
        f.write('')

    packsize=10000 # we divide because data is too long
    npack=int(filesize/ib/packsize)            
    remainder=int(filesize/ib)-npack*packsize
    print(npack,remainder)
	
    with open(datafilename,'rb') as fd:
        for j in range(npack+1): 
            print(j)
            if j==npack:
                packsize=remainder 
            input_data=np.zeros((packsize,ib,1)) 
            k=0
            while k<packsize:
                content=fd.read(ib)
                input_data[k,:,0]=struct.unpack('B'*ib,content)
                if shiftscale:
                    input_data[k,:,0]=input_data[k,:,0]-np.mean(input_data[k,:,0])
                    input_data[k,:,0]=input_data[k,:,0]/quantization
                if FFT:
                    input_data[k,:,0]=rfft(input_data[k,:,0])
                k=k+1

            print('begin pred')
            predictions=model.predict(input_data)

            with open(SPS_PATH+'log_classpredict/'+run+'/'+'log_classpredict_'+run+'_'+ID+'.txt','a') as f:
                for i in range(packsize):
                    f.write(str(round(predictions[i,1]*100))+'\n')
            
            
            
            
            
            
