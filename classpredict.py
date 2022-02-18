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


############ load model

model = keras.models.load_model(SPS_PATH+'model')


############ load data to make prediction on

shiftscale=1
run='R003564'


binfolders  = listdir(SPS_PATH+run)

for b in range(len(binfolders)):



    if binfolders[b][14:] != 'data.bin':
        continue

    print(binfolders[b])
    datafilename=SPS_PATH+run+'/'+binfolders[b]
    filesize=os.path.getsize(datafilename)

    ID=binfolders[b][10:13]

    input_data=np.zeros((int(filesize/ib),ib,1))

    i=0
    with open(datafilename,'rb') as fd:
        while i<filesize/ib:
            content=fd.read(ib)
            input_data[i,:,0]=struct.unpack('B'*ib,content)
            i=i+1

    if shiftscale:
        for i in range(int(filesize/ib)):
            input_data[i,:,0]=input_data[i,:,0]-np.mean(input_data[i,:,0])
            input_data[i,:,0]=input_data[i,:,0]/quantization

    #plt.plot(input_data[1])
    #plt.show()
    predictions=model.predict(input_data)
    #print(predictions[1])
    
                
    with open(SPS_PATH+'log_classpredict/log_classpredict_'+run+'_'+ID+'.txt','w') as f:
        for i in range(int(filesize/ib)):
            f.write(str(round(predictions[i,1]*100))+'\n')
	    
	    
	    
	    
	    
	    
