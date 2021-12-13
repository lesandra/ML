import os
import numpy as np
import struct
import matplotlib.pyplot as plt


signalfilename='/sps/trend/slecoz/MLP6SIM/MLP6SIM_simunoise.bin'
noisefilename='/sps/trend/slecoz/MLP6SIM/MLP6SIM_transient.bin'

filesize=os.path.getsize(signalfilename)
Ndata=int(filesize/1024)
noise=np.zeros((Ndata,1024))
signal=np.zeros((Ndata,1024))

i=0
with open(signalfilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(1024)
        signal[i]=struct.unpack('B'*1024,content)
        i=i+1

i=0
with open(noisefilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(1024)
        noise[i]=struct.unpack('B'*1024,content)
        i=i+1
	
	
plt.plot(noise[0])
plt.show()
