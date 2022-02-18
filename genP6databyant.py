import numpy as np
import subprocess #to launch linux commands
import os.path
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib.pyplot as plt
import sys


from shared import *



threshold=50


if tpsel:
    load=np.loadtxt(SPS_PATH+'runcoincantevtthetaphisel_space.txt',dtype='int')
else:
    load=np.loadtxt(SPS_PATH+'runcoincantevt_space.txt',dtype='int')
    
run=load[:,0]
coinc=load[:,1]
ant=load[:,2]
evt=load[:,3]
print(ant)
rununique=np.unique(run)
antunique=np.unique(ant)

cpt=np.zeros(len(antunique),dtype='int')

for i in range(len(antunique)): 
    print(antunique[i])
    for antenna in ant:
        #print(antenna)
        if antunique[i]==antenna:
            cpt[i]=cpt[i]+1
print(antunique)            
print(cpt)
print(sum(cpt))


antenough=antunique[cpt>=threshold]

selected=np.zeros((len(antenough),max(cpt),ib))
transient=np.zeros((len(antenough),max(cpt),ib))

tracecpt=np.zeros(len(antenough),dtype='int')

print(antenough)
print(tracecpt)

for r in rununique:
    for a in antunique:   
    
        occur=cpt[antunique==a]
        
        binaryfile=P6_DATA_PATH+'R00'+str(r)+'_A0'+str(a)+'_data.bin' 
        
        
        if os.path.isfile(binaryfile) and occur>=threshold:
            filesize=os.path.getsize(binaryfile)

            
            with open(binaryfile,'rb') as fd:
                for i in range(len(run)):
                    if run[i]==r and ant[i]==a:
                        
                        #print(run[i],ant[i],evt[i])
                        fd.seek(ib*(evt[i]-1))
                        content=fd.read(ib)
                        #print(tracecpt[antenough==a])
                        selected[antenough==a,tracecpt[antenough==a]]=struct.unpack('B'*ib,content)
                        
                        
                        if np.max(selected[antenough==a,tracecpt[antenough==a]])!=satup and np.min(selected[antenough==a,tracecpt[antenough==a]])!=satdown:                        
                            
                            randompos=int(random()*filesize/1024)
                            #print(randompos)
                            fd.seek(ib*(randompos))
                            content=fd.read(ib)
                            transient[antenough==a,tracecpt[antenough==a]]=struct.unpack('B'*ib,content)
                            
                            while np.max(transient[antenough==a,tracecpt[antenough==a]])==satup or np.min(transient[antenough==a,tracecpt[antenough==a]])==satdown:
                            
                                randompos=int(random()*filesize/1024)
                                #print(randompos)
                                fd.seek(ib*(randompos))
                                content=fd.read(ib)
                                transient[antenough==a,tracecpt[antenough==a]]=struct.unpack('B'*ib,content)
                        
                            tracecpt[antenough==a]=tracecpt[antenough==a]+1





print(tracecpt)
for a in antenough:
    
    print(tracecpt[antenough==a][0])
    sel=selected[antenough==a][0]
    trans=transient[antenough==a][0]
    
    lensel=tracecpt[antenough==a][0]
    print(np.shape(selected),np.shape(sel))

    selectedMLfilename=MLP6_DATA_PATH+'MLP6_'+str(a)+'_selected'+suffix+'.bin'
    with open(selectedMLfilename,'wb') as fd:
        for i in range(0,lensel):
            print(i)
            for k in range(0,ib):
                #print(i,k,sel[i,k])
                content=struct.pack('B',int(sel[i,k]))
                fd.write(content)

    transientMLfilename=MLP6_DATA_PATH+'MLP6_'+str(a)+'_transient'+suffix+'.bin'
    with open(transientMLfilename,'wb') as fd:
        for i in range(0,tracecpt[antenough==a][0]):
            for k in range(0,ib):
                content=struct.pack('B',int(trans[i,k]))
                fd.write(content)        

'''

    with open(selectedMLfilename,'rb') as fd:
        content=fd.read()
        content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
        print(len(content))    
        #plt.plot(content)
        #size=int(len(content))
        #plt.plot(content[0:1024*1])
        #plt.show()
        #plt.close()

    with open(transientMLfilename,'rb') as fd:
        content=fd.read()
        content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
        print(len(content))    
        plt.plot(content)
        #size=int(len(content))
        #plt.plot(content[0:1024*1])
        #plt.show()
        #plt.close()
'''



