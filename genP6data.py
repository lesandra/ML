import numpy as np
import subprocess #to launch linux commands
import os.path
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib.pyplot as plt
import sys


from shared import *





selected=np.zeros((nP6+1,ib))
transient=np.zeros((nP6+1,ib))

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


j=0
for r in rununique:
    for a in antunique:        
        binaryfile=P6_DATA_PATH+'R00'+str(r)+'_A0'+str(a)+'_data.bin' 
	
	if os.path.isfile(binaryfile):
	    filesize=os.path.getsize(binaryfile)
	    #print(filesize)
	    
	    with open(binaryfile,'rb') as fd:
	        for i in range(len(run)):
                    if run[i]==r and ant[i]==a:
		        
		        #print(run[i],ant[i],evt[i])
	                fd.seek(ib*(evt[i]-1))
        	        content=fd.read(ib)
        	        selected[j]=struct.unpack('B'*ib,content)
			
			
			if np.max(selected[j])!=satup and np.min(selected[j])!=satdown:			
			    
			    randompos=int(random()*filesize/1024)
			    #print(randompos)
		            fd.seek(ib*(randompos))
        	            content=fd.read(ib)
        	            transient[j]=struct.unpack('B'*ib,content)
			    
			    while np.max(transient[j])==satup or np.min(transient[j])==satdown:
			    
			        randompos=int(random()*filesize/1024)
			        #print(randompos)
		                fd.seek(ib*(randompos))
        	                content=fd.read(ib)
        	                transient[j]=struct.unpack('B'*ib,content)
			
			    j=j+1

			
			#if coinc[i]==38605:
			 #   plt.plot(selected[j-1])
			  #  plt.show()
			
			


print(j)
maxsel=np.max(selected[0:j],1)
maxtrans=np.max(transient[0:j],1)
stdsel=np.mean(np.std(selected[0:j],1))
stdtrans=np.mean(np.std(transient[0:j],1))
print(stdsel)
print(stdtrans)
print(maxsel)
print(len(maxsel[maxsel==255]))
print(maxtrans)
print(len(maxtrans[maxtrans==255]))




selectedMLfilename=MLP6_DATA_PATH+'MLP6_selected'+suffix+'.bin'
with open(selectedMLfilename,'wb') as fd:
    for i in range(0,j):
        for k in range(0,ib):
            content=struct.pack('B',selected[i,k])
            fd.write(content)
	    
transientMLfilename=MLP6_DATA_PATH+'MLP6_transient'+suffix+'.bin'
with open(transientMLfilename,'wb') as fd:
    for i in range(0,j):
        for k in range(0,ib):
            content=struct.pack('B',transient[i,k])
            fd.write(content)	
	    
	    
	    
############ check files content
	


with open(selectedMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    plt.plot(content[0:1024*1])
    plt.savefig('selected.png')
    plt.close()
	
with open(transientMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    plt.plot(content[0:1024*1])
    plt.savefig('transient.png')
    plt.close()
    
	    
	    
	    




