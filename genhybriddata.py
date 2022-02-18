import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from os import listdir
from random import *
import os

ib=1024


load=np.loadtxt('/sps/trend/slecoz/'+'runcoinchybrid_space.txt',dtype='int')
run=load[:,0]
coinc=load[:,1]
ant=load[:,2]
evt=load[:,3]
print(ant)
load=np.loadtxt('/pbs/throng/trend/soft/ana/TREND_ML/'+'maxinstd.txt')
hybmaxinstd=load[:,2]
mean=load[:,3]
std=load[:,4]

minmin=np.min((np.min(hybmaxinstd),np.min(mean)))
maxmax=np.max((np.max(hybmaxinstd),np.max(mean)))
print(minmin,maxmax)
plt.scatter(mean,hybmaxinstd,c='tab:orange',alpha=0.8)
plt.plot(np.linspace(minmin,maxmax),np.linspace(minmin,maxmax))
plt.grid()
plt.ylabel('Maximum [std unit]')
plt.xlabel('(run,ant) mean of Maximum [std unit]')
plt.title('"Hybrid" dataset')
plt.savefig("maxinstd_hybridruns.png")
plt.show()


unbal=0
#need to change also the name of the file where to save

hybrid=np.zeros((len(run),ib))
transient=np.zeros((len(run)*unbal,ib))
cpthyb=0
cpttrans=0

nhyb=42
maxinstdhyb=np.zeros((nhyb))
nbins=18
meanmaxinstd=np.zeros((nhyb))
stdmaxinstd=np.zeros((nhyb))

for i in range(0,42):
   
   #if ant[i]!=139 and ant[i]!=175 and ant[i]!=176:
   if 1:
   
        binaryfile='/sps/trend/slecoz/bin_selected_hybrid/'+'R00'+str(run[i])+'_A0'+str(ant[i])+'_data.bin' 

        filesize=os.path.getsize(binaryfile)

        with open(binaryfile,'rb') as fd:


            print(run[i],ant[i],evt[i],mean[i],std[i])
            fd.seek(ib*(evt[i]-1))
            content=fd.read(ib)
            hybrid[i]=struct.unpack('B'*ib,content)
            #plt.plot(hybrid[cpt])
            #plt.show()
            
            maxinstdhyb[i]=np.max(abs(hybrid[i]-np.mean(hybrid[i])))/np.std(hybrid[i])
            
            '''
            maxinstd=[]
            fd.seek(0)
            for j in range(0,int(filesize/1024)):

            
                content=fd.read(ib)
                trace=np.asarray(struct.unpack('B'*ib,content))
            
                #trace=np.reshape(trace,(int(len(content)/1024),1024))
                #print(trace)
                maxinstd.append((np.max(abs(trace))-np.mean(trace))/np.std(trace))
		
	    
            plt.hist(maxinstd, bins=nbins, range=(0,18), histtype='step', linewidth=2)
            plt.vlines(maxinstdhyb[cpthyb],0,10000,color='r')
            plt.yscale('log')
            plt.show()
	    
            meanmaxinstd[i]=np.mean(maxinstd)
            stdmaxinstd[i]=np.std(maxinstd)
            print(meanmaxinstd,stdmaxinstd)

            
            with open('maxinstd.txt','a') as f:
                f.write(str(run[i])+' '+str(ant[i])+' '+str(maxinstdhyb[i])+' '+str(meanmaxinstd[i])+' '+str(stdmaxinstd[i])+'\n')'''
	    
	    
	    
            for j in range(0,unbal):
            
                randompos=int(random()*filesize/1024)
                #print(randompos)
                fd.seek(ib*(randompos))
                content=fd.read(ib)
                transient[cpttrans]=struct.unpack('B'*ib,content)

                maxinstdtrans=np.max(abs(transient[cpttrans]-np.mean(transient[cpttrans])))/np.std(transient[cpttrans])

                trial=0
                while (maxinstdtrans<(maxinstdhyb[i]-0.5*std[i]) or maxinstdtrans>(maxinstdhyb[i]+0.5*std[i])) and trial<5000:
                    trial=trial+1
                    randompos=int(random()*filesize/1024)
                    #print(randompos)
                    fd.seek(ib*(randompos))
                    content=fd.read(ib)
                    transient[cpttrans]=struct.unpack('B'*ib,content)

                    maxinstdtrans=np.max(abs(transient[cpttrans]-np.mean(transient[cpttrans])))/np.std(transient[cpttrans])
                    #print(maxinstd)


                if trial==5000:
                    print(trial)
                    trial=0
                    while (maxinstdtrans<(maxinstdhyb[i]-1*std[i]) or maxinstdtrans>(maxinstdhyb[i]+1*std[i])) and trial<5000:
                        trial=trial+1
                        randompos=int(random()*filesize/1024)
                        #print(randompos)
                        fd.seek(ib*(randompos))
                        content=fd.read(ib)
                        transient[cpttrans]=struct.unpack('B'*ib,content)

                        maxinstdtrans=np.max(abs(transient[cpttrans]-np.mean(transient[cpttrans])))/np.std(transient[cpttrans])
                        #print(maxinstd)


                if trial==5000:
                    print(trial)
                    trial=0
                    while maxinstdtrans<(maxinstdhyb[i]-3*std[i]) or maxinstdtrans>(maxinstdhyb[i]+3*std[i]) and trial<5000:
                        randompos=int(random()*filesize/1024)
                        #print(randompos)
                        fd.seek(ib*(randompos))
                        content=fd.read(ib)
                        transient[cpttrans]=struct.unpack('B'*ib,content)

                        maxinstdtrans=np.max(abs(transient[cpttrans]-np.mean(transient[cpttrans])))/np.std(transient[cpttrans])
                        #print(maxinstd)                        
                        
                        
                        
                        
                
                print(maxinstdhyb[i],maxinstdtrans)
                cpttrans=cpttrans+1
                

                
                
                
                
            
            
        cpthyb=cpthyb+1
        print(cpthyb)
        print(cpttrans)






'''plt.hist(meanmaxinstd, bins=10, histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Mean of Max [std unit]')
plt.show()
plt.hist(stdmaxinstd, bins=10, histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Std of Max [std unit]')
plt.show()
'''

        

with open('/sps/trend/slecoz/MLhybrid/MLhybrid_selected.bin','wb') as fd:
    for i in range(cpthyb):
        for k in range(ib):
            content=struct.pack('B',int(hybrid[i,k]))
            fd.write(content)
            
with open('/sps/trend/slecoz/MLhybrid/MLhybrid_transient.bin','wb') as fd:
    for i in range(cpttrans):
        for k in range(ib):
            content=struct.pack('B',int(transient[i,k]))
            fd.write(content)
                        
            
   
   
   
   
   
   
   
   
   
   
   
        


'''
binfolders  = listdir('/sps/trend/slecoz/bin_selected_hybrid/tracestotest/')


#filesize=os.path.getsize(signalfilename)

Ndata=len(binfolders)
signal=np.zeros((Ndata,ib))



for b in range(len(binfolders)):

    signalfilename= binfolders[b]

    with open('/sps/trend/slecoz/bin_selected_hybrid/tracestotest/'+signalfilename,'rb') as fd:
        content=fd.read(ib)
        signal[b]=struct.unpack('B'*ib,content)
        
        



with open('/sps/trend/slecoz/bin_selected_hybrid/tracestotest/allhybridtotest.bin','wb') as fd:
    for i in range(0,Ndata):
        plt.plot(signal[i])
        plt.show()
        for k in range(0,ib):
            content=struct.pack('B',int(signal[i,k]))
            fd.write(content)
'''
