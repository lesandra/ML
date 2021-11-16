import numpy as np
import subprocess #to launch linux commands
import os.path
from os import listdir
from os.path import isfile, join
import struct #Interpret strings as packed binary data
from random import *
import os
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, rfftfreq
import sys
import cPickle as pk


from shared import *

#works with python2
#############################
#############################

#VN
def FindGain(time_unix,antenna):

    
    # Load the data.
    with open(os.path.join(GAIN_DATA_PATH, "gain.{:}.p".format(int(antenna))), "rb") as f:
        data = pk.load(f)

    # Check the outer bounds.
    if (time_unix < data[0][0]) or (time_unix > data[-1][1]):
        return None

    # Search the run for the given unix time, using a dichotomy.
    def search_index(i0, i1):
        if i1-i0 <= 1:
            if time_unix < data[i1][0]:
                return i0
            else:
                return i1
        i2 = (i0+i1)/2
        t2 = data[i2][0]
        if time_unix >= t2:
            return search_index(i2, i1)
        else:
            return search_index(i0, i2)
            
    irun = search_index(0, len(data)-1)
    t0, t1, dt, run, t, g = data[irun]
    
    # Check if the unix time is within the run.
    if time_unix > t1:
        # No data for this time.
        return None
    elif time_unix <= t[0]:
        # Time corresponds to the 1st sample.
        return g[0]
    elif time_unix >= t[-1]:
        # Time corresponds to the last sample.
        return g[-1]
    
    # Interpolate the gain.
    def search_index(i0, i1):
        if i1-i0 <= 1:
            if time_unix < t[i1]:
                return i0
            else:
                return i1
        i2 = (i0+i1)/2
        t2 = t[i2]
        if time_unix >= t2:
            return search_index(i2, i1)
        else:
            return search_index(i0, i2)
    i0 = search_index(0, len(t)-1)
    t0, t1 = t[i0], t[i0+1]
    h = time_unix-t0
    if h > 1.1*dt:
        print(time_unix,t0,t1,dt)
        return None
        #return 60 #temporaire!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    g0, g1 = g[i0], g[i0+1]
    h /= t1-t0


    #gain=0.5e5
    return(g0*(1.-h)+g1*h) #dB


#############################
#############################


necfolders = listdir(SIMU_DATA_PATH)
#print(necfolders)
cpt=0


transient=np.zeros((Ndata*2,ib)) #transient for dataset
onlysimu=np.zeros((Ndata,ib)) #simu only for dataset (same simu as in simunoise)
noise=np.zeros((Ndata,ib))  #noise for simu
simunoise=np.zeros((Ndata,ib)) #simu and noise for dataset


tf=np.zeros((Ndata,ib))
tunix=[]
#gain=[]
#gain=np.zeros((Ndata))+108.5





############ open raw files


datafilename=RAW_DATA_PATH+run+"_A0"+ID+"_data.bin"
timefilename=RAW_DATA_PATH+run+"_A0"+ID+"_time.bin"


#if we look for gain at accurate unix time
'''with open(timefilename,'rb') as ft:
    content=ft.read()                        
i=0
while i<Ndata*16:
    tunix.append(struct.unpack('I'*1,content[i:i+4])[0])
    gain.append(FindGain(tunix[-1],ID))
    i=i+16'''
    
i=0
with open(datafilename,'rb') as fd:
    while i<Ndata:
        content=fd.read(ib)
        noise[i]=struct.unpack('B'*ib,content)
        print(np.mean(noise[i]), np.std(noise[i][0:400]))
	if np.std(noise[i][0:400])<10:
      	    i=i+1

    #we want high and low std noise:
    i=0
    while i<Ndata:
        content=fd.read(ib)
        transient[i]=struct.unpack('B'*ib,content)
        print(np.mean(transient[i]), np.std(transient[i]))
	if np.std(transient[i])<10:
      	    i=i+1
    while i<Ndata*2:
        content=fd.read(ib)
        transient[i]=struct.unpack('B'*ib,content)
        print(np.mean(transient[i]), np.std(transient[i]))
	if np.std(transient[i])>10:
      	    i=i+1	    	    




'''
noisetest=noise[0][0:400]
noisetestrand=np.zeros(400)
for i in range(0, len(noisetestrand)):
    noisetestrand[i]=noisetest[random()*399]
    
plt.plot(noisetest)
plt.plot(noisetestrand)
plt.show()


h,b=np.histogram(noisetest, bins=10, range=None, normed=None, weights=None, density=None)
hrand,brand=np.histogram(noisetestrand, bins=10, range=None, normed=None, weights=None, density=None)

plt.plot(b[:-1],h)
plt.plot(brand[:-1],hrand)
plt.show()


for i in range(1,11):
    noisetestrandshift=np.zeros(400)
    noisetestrandshift=noisetestrand[i:]-noisetestrand[0:-i]

    plt.plot(noisetestrandshift)
    plt.show()

    h,b=np.histogram(noisetestrandshift, bins=10, range=None, normed=None, weights=None, density=None)

    plt.plot(b[:-1],h)
    plt.show()
'''





for necfolder in necfolders:




    if cpt>=Ndata:
        break
    txtfile=SIMU_DATA_PATH+necfolder+'/a'+str(ID)+'_efield.txt'
    txtfile2=SIMU_DATA_PATH+necfolder+'/a'+str(ID)+'_ew.txt'
    print(txtfile)
    print(txtfile2)
    if os.path.isfile(txtfile)==0 or os.path.isfile(txtfile2)==0:
    	continue
    
    display=0
    if necfolder=="15392595051_66_198_-104_227":
        display=1
	
    et=[]
    ex=[]
    ey=[]
    ez=[]
    with open (txtfile, 'r') as file :
        txt = file.read()
        txt=txt.split('\n')
        
    for l in range(0,len(txt)-1):
        line=txt[l].split(' ')
        #print(line)
        linegoods=[]
        for k,elt in enumerate(line):
            if elt!='':
                linegoods.append(elt)
        #print(linegoods)
        et.append(float(linegoods[0])*1e-6) #to convert museconds to seconds
        ex.append(float(linegoods[1])) #muV/m
        ey.append(float(linegoods[2]))
        ez.append(float(linegoods[3]))

    with open (txtfile2, 'r') as file :
        txt = file.read()
        txt=txt.split('\n')
        t=txt[0].split(' ') #already in seconds
        vs=txt[1].split(' ') #sum of the 3 following v
        vr=txt[2].split(' ')
        vt=txt[3].split(' ')
        vp=txt[4].split(' ')
        for k in range(0,len(t)):
            t[k]=float(t[k])
            vs[k]=float(vs[k]) #muV
            vr[k]=float(vr[k])
            vt[k]=float(vt[k])
            vp[k]=float(vp[k])
            
    #plt.plot(ex,'red')         
    #print((t[1]-t[0]),(et[1]-et[0]))   # 1e-9sec
    if display:
        plt.plot(vs,'g')
        plt.show()
    

        #print(et[0:100])
    tstep=np.mean(np.diff(et))
    #print('tsep=',tstep)
    # to make et and vs the same size
    for k in range(len(ex),len(vs)):
        et.append(et[k-1]+tstep)
	
	
    print(len(vs),int(round(TSAMPLING/tstep))*ib)
    vs2=np.zeros(int(round(TSAMPLING/tstep))*ib)
    vs2[int(len(vs2)/2):int(len(vs2)/2)+len(vs)]=vs
    
    if display:
        plt.plot(vs2)
        plt.show()

    vs=vs2
	

    '''#print(et[len(ex)-1:len(ex)+100])            
    #print('et step',np.mean(np.diff(et)))
    tstep=np.mean(np.diff(t)) #same for et and t
    #print('tsep=',tstep)
    vs=np.asarray(vs)
    et=np.asarray(et)
    shift=500
    vsshift=np.zeros(len(vs))
    etshift=np.zeros(len(et))
    vsshift[shift:len(vsshift)]=vs[0:len(vs)-shift] 
    etshift[shift:len(etshift)]=et[0:len(et)-shift]
    for k in range(0,shift):
        etshift[k]=et[0]-(shift-k)*tstep
    vs=vsshift
    et=etshift
    #print(et[shift-100:shift+1])
    if display:
        plt.plot(vs)
        plt.show()'''

    #signal filtration 
    F=rfftfreq(len(vs))/tstep #len(vs) points between 0 and 1/2tstep=0.5e10Hz
    #print(F)
    VS=rfft(vs)
    VS[F<FREQMIN]=0
    VS[F>FREQMAX]=0

    vs=irfft(VS)
    #print('Vs length=',len(vs), len(ex), vs[0:200], vs[len(vs)-100:len(vs)])
    if display:
        #plt.plot(VS)
        plt.plot(vs2)
	plt.plot(vs,'r')
        plt.show()



    


    #digitization
    #no meaning of vs after the time of efield (see fft tests on matlab)
    ratio=int(round(TSAMPLING/tstep))
    print(TSAMPLING,tstep,ratio,len(vs))
    ind=np.arange(0,int(np.floor(len(vs)/ratio)))*ratio
    '''if len(ind)>524:
	ind=ind[0:524]
    onlysimu[cpt,500:len(ind)+500]=vs[ind]'''
    onlysimu[cpt]=vs[ind]
    
    '''tf[cpt,500:len(ind)+500]=et[ind]
    for k in range(0,500):
	tf[cpt,k]=tf[cpt,500]-(500-k)*TSAMPLING
    for k in range(len(ind)+500,ib):
	tf[cpt,k]=tf[cpt,k-1]+TSAMPLING

    if display:
        plt.plot(tf[cpt],onlysimu[cpt])
        plt.show()'''

    if display:
        plt.plot(onlysimu[cpt],'r')
        plt.show()






    #gain
    #print(unixtrand,ID)
    #gain=FindGain(unixtrand,ID)
    gain=108.5
    gain=10**(gain/20) #linear scale
    print('GAIN!!!!!!!!!!!!!',gain)
    onlysimu[cpt]=onlysimu[cpt]*1e-6 #pour mettre en volt
    onlysimu[cpt]=onlysimu[cpt]*gain #pour mettre en lsb
    #vf[i]=vf[i]*gain*256/3.3 
    #translation of voltage to lsb
    #vf[i]=vf[i]*1e-6/SCALE
    #SCALE deja pris en compte dans gain valentin
    
    
    if display:
        plt.plot(onlysimu[cpt])
        plt.show()



    #addnoise
    noisecontent=noise[cpt][0:400]
    for i in range(0, ib):
        #simunoise[cpt][i]=onlysimu[cpt][i]+noisecontent[random()*399]
	simunoise[cpt][i]=onlysimu[cpt][i]+int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    onlysimu[cpt]=onlysimu[cpt]+np.mean(noisecontent)
   
   
    if display:
        plt.plot(noise[cpt])
        plt.plot(simunoise[cpt])
	plt.plot(onlysimu[cpt])
        plt.show()



    #if saturation or null values
    for k in range(0,ib):
        if onlysimu[cpt,k]>255:
            onlysimu[cpt,k]=255
        if onlysimu[cpt,k]<0:
            onlysimu[cpt,k]=0
    for k in range(0,ib):
        if simunoise[cpt,k]>255:
            simunoise[cpt,k]=255
        if simunoise[cpt,k]<0:
            simunoise[cpt,k]=0



    #maxvf=max(abs(vf[i,412:612]-meannoise)) #look for the max in a small window
    maxi=max(abs(simunoise[cpt]))
    #ind=np.nonzero(abs(vf[i,412:612]-meannoise)==maxvf)[0]
    ind=np.nonzero(abs(simunoise[cpt])==maxi)[0]
    print('here!!',maxi,ind,ind[0],tf[cpt,ind[0]])
    #print(len(vf[i]),len(tf[i]))
    #tefield[cpt]=tf[cpt,ind[0]]*1e9/5 #in 5 nanoseconds bins

    #print(np.nonzero(tf[i]==0)[0])



    #center the maximum on the trigger time (512)
    vtemp=np.zeros(ib)
    trigtime=512
    if ind[0]<trigtime:
	vtemp[trigtime-ind[0]:trigtime]=simunoise[cpt,0:ind[0]]
	vtemp[trigtime:ib]=simunoise[cpt,ind[0]:ind[0]+trigtime]
	for k in range(0,trigtime-ind[0]):                      
            #vtemp[k]=noisecontent[random()*399]
	    vtemp[k]=int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    elif ind[0]>trigtime:
	vtemp[0:trigtime]=simunoise[cpt,ind[0]-trigtime:ind[0]]
	vtemp[trigtime:trigtime+ib-ind[0]]=simunoise[cpt,ind[0]:ib]
	for k in range(trigtime+ib-ind[0],ib):                      
            #vtemp[k]=noisecontent[random()*399]
	    vtemp[k]=int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    elif ind[0]==trigtime:
	vtemp=simunoise[cpt]
    simunoise[cpt]=vtemp
    
    vtemp=np.zeros(ib)
    trigtime=512
    if ind[0]<trigtime:
	vtemp[trigtime-ind[0]:trigtime]=onlysimu[cpt,0:ind[0]]
	vtemp[trigtime:ib]=onlysimu[cpt,ind[0]:ind[0]+trigtime]
	for k in range(0,trigtime-ind[0]):                      
            #vtemp[k]=noisecontent[random()*399]
	    vtemp[k]=np.mean(noisecontent)
    elif ind[0]>trigtime:
	vtemp[0:trigtime]=onlysimu[cpt,ind[0]-trigtime:ind[0]]
	vtemp[trigtime:trigtime+ib-ind[0]]=onlysimu[cpt,ind[0]:ib]
	for k in range(trigtime+ib-ind[0],ib):                      
            #vtemp[k]=noisecontent[random()*399]
	    vtemp[k]=np.mean(noisecontent)
    elif ind[0]==trigtime:
	vtemp=onlysimu[cpt]
    onlysimu[cpt]=vtemp
    
    
    
    

    if 0:
        plt.plot(simunoise[cpt],'g')
        #plt.plot(onlysimu[cpt])
        plt.show()
        
	
    cpt=cpt+1

print('cpt',cpt)
#data file implementation
if os.path.isdir(ML_DATA_PATH)==0:
    pdir=subprocess.Popen('mkdir '+ML_DATA_PATH, shell=True)
    pdir.wait()



############ save to file


simuMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_simu.bin"
with open(simuMLfilename,'wb') as fd:
    for i in range(0,cpt):
        for k in range(0,ib):
            content=struct.pack('B',int(onlysimu[i,k]))
            fd.write(content)
	   

transientMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_transient.bin"
with open(transientMLfilename,'wb') as fd:
    for i in range(0,Ndata*2):
        for k in range(0,ib):
            content=struct.pack('B',int(transient[i,k]))
            fd.write(content)
	    
	    
dataMLfilename=ML_DATA_PATH+'ML'+run+"_A0"+ID+"_data.bin"
with open(dataMLfilename,'wb') as fd:
    for i in range(0,cpt):
        for k in range(0,ib):
            content=struct.pack('B',int(simunoise[i,k]))
            fd.write(content)	    
 

	
############ check files content
	


with open(simuMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    plt.plot(content)
    plt.show()
	
with open(dataMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    plt.plot(content)
    plt.show()

with open(transientMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    plt.plot(content)
    plt.show()
    
    















