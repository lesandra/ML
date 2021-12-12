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
#import cPickle as pk


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

def AveragedGain():

    antennas=listdir(GAIN_DATA_PATH)
    antennas=np.asarray(antennas)
    antennas=antennas[antennas!='run-times.py']
    print(antennas)
    #gain=np.zeros(len(antennas))
    ant=np.zeros(len(antennas),dtype=int)
    for a in range(len(antennas)):
        ant[a]=antennas[a][5:8]
        '''avgain=[]
        with open(GAIN_DATA_PATH+antennas[a], "rb") as f:
            data = pk.load(f)
        for i in range(len(data)):        
            avgain.append(np.mean(data[i][-1]))
        gain[a]=np.mean(avgain)

    with open(SPS_PATH+'averagedgain.txt','w') as f:
        for a in range(len(antennas)):
            f.write(str(ant[a])+' '+str(gain[a])+'\n')'''
 
    return ant
        






#############################
#############################



def Treatment(txtfile,noise):


    onlysimu=np.zeros((ib)) #simu only for dataset (same simu as in simunoise)
    simunoise=np.zeros((ib)) #simu and noise for dataset

    display=0


    with open (txtfile, 'r') as file :
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


    if display:
        plt.plot(vs,'g')
        plt.show()

    tstep=np.mean(np.diff(t))
    #print(len(vs),int(round(TSAMPLING/tstep))*ib)
    vs2=np.zeros(int(round(TSAMPLING/tstep))*ib)
    vs2[int(len(vs2)/2):int(len(vs2)/2)+len(vs)]=vs

    if display:
        plt.plot(vs2)
        plt.show()

    vs=vs2



    #signal filtration 
    F=rfftfreq(len(vs))/tstep #len(vs) points between 0 and 1/2tstep=0.5e10Hz
    #print(F)
    VS=rfft(vs)
    VS[F<FREQMIN]=0
    VS[F>FREQMAX]=0

    vs=irfft(VS)
    #print('Vs length=',len(vs), len(ex), vs[0:200], vs[len(vs)-100:len(vs)])
    if display:
        plt.plot(VS)
        plt.plot(vs2)
        plt.plot(vs,'r')
        plt.show()


    #digitization
    #no meaning of vs after the time of efield (see fft tests on matlab)
    ratio=int(round(TSAMPLING/tstep))
    #print(TSAMPLING,tstep,ratio,len(vs))
    ind=np.arange(0,int(np.floor(len(vs)/ratio)))*ratio

    onlysimu=vs[ind]



    if display:
        plt.plot(onlysimu,'r')
        plt.show()






    #gain
    #print(unixtrand,ID)
    #gain=FindGain(unixtrand,ID)
    #gain=108.5
    gain=110
    gain=10**(gain/20) #linear scale
    #print('GAIN!!!!!!!!!!!!!',gain)
    onlysimu=onlysimu*1e-6 #pour mettre en volt
    onlysimu=onlysimu*gain #pour mettre en lsb
    #vf[i]=vf[i]*gain*256/3.3 
    #translation of voltage to lsb
    #vf[i]=vf[i]*1e-6/SCALE
    #SCALE deja pris en compte dans gain valentin


    if display:
        plt.plot(onlysimu)
        plt.show()



    maxi=max(abs(onlysimu))
    ind=np.nonzero(abs(onlysimu)==maxi)[0]
    #print('here!!',maxi,ind,ind[0])




    #center the maximum on the trigger time (512)
    '''vtemp=np.zeros(ib)
    trigtime=512
    if ind[0]<trigtime:
        vtemp[trigtime-ind[0]:trigtime]=simunoise[0:ind[0]]
        vtemp[trigtime:ib]=simunoise[ind[0]:ind[0]+trigtime]
        for k in range(0,trigtime-ind[0]):                      
            #vtemp[k]=noisecontent[random()*399]
            vtemp[k]=int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    elif ind[0]>trigtime:
        vtemp[0:trigtime]=simunoise[ind[0]-trigtime:ind[0]]
        vtemp[trigtime:trigtime+ib-ind[0]]=simunoise[ind[0]:ib]
        for k in range(trigtime+ib-ind[0],ib):                      
            #vtemp[k]=noisecontent[random()*399]
            vtemp[k]=int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    elif ind[0]==trigtime:
        vtemp=simunoise
    simunoise=vtemp'''

    vtemp=np.zeros(ib)
    trigtime=512
    if ind[0]<trigtime:
        vtemp[trigtime-ind[0]:trigtime]=onlysimu[0:ind[0]]
        vtemp[trigtime:ib]=onlysimu[ind[0]:ind[0]+trigtime]
        #for k in range(0,trigtime-ind[0]):                      
            #vtemp[k]=noisecontent[random()*399]
            #vtemp[k]=np.mean(noisecontent)
    elif ind[0]>trigtime:
        vtemp[0:trigtime]=onlysimu[ind[0]-trigtime:ind[0]]
        vtemp[trigtime:trigtime+ib-ind[0]]=onlysimu[ind[0]:ib]
        #for k in range(trigtime+ib-ind[0],ib):                      
            #vtemp[k]=noisecontent[random()*399]
            #vtemp[k]=np.mean(noisecontent)
    elif ind[0]==trigtime:
        vtemp=onlysimu
    onlysimu=vtemp




    #addnoise
    #noisecontent=noise[0:400]
    simunoise=onlysimu+noise


        #for i in range(0, ib):
        #simunoise[cpt][i]=onlysimu[cpt][i]+noisecontent[random()*399]
        #simunoise[i]=onlysimu[i]+int(round(gauss(np.mean(noisecontent),np.std(noisecontent))))
    onlysimu=onlysimu+np.mean(noise)


    if display:
        plt.plot(noise)
        plt.plot(simunoise)
        plt.plot(onlysimu)
        plt.show()



    #if saturation or null values
    for k in range(0,ib):
        if onlysimu[k]>255:
            onlysimu[k]=255
        if onlysimu[k]<0:
            onlysimu[k]=0
    for k in range(0,ib):
        if simunoise[k]>255:
            simunoise[k]=255
        if simunoise[k]<0:
            simunoise[k]=0



    maxi=max(abs(simunoise))
    ind=np.nonzero(abs(simunoise)==maxi)[0]
    #print('here!!',maxi,ind,ind[0])



    overthreshold=1
    #print(maxi,np.mean(simunoise)+6*np.std(simunoise))
    if maxi<np.mean(simunoise)+6*np.std(simunoise):
        overthreshold=0
        return overthreshold, onlysimu, simunoise








    if 0:
        plt.plot(simunoise,'g')
        plt.plot(onlysimu)
        plt.show()
            
    return overthreshold, onlysimu, simunoise
            
            
#############################
#############################         Main   


'''with open('/sps/trend/slecoz/BACK/R003199_A0131_BACK_data.bin','rb') as fd:
    while 1:
    
        content=fd.read(1024)
        plt.plot(struct.unpack('B'*1024,content))
        plt.show()'''




ant=AveragedGain()


binfolders  = listdir(P6_DATA_PATH)
#energies = listdir(SIMU_DATA_PATH)
#energies= ['2e17','3e17','5e17','7e17','1e18','2e18','3e18']
energies= ['1e17','2e17','3e17','5e17','7e17','1e18']
#esize=[1738, 1995, 1407, 1406, 1476, 2186, 2549]
esize=[5471, 1738, 1995, 1407, 1406, 1476, 2186]

transient=np.zeros((Ndata,ib)) #transient for dataset
onlysimu=np.zeros((Ndata,ib)) #simu only for dataset (same simu as in simunoise)
noise=np.zeros((Ndata,ib))  #noise for simu
simunoise=np.zeros((Ndata,ib)) #simu and noise for dataset

logsim=[]


tunix=[]
#gain=[]
#gain=np.zeros((Ndata))+108.5





############ open raw files



cpt_sim=0
cpt_trans=0
cpt_noise=0
necnum=np.zeros((len(ant),len(energies)),dtype=int)
print(ant)

for b in range(len(binfolders)):
#for b in range():   

    datafilename=P6_DATA_PATH+binfolders[b]
    filesize=os.path.getsize(datafilename)
    #datafilename=RAW_DATA_PATH+run+"_A0"+ID+"_data.bin"
    #timefilename=RAW_DATA_PATH+run+"_A0"+ID+"_time.bin"


    #if we look for gain at accurate unix time
    '''with open(timefilename,'rb') as ft:
        content=ft.read()                        
    i=0
    while i<Ndata*16:
        tunix.append(struct.unpack('I'*1,content[i:i+4])[0])
        gain.append(FindGain(tunix[-1],ID))
        i=i+16'''

    ID=binfolders[b][10:13]
    run=binfolders[b][0:7]
    print(ID,run)
    
    
    backfilename=run+'_A0'+ID+'_BACK_data.bin'
    print(backfilename)
    if os.path.isfile(BACK_PATH+backfilename)==0:
        pdir=subprocess.Popen('iget '+RAW_DATA_PATH+run+'/'+backfilename+' '+BACK_PATH, shell=True) 
        pdir.wait()
    if os.path.isfile(BACK_PATH+backfilename)==0:
        backfilename='R003562_A0'+ID+'_BACK_data.bin'
    print(backfilename)
	
    backfilename=BACK_PATH+backfilename
    filesizeback=os.path.getsize(backfilename)
    
    ind=np.nonzero(ant==int(ID))[0][0]
    #print(ind)
    print(ant[ind],ind,necnum[ind][0])
    
    with open(datafilename,'rb') as fd:
        
        for energy in energies:
            
            randompos=int(random()*filesize/1024)
            print(randompos)
            fd.seek(1024*(randompos))
            content=fd.read(1024)
            transient[cpt_trans]=struct.unpack('B'*1024,content)
	    
            while np.std(transient[cpt_trans][0:400])>10:
                randompos=int(random()*filesize/1024)
                fd.seek(1024*(randompos))
                content=fd.read(1024)
                transient[cpt_trans]=struct.unpack('B'*1024,content)
            cpt_trans=cpt_trans+1
		
		
    with open(backfilename,'rb') as fd:
        
        for energy in energies:
            	
            randompos=int(random()*filesizeback/1024)
            fd.seek(1024*(randompos))
            content=fd.read(1024)
            noise[cpt_noise]=struct.unpack('B'*1024,content)

            while np.std(noise[cpt_noise][0:400])>10:
                randompos=int(random()*filesizeback/1024)
                fd.seek(1024*(randompos))
                content=fd.read(1024)
                noise[cpt_noise]=struct.unpack('B'*1024,content)


            cpt_noise=cpt_noise+1
	    

    for e in range(len(energies)):
    

        energy=energies[e]
        print(energy)
        necfolders = listdir(SIMU_DATA_PATH+energy+'/voltages/')
	
        if necnum[ind][e]>=esize[e]:
            continue

        
        #print(necfolders[necnum[ind][e]],necnum[ind][e])        
        necfolder=SIMU_DATA_PATH+energy+'/voltages/'+necfolders[necnum[ind][e]]
        txtfile=necfolder+'/a'+str(ID)+'_ew.txt'
        #print(txtfile)
        while os.path.isfile(txtfile)==0 and necnum[ind][e]<esize[e]-1:
            necnum[ind][e]=necnum[ind][e]+1
            #print(necfolders[necnum[ind][e]],necnum[ind][e])        
            necfolder=SIMU_DATA_PATH+energy+'/voltages/'+necfolders[necnum[ind][e]]
            txtfile=necfolder+'/a'+str(ID)+'_ew.txt'
            #print(txtfile)
        if os.path.isfile(txtfile)!=0:
            overthreshold,onlysimu[cpt_sim],simunoise[cpt_sim]=Treatment(txtfile,noise[cpt_sim])
        
            while overthreshold==0 and necnum[ind][e]<esize[e]-1:
                necnum[ind][e]=necnum[ind][e]+1
                #print(necfolders[necnum[ind][e]],necnum[ind][e])        
                necfolder=SIMU_DATA_PATH+energy+'/voltages/'+necfolders[necnum[ind][e]]
                txtfile=necfolder+'/a'+str(ID)+'_ew.txt'
                #print(txtfile)
                while os.path.isfile(txtfile)==0 and necnum[ind][e]<esize[e]-1:
                    necnum[ind][e]=necnum[ind][e]+1
                    #print(necfolders[necnum[ind][e]],necnum[ind][e])        
                    necfolder=SIMU_DATA_PATH+energy+'/voltages/'+necfolders[necnum[ind][e]]
                    txtfile=necfolder+'/a'+str(ID)+'_ew.txt'
                    #print(txtfile)
                if os.path.isfile(txtfile)!=0:
                    overthreshold,onlysimu[cpt_sim],simunoise[cpt_sim]=Treatment(txtfile,noise[cpt_sim])
            

        necnum[ind][e]=necnum[ind][e]+1
	
        if overthreshold!=0:
	
            logsim.append(txtfile)          
            cpt_sim=cpt_sim+1
	    
    cpt_noise=cpt_sim
	    


print(cpt_sim,cpt_noise,cpt_trans,necnum)

############ save to file



if trace400:
    simuMLfilename=MLP6SIM400_DATA_PATH+'MLP6SIM400_simu.bin'
    transientMLfilename=MLP6SIM400_DATA_PATH+'MLP6SIM400_transient.bin'
    simunoiseMLfilename=MLP6SIM400_DATA_PATH+'MLP6SIM400_simunoise.bin'
else:
    simuMLfilename=MLP6SIM_DATA_PATH+'MLP6SIM_simu.bin'
    transientMLfilename=MLP6SIM_DATA_PATH+'MLP6SIM_transient.bin'
    simunoiseMLfilename=MLP6SIM_DATA_PATH+'MLP6SIM_simunoise.bin'

#data file implementation
if os.path.isdir(MLP6SIM400_DATA_PATH)==0:
    pdir=subprocess.Popen('mkdir '+MLP6SIM400_DATA_PATH, shell=True)
    pdir.wait()




with open(simuMLfilename,'wb') as fd:
    for i in range(0,cpt_sim):
        for k in range(0,ib):
            content=struct.pack('B',int(onlysimu[i,k]))
            fd.write(content)
           


with open(transientMLfilename,'wb') as fd:
    for i in range(0,cpt_sim):
        for k in range(0,ib):
            content=struct.pack('B',int(transient[i,k]))
            fd.write(content)
            
            

with open(simunoiseMLfilename,'wb') as fd:
    for i in range(0,cpt_sim):
        for k in range(0,ib):
            content=struct.pack('B',int(simunoise[i,k]))
            fd.write(content)          
	      
            
with open('logsim.txt','w') as f:
    for i in range(len(logsim)):
        f.write(logsim[i]+'\n')
 

        
############ check files content
        


with open(simuMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    #plt.plot(content)
    #plt.show()
        
with open(simunoiseMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    #plt.plot(content)
    #plt.show()

with open(transientMLfilename,'rb') as fd:
    content=fd.read()
    size=int(len(content)) #8bits data = 1 bytes data
    print(len(content))
    content=struct.unpack('B'*len(content),content) #https://docs.python.org/2/library/struct.html
    print(len(content))    
    #size=int(len(content))
    #plt.plot(content)
    #plt.show()
    
    















