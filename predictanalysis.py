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


from shared import *




def SetupCharacteristics_TREND50(nrun,dets):

    delay=dict()

    for i in dets:

        if i==101:
            delay[i]=2.31
            
        elif i==102:
            delay[i]=98.7
            
        elif i==103:
            delay[i]=281.7
            
        elif i==104:
            delay[i]=276.4
            
        # Warning!!!! Inverted with 106
        elif i==105:
            if nrun<=4690:
                delay[i]=266.9
            
            else:
                delay[i]=537.6
                            
        elif i==106:
            if nrun<=4690:
                delay[i]=537.6
            else:
                delay[i]=266.9
            
        elif i==107:
            delay[i]=547.0
            
        elif i==108:
            delay[i]=658.0
            
        elif i==109:
            delay[i]=584.2
            
        elif i==110:
            delay[i]=609.2
            
        elif i==111:
            delay[i]=647.0
            
        elif i==112:
            delay[i]=731.4
            
        elif i==113:
            delay[i]=728.3
            
        elif i==114:
            delay[i]=748.7
            
        elif i==115:
            delay[i]=785.7
            
        elif i==116:
            delay[i]=947.6
            
        elif i==117:
            delay[i]=977.1
            
        elif i==118:
            delay[i]=1206.5
            
        elif i==119:
            delay[i]=1320.0
            
        elif i==120:
            delay[i]=1358.2
            
        elif i==121:
            delay[i]=130.9
            
        elif i==122:
            delay[i]=182.2
            
        elif i==123:
            delay[i]=171.6
            
        elif i==124:
            delay[i]=470.7
            
        elif i==125:
            delay[i]=535.5
            
        elif i==126:
            delay[i]=612.1
            
        elif i==127:
            delay[i]=539.0
            
        elif i==128:
            delay[i]=673.7
            
        elif i==129:
            delay[i]=740.0
            
        elif i==130:
            delay[i]=759.2
            
        elif i==131:
            delay[i]=740.2
            
        elif i==132:
            delay[i]=746.4
            
        elif i==133:
            delay[i]=917.0
            
        elif i==134:
            delay[i]=874.3
            
        elif i==135:
            delay[i]=909.1
            
        elif i==136:
            delay[i]=1101.1
            
        elif i==137:
            delay[i]=1064.9
            
        elif i==138:
            delay[i]=1098.6
            
        elif i==140:
            delay[i]=1257.9
            
        # No cable measurment for NS antennas    
        elif i==148:  #Z channel  
            delay[i]=1893.2
                 
        elif i==149: #Y channel
            delay[i]=1841.6
            
        elif i==150:  #X channel
            delay[i]=1828.8
            
        elif i==151:
            delay[i]=1756.2
            
        elif i==152:
            delay[i]=1728.2
            
        elif i==153:
            delay[i]=1740.2
            
        elif i==154:
            delay[i]=1553.3
            
        elif i==155:
            delay[i]=1535.0
            
        elif i==156:
            delay[i]=1400
            
        elif i==157:
            delay[i]=1515.5
            
        elif i==158:     
            delay[i]=1563.4



    print(delay)
    return(delay)





ant,x,y,z=np.loadtxt('coord_antennas_all_TREND50.txt', unpack=True, dtype=[('f1','int'),('f2','float'),('f3','float'),('f4','float')])
print((ant))
print(z)

delay=SetupCharacteristics_TREND50(3562,ant)
print(delay[101])

#############  write event time table for p(signal) above threshold


run='R003564'
threshold=90


files=listdir(SPS_PATH+'log_classpredict')
scorelist=[]
total=0
totaltotal=0


with open(SPS_PATH+'eventtimetable_'+str(threshold)+'_'+run+'.txt','w') as f:
    f.write('')


for i in range(len(files)):


    ID=files[i][25:28]
    #print(ID)
    scores=np.loadtxt(SPS_PATH+'log_classpredict/'+files[i])
    total=total+len(scores[scores>threshold])
    totaltotal=totaltotal+len(scores)
    
    print(files[i], len(scores[scores>threshold]), len(scores[scores>threshold])/len(scores))
    print(total, totaltotal)
    
    ind=np.arange(len(scores))
    indscores=ind[scores>threshold]
    times=np.zeros((len(indscores)))
    

    antdelay=delay[int(ID)]


    timefilename=SPS_PATH+run+'/'+run+'_A0'+ID+'_time.bin'
    with open(timefilename,'rb') as ft:
    
        for j in range(len(indscores)):
            
            ft.seek(indscores[j]*16)

            content=ft.read(4)
            content=ft.read(4)
            if len(content)==4:
                t1event=(struct.unpack('I'*1,content)[0]-1)*2**28
            content=ft.read(4)
            if len(content)==4:
                t2event=(struct.unpack('I'*1,content)[0]-1)*1024
            content=ft.read(4)
            if len(content)==4:
                t3event=struct.unpack('I'*1,content)[0]

            times[j]=t1event+t2event+t3event-antdelay

     
    with open(SPS_PATH+'eventtimetable_'+str(threshold)+'_'+run+'.txt','a') as f:
        for j in range(len(indscores)):
            f.write(ID+' '+str(times[j])+'\n')


#############  write coinc time table for signals in coinc

nantmin=5

timetable=np.loadtxt(SPS_PATH+'eventtimetable_'+str(threshold)+'_'+run+'.txt', dtype=[('f1','int'),('f2','float')])

timetable=np.sort(timetable,order='f2')
timetable=np.asarray(timetable.tolist())
print(timetable)

coincidences=np.zeros((len(timetable),3))

ref=0
cpt=0
cpt_coinc=0

while ref<len(timetable):

    print(ref,cpt)

    refid=timetable[ref,0]
    refpos=np.asarray([x[ant==refid],y[ant==refid],z[ant==refid]])
    reftime=timetable[ref,1]
    
    coincidences[cpt,0]=cpt_coinc+1
    coincidences[cpt,1]=refid
    coincidences[cpt,2]=reftime
    
    print(coincidences[cpt])

    nant=1
    test=ref+1
    antincoinc=[]

    while test<ref+2*len(ant) and test<len(timetable):
        
        testid=timetable[test,0]
        testpos=np.asarray([x[ant==testid],y[ant==testid],z[ant==testid]])
        testtime=timetable[test,1]
        distance=np.linalg.norm(testpos-refpos) # meters
        distance=distance/c0/TSAMPLING #5ns bins
		
        if testtime-reftime<distance*1.5:
            print("oui")
            print(testtime-reftime,distance)
	    
            if len(np.asarray(antincoinc)[np.asarray(antincoinc)==testid])==0:
                cpt=cpt+1
                nant=nant+1
                antincoinc.append(testid)
                coincidences[cpt,0]=cpt_coinc+1
                coincidences[cpt,1]=testid
                coincidences[cpt,2]=testtime
		
                print(coincidences[cpt])
	    
        test=test+1
    
    if nant<nantmin:
        cpt=cpt-nant
    else:
        cpt_coinc=cpt_coinc+1


    ref=ref+nant
    cpt=cpt+1


with open(SPS_PATH+'coinctimetable_'+str(threshold)+'_'+run+'.txt','w') as f:
    for j in range(cpt):
       f.write(str(int(coincidences[j,0]))+' '+str(int(coincidences[j,1]))+' '+str(coincidences[j,2])+'\n')





