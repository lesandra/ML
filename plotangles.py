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

import datetime

from shared import *


#TREND candidates
content=np.loadtxt(SPS_PATH+'runcoincantevtangle_space.txt')
runs=content[:,0]
coincs=content[:,1]
ants=content[:,2]
antu=np.unique(ants)
evtids=content[:,3]
thetacand=content[:,4]
phicand=content[:,5]


def Recons(threshold):
    
    theta=[]    
    phi=[]
    chisq=[]
    x=[]
    y=[]
    z=[]    
    times=[]
    reject=[]
    reject2=[]
    rejectphi=[]
    rejectphitime=[]    
    diff=[]
    tunix=[]  
    trendcoinc=[]
    multbef=[]
    thetabef=[]
    phibef=[]
       
       
    reprecsph=SPS_PATH+'MLrecons/sphrecons/'
    repcoinctable=SPS_PATH+'MLtimetables/'
    #tab=[0,1,2,3,4,5,6,7,8,9]
    tab=range(len(listdir(reprecsph+threshold+'/')))
    txtfilesph  = listdir(reprecsph+threshold+'/')
    txtfilesph.sort()
    
    reprecplane=SPS_PATH+'MLrecons/planerecons/'
    txtfileplane  = listdir(reprecplane+threshold+'/')
    txtfileplane.sort()
    
    for f in tab:
    
        filesize=os.path.getsize(reprecsph+threshold+'/'+txtfilesph[f])
        print(txtfilesph[f],filesize) 
        if filesize==0:
            continue
            
        content=np.loadtxt(reprecsph+threshold+'/'+txtfilesph[f])
        print("recons loaded")
        print(len(np.shape(content)))
        if len(np.shape(content))==1:            
            x_1=np.array([content[3]])
            y_1=np.array([content[4]])
            z_1=np.array([content[5]])
            x=np.concatenate((x,x_1))
            y=np.concatenate((y,y_1))
            z=np.concatenate((z,z_1))
    
        else:
            x_1=content[:,3]
            y_1=content[:,4]
            z_1=content[:,5]
            x=np.concatenate((x,x_1))
            y=np.concatenate((y,y_1))
            z=np.concatenate((z,z_1))    
    
        radius_1=np.sqrt(x_1*x_1+y_1*y_1+z_1*z_1)
    

        typ='plane'
        if typ=='plane':
            run=txtfileplane[f][1:5]
            content=np.loadtxt(repcoinctable+'coinctimetable_'+threshold+'_R00'+run+'.txt')
            print("coinctimetable loaded")
            #get trendcand  
            trendcoinc_1=np.zeros(int(content[-1][0]))
            multbef_1=np.zeros(int(content[-1][0]))        
            thetabef_1=np.zeros(int(content[-1][0]))-1
            phibef_1=np.zeros(int(content[-1][0]))-1
            for i in range(len(antu)):
                trendevt=evtids[(runs==float(run)) & (ants==antu[i])]
                predevt=content[:,2][content[:,1]==antu[i]]
                inter=np.intersect1d(trendevt, predevt)
                #if len(inter)>0:
                 #   print(antu[i],trendevt,predevt,inter)
                for j in range(len(inter)):
                    coincid=content[:,0][(content[:,1]==antu[i]) & (content[:,2]==inter[j])]
                    trendcoinc_1[int(coincid)-1]=trendcoinc_1[int(coincid)-1]+1
                    coincbef=coincs[(runs==float(run)) & (ants==antu[i]) & (evtids==inter[j])]
                    thetabef_1[int(coincid)-1]=thetacand[(runs==float(run)) & (ants==antu[i]) & (evtids==inter[j])]
                    phibef_1[int(coincid)-1]=phicand[(runs==float(run)) & (ants==antu[i]) & (evtids==inter[j])]
                    multbef_1[int(coincid)-1]=len(coincs[coincs==coincbef])
            trendcoinc=np.concatenate((trendcoinc,trendcoinc_1))
            multbef=np.concatenate((multbef,multbef_1))          
            thetabef=np.concatenate((thetabef,thetabef_1))
            phibef=np.concatenate((phibef,phibef_1))                                    
            if len(trendcoinc_1[trendcoinc_1==1])==1:
                ind=np.arange(len(trendcoinc_1))
                print(ind[trendcoinc_1==1])
                print("HERE!!!!") 
            #print(trendcoinc_1)
            
            #get coinc times            
            #times_1=np.zeros(int(content[-1][0]))
            #tunix_1=np.zeros(int(content[-1][0]))
            print(run,int(content[-1][0]))
            #strt_time = datetime.datetime.now()
            content=np.loadtxt(repcoinctable+'coinctimes_'+threshold+'_R00'+run+'.txt')
            times_1=content[:,0]
            tunix_1=content[:,1]
            #for i in range(len(times_1)):
                #times_1[i]=np.mean(content[:,3][(content[:,0]==i+1)])
                #times_1[i]=content[(content[:,0]==i+1)][0,3]
                #if threshold=='80':
                #    tunix_1[i]=np.mean(content[:,4][(content[:,0]==i+1) & (content[:,4]>1300000000)])
                #else:
                #    tunix_1[i]=times_1[i]
            times=np.concatenate((times,times_1)) 
            tunix=np.concatenate((tunix,tunix_1))
            #print(times_1)
            #curr_time = datetime.datetime.now()
            #timedelta = curr_time - strt_time
            #print(timedelta.total_seconds())

            print("end times")
            #conscoincfilter
            treject=0.2e8*2 #0.1sec
	    #treject=0.2e9 #1s
            treject2=treject*60*10 #10min
            diff_0=times_1[1:]-times_1[:-1]
            diff_1=np.zeros(len(times_1))+treject2
            diff_1[1:]=diff_0
            diff_2=np.zeros(len(times_1))+treject2
            diff_2[:-1]=diff_0
            reject_1=np.zeros(len(times_1))
            reject_1[(diff_1<treject) | (diff_2<treject)]=1
            reject=np.concatenate((reject,reject_1))
            reject_2=np.zeros(len(times_1))
            reject_2[(diff_1<treject2) | (diff_2<treject2)]=1
            reject2=np.concatenate((reject2,reject_2))
            #print(reject_2)            
        
        

            
        content=np.loadtxt(reprecplane+threshold+'/'+txtfileplane[f])
        print("recons loaded")
        print(len(np.shape(content)))
        if len(np.shape(content))==1:
            if typ=='plane':
                theta_1=np.array([content[3]])
                phi_1=np.array([content[5]])
                rejectphi_1=np.array([0])
                rejectphitime_1=np.array([0])
                chisq_1=np.array([content[7]])
                theta=np.concatenate((theta,theta_1))
                phi=np.concatenate((phi,phi_1))
                rejectphi=np.concatenate((rejectphi,rejectphi_1))
                rejectphitime=np.concatenate((rejectphitime,rejectphitime_1))
                chisq=np.concatenate((chisq,chisq_1))    
            else:  
                x_1=np.array([content[3]])
                y_1=np.array([content[4]])
                z_1=np.array([content[5]])
                x=np.concatenate((x,x_1))
                y=np.concatenate((y,y_1))
                z=np.concatenate((z,z_1))
            continue  
        if typ=='plane': 
            theta_1=content[:,3]
            phi_1=content[:,5]
            preject=10
            diffphi_0=phi_1[1:]-phi_1[:-1]
            diffphi_1=np.zeros(len(phi_1))+preject
            diffphi_1[1:]=diffphi_0
            diffphi_2=np.zeros(len(phi_1))+preject
            diffphi_2[:-1]=diffphi_0
            rejectphi_1=np.zeros(len(phi_1))  
            rejectphi_1[(diffphi_1<preject) | (diffphi_2<preject)]=1
            chisq_1=content[:,7]
            theta=np.concatenate((theta,theta_1))
            phi=np.concatenate((phi,phi_1))
            rejectphi=np.concatenate((rejectphi,rejectphi_1))
            chisq=np.concatenate((chisq,chisq_1))    
            
            rejectphitime_1=np.zeros(len(times_1))
            phicut=10
            radiuscut=4000
            phibis=np.mod(phi_1+phicut,360)
            for i in range(len(times_1)):
                if rejectphitime_1[i]!=1 and radius_1[i]>radiuscut and reject_1[i]!=1:
                    difft=abs(times_1-times_1[i])
                    diffp=abs(phi_1-phi_1[i])                    
                    diffpbis=abs(phibis-phibis[i])
                    rdifft=difft<0.2e9*60*10 #10min
                    rdiffp=(diffp<phicut) | (diffpbis<phicut)
                    if len(difft[rdifft & rdiffp & (radius_1>radiuscut)])>1:
                        #rejectphitime_1[i]=1
                        rejectphitime_1[rdifft & rdiffp & (radius_1>radiuscut)]=1         
            rejectphitime=np.concatenate((rejectphitime,rejectphitime_1))

            
        else:
            x_1=content[:,3]
            y_1=content[:,4]
            z_1=content[:,5]
            x=np.concatenate((x,x_1))
            y=np.concatenate((y,y_1))
            z=np.concatenate((z,z_1))
   
       

        
    #if typ=='plane':
    return x,y,z,theta,phi,chisq,times,tunix,reject,reject2,trendcoinc,rejectphi,rejectphitime,multbef,thetabef,phibef
    #else:
    #    return x,y,z



threshold=['50']
theta=[]
phi=[]
chisq=[]
rho=[]
times=[]
diff=[]
reject=[]
reject2=[]
tunix=[]
trendcoinc=[]
rejectphi=[]
rejectphitime=[]
multbef=[]
thetabef=[]
phibef=[]

keep1=[]
keep2=[]

for t in threshold:
    x_,y_,z_,theta_,phi_,chisq_,times_,tunix_,reject_,reject2_,trendcoinc_,rejectphi_,rejectphitime_,multbef_,thetabef_,phibef_=Recons(t)
    #x_,y_,z_=Recons('sph',t)
    theta.append(theta_)
    phi.append(phi_)
    chisq.append(chisq_)
    rho.append(np.sqrt(x_*x_+y_*y_+z_*z_))
    times.append(times_)
    tunix.append(tunix_)
    reject.append(reject_)
    reject2.append(reject2_)    
    trendcoinc.append(trendcoinc_)
    rejectphi.append(rejectphi_)
    rejectphitime.append(rejectphitime_)    
    multbef.append(multbef_)
    thetabef.append(thetabef_)
    phibef.append(phibef_)
    diff_=times_[1:]-times_[:-1]
    diff__=np.zeros(len(times_))
    diff__[1:]=diff_
    diff.append(diff__)
 

 
nbinsphi=360
nbinstheta=90

ind=np.arange(len(trendcoinc[0]))
print(trendcoinc[0][trendcoinc[0]>0], ind[trendcoinc[0]>0], multbef[0][multbef[0]>0])


for i in range(len(phi)):   
    plt.plot(np.arange(17),np.arange(17))
    plt.scatter(multbef[i][multbef[i]>0],trendcoinc[i][trendcoinc[i]>0],c='tab:orange',alpha=0.3)
    plt.grid()
    plt.xlabel('Multiplicity (without NN)')
    plt.ylabel('Multiplicity (after NN)')  
    plt.title('TREND Candidates only')  
    #if showstat:
    plt.show()
    #plt.close()

for i in range(len(theta)):
    theta[i]=180-theta[i]
    theta[i][theta[i]>90]=180-theta[i][theta[i]>90]
    
for i in range(len(phi)):   
    #ref is east with +90
    phi[i]=phi[i]+180+90   
    
  
for i in range(len(phi)):  
    print(trendcoinc[i][phibef[i]>-1][abs(np.mod(phibef[i][phibef[i]>-1]+90,360)-np.mod(phi[i][phibef[i]>-1],360))>10])
    print(len(phibef[i][phibef[i]>-1]))   
    plt.plot(np.arange(360),np.arange(360))
    plt.scatter(np.mod(phibef[i][phibef[i]>-1]+90,360),np.mod(phi[i][phibef[i]>-1],360),c='tab:orange',alpha=0.3)
    plt.grid()
    plt.xlabel('Phi (without NN)')
    plt.ylabel('Phi (after NN)')  
    plt.title('TREND Candidates only')  
    #if showstat:
    plt.show()
    #plt.close()
   
radiuscut=3000
chisqcut=1000
for i in range(len(phi)):
    #keep1.append((rho[i]>4000) & (theta[i]<80) & (reject[i]<1) & (chisq[i]<700) & (rejectphi[i]<1)) #not chisq<30 because of the lack of traces for recons 
    keep1.append((rho[i]>radiuscut) & (theta[i]<80) & (rejectphitime[i]<1) & (chisq[i]<chisqcut) & (reject[i]<1))
    #keep3=(rho[i]>3000) & (theta[i]<80) & (reject2[i]<1) & (chisq[i]<700)
    #keep2.append((rho[i]>4000) & (theta[i]<80) & (reject[i]<1) & (chisq[i]<700) & (rejectphi[i]<1) & (trendcoinc[i]>0))
    keep2.append((rho[i]>radiuscut) & (theta[i]<80) & (rejectphitime[i]<1) & (chisq[i]<chisqcut)  & (trendcoinc[i]>0) & (reject[i]<1))    
    #keep4[i]=(rho[i]>3000) & (theta[i]<80) & (reject2[i]<1) & (chisq[i]<700) & (trendcoinc[i]>0)
    print(len(times[i]))
    print(len(times[i][trendcoinc[i]>0]))
    print(len(times[i][keep1[i]]))
    print(len(times[i][keep2[i]]))
    

    
    
'''for i in range(len(diff)):
    plt.hist(diff[i][diff[i]<10000],bins=30, range=(0,10000))
    plt.hist(diff[i][(rho[i]>4000) & (theta[i]<80) & (chisq[i]<30)],bins=30, range=(0,10000))
    plt.yscale('log')
    #plt.show()
    plt.close()
    
    plt.plot(tunix[i][(tunix[i]!=0)],np.mod(phi[i][(tunix[i]!=0)],360),'*')
    plt.plot(tunix[i][keep1[i] & (tunix[i]!=0)], np.mod(phi[i][keep1[i] & (tunix[i]!=0)],360),'*')
    plt.plot(tunix[i][keep2[i] & (tunix[i]!=0)], np.mod(phi[i][keep2[i] & (tunix[i]!=0)],360),'*')
    plt.grid()
    plt.xlabel('Unix time')
    plt.ylabel('Phi')
    plt.savefig(PBS_PATH+'tphi.png')
    plt.show()'''
    
for i in range(len(theta)):    
    plt.hist(theta[i], bins=nbinstheta, histtype='step', range=(0,90),linewidth=2)
plt.grid()
plt.xlabel('Theta')
plt.legend(labels=['80%','90%', '92%' , '95%'])
plt.savefig(PBS_PATH+'theta.png')
#if showstat:
plt.show()

nbinstheta=30
for i in range(len(theta)):   
    #plt.hist(theta[i][keep3], bins=nbinstheta, histtype='step',range=(0,90), linewidth=2)
    plt.hist(theta[i][keep1[i]], bins=nbinstheta, histtype='step',range=(0,90), linewidth=2)  
    #plt.hist(theta[i][keep4], bins=nbinstheta, histtype='step',range=(0,90), linewidth=2)
    plt.hist(theta[i][keep2[i]], bins=nbinstheta, histtype='step',range=(0,90), linewidth=2)       
plt.grid()
plt.xlabel('Theta (with cuts)')
plt.legend(labels=['80% all','80% cand'])
plt.savefig(PBS_PATH+'thetacuts.png')
#if showstat:
plt.show()
#plt.close()


for i in range(len(phi)):   
    plt.hist(np.mod(phi[i],360), bins=nbinsphi, histtype='step', range=(0,360),linewidth=2)
plt.grid()
plt.xlabel('Phi')
plt.legend(labels=['80%','90%', '92%', '95%'])
plt.savefig(PBS_PATH+'phi.png')
#if showstat:
plt.show()
#plt.close()

nbinsphi=30
for i in range(len(phi)):   
    #plt.hist(np.mod(phi[i][keep3],360), bins=nbinsphi, histtype='step',range=(0,360), linewidth=2)
    plt.hist(np.mod(phi[i][keep1[i]],360), bins=nbinsphi, histtype='step',range=(0,360), linewidth=2)    
    #plt.hist(np.mod(phi[i][keep4],360), bins=nbinsphi, histtype='step',range=(0,360), linewidth=2)
    plt.hist(np.mod(phi[i][keep2[i]],360), bins=nbinsphi, histtype='step',range=(0,360), linewidth=2)      
plt.grid()
plt.xlabel('Phi')
plt.legend(labels=['coincs','coincs label gerbe'])
plt.savefig(PBS_PATH+'phicuts.png')
#if showstat:
plt.show()
#plt.close()

for i in range(len(rho)):   
    plt.hist(rho[i][keep1[i] & (rho[i]<10000)], bins=nbinstheta, histtype='step', linewidth=2)
    plt.hist(rho[i][keep2[i] & (rho[i]<10000)], bins=nbinstheta, histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Rho')
#plt.legend(labels=['90%', '92%' , '95%'])
plt.savefig(PBS_PATH+'rho.png')
#if showstat:
plt.show()
#plt.close()

for i in range(len(chisq)):   
    plt.hist(chisq[i][keep1[i]], bins=nbinstheta, histtype='step', linewidth=2)
    plt.hist(chisq[i][keep2[i]], bins=nbinstheta, histtype='step', linewidth=2)
plt.grid()
plt.xlabel('Chi Squared')
#plt.legend(labels=['90%', '92%' , '95%'])
plt.savefig(PBS_PATH+'chisq.png')
#if showstat:
plt.show()
#plt.close()


