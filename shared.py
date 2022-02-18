#E='1e18'
#run='R003562'
#run='R003564'
#ID='149'
#Ndata=758*10
#Ndata=200

trainingP6byevt=1
trainingP6byant=0
trainingP6hyb=0
traininghyb=0
nhyb=42
trainingP6=0
tpsel=0 # theta inf 70, phi inf 70 ou sup 150
takeoffsat=0

#nP6=1413
#suffix=''

satup=256
satdown=-1
if takeoffsat:
    satup=255
    satdown=0

if tpsel:
    suffix='tpsel'
    nP6=926
    if takeoffsat:
        suffix='tpselnosat'
        nP6=823
else:
    suffix=''
    nP6=1413
    if takeoffsat:
        suffix='nosat'
        np6=1237

    
testonP6=0
testonhybrid=0


ratiotrain=0.7
validintrain=0.2


SPS_PATH='/sps/trend/slecoz/'
PBS_PATH='/pbs/throng/trend/soft/ana/TREND_ML/'

SIMU_DATA_PATH='/sps/trend/zhaires/trend-50/without-freq-iron/'
RAW_DATA_PATH='/trend/home/trirods/data/raw/'
MLP6SIM_DATA_PATH=SPS_PATH+'MLP6SIM/'
GAIN_DATA_PATH=SPS_PATH+'gainkeep/'

P6_DST_PATH=SPS_PATH+'dst_selectedP6/'
P6_DATA_PATH=SPS_PATH+'bin_selectedP6/'
MLP6_DATA_PATH=SPS_PATH+'MLP6/'
BACK_PATH=SPS_PATH+'BACK/'
HYBRID_DATA_PATH=SPS_PATH+'bin_selected_hybrid/'
MLHYB_DATA_PATH=SPS_PATH+'MLhybrid/'
MLP6HYB_DATA_PATH=SPS_PATH+'MLP6hybrid/'

SCALE=280e-3/37
TSAMPLING=5e-9
ibufft=4
ib=1024
FREQMIN=50e6
FREQMAX=100e6
quantization=255

c0=3e8
