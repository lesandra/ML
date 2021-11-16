E='1e18'
run='R003562'
ID='149'
Ndata=200

SIMU_DATA_PATH='/sps/trend/zhaires/trend-50/without-freq-iron/'+E+'/voltages/'
RAW_DATA_PATH='/sps/trend/slecoz/'+run+'/'
ML_DATA_PATH='/sps/trend/slecoz/ML'+run+'/'
GAIN_DATA_PATH='/sps/trend/slecoz/gainkeep'

SCALE=280e-3/37
TSAMPLING=5e-9
ibufft=4
ib=1024
FREQMIN=50e6 #inconsistency between the 2 frequencies in evasimu.matlab
FREQMAX=100e6
