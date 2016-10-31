
# import sys
# sys.path.append('..')
# sys.path.append('../..')

import numpy as np
from pulse2percept import electrode2currentmap as e2cm
from pulse2percept import effectivecurrent2brightness as ec2b
from pulse2percept import utils
from pulse2percept import files as n2sf
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar# import npy2savedformats as n2sf
import matplotlib.pyplot as plt
import importlib as imp
#imp.reload(n2sf)

def findampval(amp, ecs, retina, rsample, whichlayer):                                
    pt=e2cm.Psycho2Pulsetrain(tsample=tm.tsample, current_amplitude=amp,dur=.6, delay=10/1000, 
                              pulse_dur=pd / 1000,interphase_dur=10/1000, freq=2)    
    resp =  ec2b.pulse2percept(tm, ecs,r, [pt], rsample=rsample,  dolayer=whichlayer, dojit=True, engine='serial')   
    return (( (np.max(resp.data)*1000) - 67.89) ** 2)
       
xlist=[]
ylist=[]
rlist=[] #electrode radius, microns
hlist=[] # lift of electrode from retinal surface, microns
e_spacing=525 # spacing in microns
for x in np.arange(-1, 1, e_spacing):  
    for y in np.arange(-1, 1, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100) # electrode radiues
        hlist.append(0); #179.6) # electrode lift from retinal surface,      
        # epiretinal array - distance to the ganglion layer
        # subretinal array - distance to the bipolar layer
        # in Argus 1 179.6 is a good approx of height in a better patient
        
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist,hlist, ptype='epiretinal') 
                
# create retina, input variables include the sampling and how much of the retina is simulated, in microns   
# (0,0 represents the fovea) 
retinaname='SmallL80S75WL500'
r = e2cm.Retina(axon_map=None,sampling=75, ylo=-500, yhi=500, xlo=-500, xhi=500, axon_lambda=8)     

# the effective current spread that incorporates axonal stimulation    
    
myout=[]
d=.1
fps=30
pt=[]
inl_out=[]
nfl_out=[]

modelver='Krishnan' 

#for d in [.1, .2, .45, .75, 1., 2., 4., 8., 16., 32.]:
tm = ec2b.TemporalModel() 
rsample=int(np.round((1/tm.tsample) / 60 )) # resampling of the output to fps
# at 0 off the retinal surface a 0.45 pulse in the nfl gives a response of 1

[ecs, cs]  = r.electrode_ecs(e_all)  

inl_amp = []
nfl_amp = []
for pd in [.01, .02, .04, .08, .16, .32, .64, 1.28, 2.56, 5.12, 10.24, 20.48]:
    xamp=120
    dolayer='INL'
    tmp=minimize(findampval, xamp, args=(ecs, r,  rsample, 'INL', ))
    inl_amp.append(tmp.x) 
    print(pd)
    print('minimized inl layer')
    print(tmp.x)
    dolayer='NFL'
    tmp=minimize(findampval, xamp, args=(ecs, r,  rsample, 'NFL', ))
    inl_amp.append(tmp.x)
    print('minimized nfl layer')
    print(tmp.x)

#inl_r = ec2b.pulse2percept(tm, ecs, r, [pt_2], rsample=rsample, dolayer='INL', dojit=False, engine='serial')
#def pulse2percept(tm, ecs, retina, ptrain, rsample, dolayer,
#                  engine='joblib', dojit=True, n_jobs=-1, tol=.05):                           
#inl_r = ec2b.pulse2percept(tm, ecs, r, [pt_2], rsample=rsample, dolayer='INL', dojit=False, engine='serial')
#
#omparenflinl(.636, ecs, r, [pt_2], [pt_01], rsample, False, 'serial')
#myout=minimize(comparenflinl, x0, args=(ecs, r, [pt_2], [pt_01], rsample, False, 'serial', ))
                