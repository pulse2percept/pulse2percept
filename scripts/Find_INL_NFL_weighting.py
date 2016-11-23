
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

def comparenflinl(ll, ecs, retina, pt_inl, pt_nfl, rsample, dolayer, engine='joblib', dojit=True, n_jobs=-1, tol=.05):    
    tm = ec2b.TemporalModel(lweight=ll)                             
    inl_r =  ec2b.pulse2percept(tm, ecs,r, pt_inl, rsample=rsample,  dolayer='INL', dojit=False, engine='serial')
    
    nfl_r =  ec2b.pulse2percept(tm, ecs, r, pt_nfl, rsample=rsample, dolayer='NFL', dojit=False, engine='serial')
    print(ll)
    print(np.max(inl_r.data))
    print(np.max(nfl_r.data))
    return (np.max(10*inl_r.data)-np.max(10*nfl_r.data)) ** 2
    
# Recreation of the Dorn 2013 paper, where subjects had to guess the direction of motion of a moving bar

# Surgeons were instructed to place the array centered over the macula (0, 0). 
# Each of the 60 electrodes (in a 6 × 10 grid) were 200 μm in diameter
# The array (along the diagonal) covered an area of retina corresponding to 
#about 20° in visual angle  assuming 293 μm on the retina equates to 1° of 
#visual angle. a=1.72, sqrt((a*6)^2+(a*10)^2)=20 so the 10 side is 17.2 degrees, 
#the 6 side is 10.32 degrees 

# Create electrode array for the Argus 2
# 293 μm equals 1 degree, electrode spacing is done in microns
# when you include the radius of the electrode  the electrode centers span +/- 2362 and +/- 1312

# based on Ahuja et al 2013. Factors affecting perceptual thresholds in Argus ii retinal prosthesis subjects
# (figure 4, pdf is in retina folder) the mean height from the array should be  179.6 μm
# with a range of ~50-750μm

# Alternative model is currently the 'Krishnan' model which assumes that charge accumulation
# occurs at the electrode, not neurally. The models are in fact metamers of each other if one is
# only simulating the NFL
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

pt_01=e2cm.Psycho2Pulsetrain(tsample=tm.tsample, current_amplitude=120,dur=.6, delay=10/1000, pulse_dur=.1/1000.,interphase_dur=10/1000, freq=2)
pt_2=e2cm.Psycho2Pulsetrain(tsample=tm.tsample, current_amplitude=120, dur=.6, delay=10/1000, pulse_dur=2/1000.,interphase_dur=10/1000, freq=2) 
#def pulse2percept(tm, ecs, retina, ptrain, rsample, dolayer,
#                  engine='joblib', dojit=True, n_jobs=-1, tol=.05):
                            

#inl_r = ec2b.pulse2percept(tm, ecs, r, [pt_2], rsample=rsample, dolayer='INL', dojit=False, engine='serial')
#

comparenflinl(.636, ecs, r, [pt_2], [pt_01], rsample, False, 'serial')
#myout=minimize(comparenflinl, x0, args=(ecs, r, [pt_2], [pt_01], rsample, False, 'serial', ))
                