
# import sys
# sys.path.append('..')
# sys.path.append('../..')

import numpy as np
from pulse2percept import electrode2currentmap as e2cm
from pulse2percept import effectivecurrent2brightness as ec2b
from pulse2percept import utils
from pulse2percept import files as n2sf
# import npy2savedformats as n2sf
import matplotlib.pyplot as plt
import importlib as imp
#imp.reload(n2sf)

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

modelver='Nanduri' # this is the standard model based on the Nanduri 2012 paper. 
# Alternative model is currently the 'Krishnan' model which assumes that charge accumulation
# occurs at the electrode, not neurally. The models are in fact metamers of each other,

xlist=[]
ylist=[]
rlist=[] #electrode radius, microns
hlist=[] # lift of electrode from retinal surface, microns
e_spacing=525 # spacing in microns
for x in np.arange(-0, 500, e_spacing):  
    for y in np.arange(-0, 500, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100) # electrode radiues
        hlist.append(179.6) 
        # electrode lift from retinal surface, 
        # epiretinal array - distance to the ganglion layer
        # subretinal array - distance to the bipolar layer
              
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist,hlist, ptype='subretinal')
del xlist, ylist, rlist, hlist 
      
# create retina, input variables include the sampling and how much of the retina is simulated, in microns   
# (0,0 represents the fovea) 
retinaname='SmallL80S150'
r = e2cm.Retina(axon_map=None,sampling=150, ylo=-700, yhi=700, xlo=-500, xhi=500, axon_lambda=8)     
   
e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))
    
    
[ecs, cs]  = r.electrode_ecs(e_all, integrationtype='maxrule')      

tm = ec2b.TemporalModel()
myout=[]

rsample=(1/30)*tm.tsample
for d in range(1, 40):
    ptrain=e2cm.Psycho2Pulsetrain(current_amplitude=3, dur=.5, pulse_dur=d/1000.,interphase_dur=.45/1000, tsample=tm.tsample, freq=5)
    tmp = ec2b.pulse2percept(tm, ecs, r, ptrain, rsample,  dojit=False)
    myout.append(tmp)                  

    
