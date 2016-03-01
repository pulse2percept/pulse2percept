
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from scipy import interpolate
from utils import TimeSeries
import matplotlib.pyplot as plt

# import imp 
# imp.reload(effectivecurrent2brightness.py)

xlist=[]
ylist=[]
rlist=[]
e_spacing=525
 
for x in np.arange(-2362, 2364, e_spacing):  
    for y in np.arange(-1312, 1314, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100)  
        
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist)

    
for al in np.arange(0,12, .5):
    retinaname='retina_1700x2900L' + str(al) +'.npy'
    r = e2cm.Retina(axon_map=retinaname, 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800,axon_lambda=al)
   
    e_rf=[]
    for e in e_all.electrodes:
       e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))
          
    
    for ee in np.arange(0, len(xlist)): 
        pt=[]
        for ct, rf in enumerate(e_rf):
            if ct==ee:           
                e2cm.Psycho2Pulsetrain(rf, current_amplitude=40)
            else:
                ptrain=e2cm.Psycho2Pulsetrain(rf, current_amplitude=0)
            pt.append(ptrain) 
    
        [ecs_list, cs_list]  = r.electrode_ecs(e_all)    
        tm1 = ec2b.TemporalModel()
    #fr=np.zeros([e_rf[0].shape[0],e_rf[0].shape[1], len(pt[0].data)])

        for xx in range(r.gridx.shape[1]):
            for yy in range(r.gridx.shape[0]):
                ecm = r.ecm(xx, yy, ecs_list, pt)
                fr = tm1.fast_response(ecm, dojit=False)    
                ca = tm1.charge_accumulation(fr, ecm)
                sn = tm1.stationary_nonlinearity(ca)
                sr = tm1.slow_response(sn)
                intfunc= interpolate.interp1d(np.linspace(0, len(sr.data),len(sr.data)),
                                      sr.data)
                sr_rs=intfunc(np.linspace(0, len(sr.data), len(sr.data)*sr.tsample*fps))
                if xx==0 and yy==0:
                    brightnessmovie = np.zeros(r.gridx.shape + (len(sr_rs),))  
           
                brightnessmovie[yy, xx, :] = sr_rs
    filename='movie20160222_' + str(np.round(o,3)) +'.npy'
    
    np.save(filename, brightnessmovie)      
    
