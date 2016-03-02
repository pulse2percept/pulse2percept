
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from scipy import interpolate
from utils import TimeSeries
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('Agg')
import imp
import utils
imp.reload(utils)

def calc_pixel(arr, idx, r, ecs_list, pt, tm, rs):
     ecm = r.ecm(*idx, ecs_list, pt)
     fr = tm.fast_response(ecm, dojit=False)    
     ca = tm.charge_accumulation(fr, ecm)
     sn = tm.stationary_nonlinearity(ca)
     sr = tm.slow_response(sn)
     sr.resample(rs)
     return sr.data
     
def brightness(r, xx, yy, ecs_list, pt, tm, rs):
    ecm = r.ecm(xx, yy, ecs_list, pt)
    fr = tm.fast_response(ecm, dojit=False)    
    ca = tm.charge_accumulation(fr, ecm)
    sn = tm.stationary_nonlinearity(ca)
    sr = tm.slow_response(sn)
    sr.resample(rs)
    return sr.data
    
# import imp 
# imp.reload(effectivecurrent2brightness.py)

fps=30 #fps for the final movie

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
    print("loading or creating a retina" )
    retinaname='retina_1700x2900L' + str(al*10)
    r = e2cm.Retina(axon_map=retinaname, 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800,axon_lambda=al)
   
    print("creating receptive fields" )    
    e_rf=[]
    for e in e_all.electrodes:
       e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))
          

    for ee in np.arange(0, len(xlist)): 
        print("Creating waveform for elecrodes" )
        pt=[]
        for ct, rf in enumerate(e_rf):
            if ct==ee:           
                ptrain=e2cm.Psycho2Pulsetrain(current_amplitude=40, dur=0.25, pulse_dur=.45/1000., 
                                               interphase_dur=.45/1000., tsample=.05/1000.)
            else:
                ptrain=e2cm.Psycho2Pulsetrain(current_amplitude=40, dur=0.25, pulse_dur=.45/1000., 
                                               interphase_dur=.45/1000., tsample=.05/1000.)
            pt.append(ptrain) 
            
            
        [ecs_list, cs_list]  = r.electrode_ecs(e_all)    
        tm1 = ec2b.TemporalModel()

        rs=1/(fps*pt[0].tsample)                     
        sr_tmp = brightness(r, 0, 0, ecs_list, pt, tm1, rs) 
        brightness_movie = np.zeros((4, 4, sr_tmp.shape[0]))        

        idx = list(product(*(range(s) for s in brightness_movie.shape[:-1])))

      #  %%timeit
       # ff = np.array([calc_pixel(brightness_movie, i, r, ecs_list, pt, tm1) for i in idx]).reshape(brightness_movie.shape)

        #timeit
        answer = utils.parfor(brightness_movie, calc_pixel, r, ecs_list, pt, tm1, n_jobs=10, axis=-1)

      #  brightnessmovie[yy, xx, :] = sr_rs
        filename='SE_' + retinaname + '_E' + str(ee)      
        np.save(filename, brightnessmovie)   
     #   moviefilename='singleelectrode_' + retinaname + str(ee)
     #   npy2movie(filename,moviefilename)

