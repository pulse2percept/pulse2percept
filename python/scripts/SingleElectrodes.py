
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from scipy import interpolate
from utils import TimeSeries
import matplotlib.pyplot as plt
from itertools import product
import utils
import time
import scipy



import matplotlib
matplotlib.use('Agg')
import imp
    

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

for al in np.arange(12,16, 1):
    printstr='loading or creating a retina ' + str(al) + ' '+  time.strftime("%H:%M") 
    print(printstr)
    retinaname='retina_1700x2900L' + str(al*10)
    r = e2cm.Retina(axon_map=retinaname + '.npz', 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800,axon_lambda=al)
     
    e_rf=[]
    for e in e_all.electrodes:
       e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))
          

    for ee in np.arange(2, len(xlist), 6): 
        printstr='simulating a single electrode ' + str(ee) + ' ' +  time.strftime("%H:%M") 
        print(printstr)
    
        pt=[]
        for ct, rf in enumerate(e_rf):
            if ct==ee:           
                ptrain=e2cm.Psycho2Pulsetrain(current_amplitude=3, dur=0.25, pulse_dur=.5/1000., 
                                               interphase_dur=.5/1000., tsample=.1/1000.)
            else:
                ptrain=e2cm.Psycho2Pulsetrain(current_amplitude=0, dur=0.25, pulse_dur=.5/1000., 
                                               interphase_dur=.5/1000., tsample=.1/1000.)
            pt.append(ptrain) 
            
            
        [ecs_list, cs_list]  = r.electrode_ecs(e_all)    
        tm1 = ec2b.TemporalModel()

        rs=1/(fps*pt[0].tsample)  
                
        sr_tmp=ec2b.calc_pixel(0, 0, r, ecs_list, pt, tm1, rs, dojit=False) 
        brightness_movie = np.zeros((r.gridx.shape[0], r.gridx.shape[1], sr_tmp.shape[0]))
       
        #brightness_movie = np.zeros((4, 4, sr_tmp.shape[0]))          
        def parfor_calc_pixel(arr, idx, r, ecs_list, pt, tm, rs, dojit=False):            
            sr=ec2b.calc_pixel(idx[1], idx[0], r, ecs_list, pt, tm, rs, dojit)           
            return sr.data        

        brightness_movie = utils.parfor(brightness_movie, parfor_calc_pixel, r, ecs_list, pt, tm1, rs, dojit=False, n_jobs=12, axis=-1)        

      #  brightnessmovie[yy, xx, :] = sr_rs
        filename='SE_' + retinaname + '_E' + str(ee)      
        np.save(filename, brightness_movie) 
        scipy.misc.imsave(filename +'.jpg', brightness_movie[:, :, 10])
   
     #   moviefilename='singleelectrode_' + retinaname + str(ee)
     #   npy2movie(filename,moviefilename)

