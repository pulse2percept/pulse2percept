
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from scipy import interpolate
import scipy.io as sio
from utils import TimeSeries
import matplotlib.pyplot as plt
import utils


fps=30

xlist=[]
ylist=[]
rlist=[]
e_spacing=525
 
# Create electrode array 
# 293 μm equals 1 degree
# electrode spacing is done in microns
 
for x in np.arange(-2362, 2364, e_spacing):  
    for y in np.arange(-1312, 1314, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100) 
        
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist)

del xlist, ylist, rlist
        
r = e2cm.Retina(axon_map='../retina_1700by2900L80.npz', 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2900, xhi=2900, axon_lambda=8)
     
e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))


# create movie
# original screen was [52.74, 63.32]  visual angle
# res=[768 ,1024] # resolution of screen
#pixperdeg=degscreen/res
# no need to simulate the whole movie, just match it to the electrode array
# xhi+xlo/294 (microns per degree)

degscreen=[13.31, 20.82] # array visual angle,
res=[e_rf[0].shape[0],e_rf[1].shape[1]] # resolution of screen

fps=30
bar_width=6.7
[X,Y]=np.meshgrid(np.linspace(-degscreen[1]/2, degscreen[1]/2, res[1]), 
np.linspace(-degscreen[0]/2, degscreen[0]/2, res[0]));

for o in np.arange(0, 2*np.pi, 2*np.pi/4): # each orientation
    M=np.cos(o)*X +np.sin(o)*Y

 #   for sp in range (32:32): # DEBUG each speed, eventually 8:32  
    for sp in np.arange(8, 20, 3):
        movie=np.zeros((res[0],res[1], int(np.ceil((70/5)*30))))
        st=np.min(M)
        fm_ct=1
        while (st<np.max(M)):
            img=np.zeros(M.shape)
            ind=np.where((M>st) & (M<st+bar_width))
            img[ind]=1    
            movie[:,:, fm_ct]=img
            fm_ct=fm_ct+1
            st=st+(sp/fps)   
         
        movie=movie[:,:, 0:fm_ct-1]   
        moviedur=movie.shape[2]/fps
        del M, img
    
        pt=[]
        for rf in e_rf:
            rflum= e2cm.retinalmovie2electrodtimeseries(rf, movie) 
        #plt.plot(rflum)
            ptrain=e2cm.Movie2Pulsetrain(rflum)
        #plt.plot(ptrain.data)
            pt.append(ptrain) 
     #   plt.plot(pt[ct].data)
        del movie
          
        [ecs_list, cs_list]  = r.electrode_ecs(e_all)    
        tm1 = ec2b.TemporalModel()
    
        rs=1/(fps*pt[0].tsample) 
    #fr=np.zeros([e_rf[0].shape[0],e_rf[0].shape[1], len(pt[0].data)])

        sr_tmp=ec2b.calc_pixel(0, 0, r, ecs_list, pt, tm1, rs, dojit=False) 
        brightness_movie = np.zeros((r.gridx.shape[0], r.gridx.shape[1], sr_tmp.shape[0]))

        def parfor_calc_pixel(arr, idx, r, ecs_list, pt, tm, rs, dojit=False):            
            sr=ec2b.calc_pixel(idx[1], idx[0], r, ecs_list, pt, tm, rs, dojit)           
            return sr.data     
  
        brightness_movie = utils.parfor(brightness_movie, parfor_calc_pixel, r, ecs_list, pt, tm1, rs, dojit=False, n_jobs=1, axis=-1)        

      #  brightnessmovie[yy, xx, :] = sr_rs
        filename='Bar_S' + str(sp) + '_O' + str(o)      
        np.save(filename, brightness_movie) 
        sio.savemat(filename, brightness_movie)
    
   

