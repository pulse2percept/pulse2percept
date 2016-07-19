
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

brightnessmovie=np.random.rand(100,200, 40)
[onmovie, offmovie] = ec2b.onoffFiltering(brightnessmovie, np.array([5, 10])


# relevant bits of Dorn paper
# Surgeons were instructed to place the array centered
#over the macula. 
#Each of the 60 electrodes (in a 6 × 10 grid) were 200 μm 
#in diameter
# The array (along the diagonal) covered an area
# of retina corresponding to about 20° in visual angle
#  assuming 293 μm on the retina equates to 1° of visual angle.
# a=1.72, sqrt((a*6)^2+(a*10)^2)=20
# so the 10 side is 17.2 degrees
# the 6 side is 10.32 degrees 


# Create electrode array for the Argus 2
# 293 μm equals 1 degree
# electrode spacing is done in microns
# when you include the radius of the electrode 
# the electrode centers span +/- 2362 and +/- 1312

xlist=[]
ylist=[]
rlist=[] #electrode radius, microns
llist=[] # lift of electrode from retinal surface, microns
e_spacing=525
for x in np.arange(-2362, 2364, e_spacing):  
    for y in np.arange(-1312, 1314, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100) # electrode radiues
        llist.append(0) # electrode lift from retinal surface
        
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist,llist)
del xlist, ylist, rlist, llist

# create retina, input variables include the sampling 
# and how much of the retina is simulated, in microns   
# (0,0 represents the fovea)   
r = e2cm.Retina(axon_map='../retina_1700by2900L80.npz', 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2900, xhi=2900, axon_lambda=8)     

e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))

[ecs_list, cs_list]  = r.electrode_ecs(e_all, integrationtype='maxrule')
#THIS HAS A NORMALIZATION STEP IN THERE, DO WE WANT IT    
       
# create movie
# original screen was [52.74, 63.32]  visual angle
# res=[768 ,1024] # resolution of screen
#pixperdeg=degscreen/res
# no need to simulate the whole movie, just match it to the electrode array
# xhi+xlo/294 (microns per degree)
fps=30
degscreen=[10.32+5, 17.2+5] # match to array visual angle,
res=[e_rf[0].shape[0],e_rf[1].shape[1]] # resolution of screen
fps=30
# the bar is 1.4 inches in width at 12 inches, 
# corresponds to 6.67 degrees visual angle
bar_width=6.77 
[X,Y]=np.meshgrid(np.linspace(-degscreen[1]/2, degscreen[1]/2, res[1]), 
np.linspace(-degscreen[0]/2, degscreen[0]/2, res[0]));

for o in np.arange(0, 2*np.pi,2): #DEBUG 2*np.pi/4): # each orientation
    M=np.cos(o)*X +np.sin(o)*Y
 #   for sp in range (32:32): # DEBUG each speed, eventually 8:32  
    for sp in np.arange(32, 33, 1): #(7.9, 31.6, 3):
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
    
        # create pulsetrain corresponding to the movie
        pt=[]
        for rf in e_rf:
            rflum= e2cm.retinalmovie2electrodtimeseries(rf, movie)         
            ptrain=e2cm.Movie2Pulsetrain(rflum)
            ptrain=e2cm.accumulatingvoltage(ptrain) 
            pt.append(ptrain)
            
            #  plt.plot(rflum)  plt.plot(pt[ct].data)   plt.plot(ptrain.data)
        
        del movie
        

        tm = ec2b.TemporalModel()
    
        rs=1/(fps*ptrain.tsample) 
    #fr=np.zeros([e_rf[0].shape[0],e_rf[0].shape[1], len(pt[0].data)])

    # This seems obsolete
        
        sr_tmp=ec2b.calc_pixel(0, 0, r, ecs_list, pt, tm, rs, dojit=False) 
        brightness_movie = np.zeros((r.gridx.shape[0], r.gridx.shape[1], sr_tmp.shape[0]))

        def parfor_calc_pixel(arr, idx, r, ecs_list, pt, tm, rs, dojit=False):            
            sr=ec2b.calc_pixel(idx[1], idx[0], r, ecs_list, pt, tm, rs, dojit)           
            return sr.data     
  
        brightness_movie = utils.parfor(brightness_movie, parfor_calc_pixel, r, ecs_list, pt, tm, rs, dojit=False, n_jobs=1, axis=-1)        

      #  brightnessmovie[yy, xx, :] = sr_rs
        filename='Bar_S' + str(sp) + '_O' + str(o)      
        np.save(filename, brightness_movie) 
        sio.savemat(filename, brightness_movie)
    
   

