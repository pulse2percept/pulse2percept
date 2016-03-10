
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
from PIL import Image




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
        
r = e2cm.Retina(axon_map='C:\\Users\\Pulse2Percept\\Documents\\pulse2percept\\python\\scripts\\retinae\\retina_1700x2900_L8.npz', 
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
im=Image.open('whiteonblack.jpg')
imarray=np.array(im)

movie=np.zeros((res[0],res[1], 15))

for f in range(0, 15, 1):
    movie[:,:, f]=imarray/255

pt=[]
for rf in e_rf:
    rflum= e2cm.retinalmovie2electrodtimeseries(rf, movie) 
        #plt.plot(rflum)
    ptrain=e2cm.Movie2Pulsetrain(rflum)
        #plt.plot(ptrain.data)
    pt.append(ptrain) 
     #   plt.plot(pt[ct].data)
del movie
          
[ecs_mat, cs_mat]  = r.electrode_ecs(e_all)    
# returns a 3D array res x nelectrodes
tm1 = ec2b.TemporalModel()
    
rs=1/(fps*pt[0].tsample) 

#fr=np.zeros([e_rf[0].shape[0],e_rf[0].shape[1], len(pt[0].data)])
brightness_movie = ec2b.pulse2percept(tm1, ecs_mat, r, pt,
              rs, n_jobs=8, dojit=False, tol=.5)
        

      #  brightnessmovie[yy, xx, :] = sr_rs
filename='Bar_S' + str(sp) + '_O' + str(o)      
np.save(filename, brightness_movie) 
sio.savemat(filename, brightness_movie)
#    
#   

