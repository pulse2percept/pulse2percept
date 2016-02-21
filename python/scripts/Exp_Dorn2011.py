
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


# create  retina, 
#should be the size of the electrode array plus at least half 
# the electrode spacing

r = e2cm.Retina(axon_map='../retina_1700by2800.npz', 
                sampling=25, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800)
   
# Create electrode array 
# 293 μm equals 1 degree
# electrode spacing is done in microns

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

e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))


# create movie
# original screen was [52.74, 63.32]  visual angle
# res=[768 ,1024] # resolution of screen
#pixperdeg=degscreen/res
# no need to simulate the whole movie, just match it to the retina
# xhi+xlo/294 (microns per degree)

degscreen=[11.6, 19.1] # array visual angle,
res=[e_rf[0].shape[0],e_rf[1].shape[1]] # resolution of screen

fps=30
bar_width=6.7
[X,Y]=np.meshgrid(np.linspace(-degscreen[1]/2, degscreen[1]/2, res[1]), 
np.linspace(-degscreen[0]/2, degscreen[0]/2, res[0]));

for o in np.arange(0, 2*np.pi, 2*np.pi/4): # each orientation
    M=np.cos(o)*X +np.sin(o)*Y

 #   for sp in range (32:32): # DEBUG each speed, eventually 8:32  
    sp=8
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
    
    pt=[]
    for rf in e_rf:
        rflum= e2cm.retinalmovie2electrodtimeseries(rf, movie) 
        #plt.plot(rflum)
        ptrain=e2cm.Movie2Pulsetrain(rflum)
        #plt.plot(ptrain.data)
        pt.append(ptrain) 
     #   plt.plot(pt[ct].data)
          
    [ecs_list, cs_list]  = r.electrode_ecs(e_all)    
    tm1 = ec2b.TemporalModel()
    #fr=np.zeros([e_rf[0].shape[0],e_rf[0].shape[1], len(pt[0].data)])

    brightnessmovie = np.zeros(r.gridx.shape + ((moviedur+1)*fps,))
    #DEBUG for xx in range(ecm.shape[0]):
    #    for yy in range(ecm.shape[1]):
    for xx in range(r.gridx.shape[0]):
        for yy in range(r.gridx.shape[1]):
            ecm = r.ecm(xx, yy, ecs_list, pt)
            fr = tm1.fast_response(ecm, dojit=False)    
            ca = tm1.charge_accumulation(fr, ecm)
            sn = tm1.stationary_nonlinearity(ca)
            sr = tm1.slow_response(sn)
#            sr.data=sr.data[0:int(np.round((moviedur+1)/sr.tsample))]
            intfunc= interpolate.interp1d(np.linspace(0, len(sr.data),len(sr.data)),
                                      sr.data)
            
            sr_rs=intfunc(np.linspace(0, len(sr.data), len(sr.data)*sr.tsample*fps))
            brightnessmovie[xx, yy, :] = sr_rs

    np.save('movie20160220.npy', brightnessmovie)      
    
