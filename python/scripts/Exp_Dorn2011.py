
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from utils import TimeSeries
import matplotlib.pyplot as plt

# create  retina, should be larger than the elecrtrode array
r = e2cm.Retina(axon_map='../DornMovie.npz', 
                sampling=25, xlo=-1500, xhi=1500, ylo=-2700, yhi=2700)
                #'../axon.npz'
                
# Create electrode array 
# covers 10.32 dva (6 electrodes) by 17.2 (10 electrodes) dva. 
# 293 μm equals 1 degree
# electrode spacing is done in microns
e_spacing=550
xlist=[]
ylist=[]
rlist=[]
for x in np.arange(-1375, 1376, e_spacing):
    for y in np.arange(-2475, 2476, e_spacing):   
        xlist.append(x)
        ylist.append(y)
        rlist.append(100)        
e_all = e2cm.ElectrodeArray(xlist,ylist, rlist)

e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing/25))


# create movie
#degscreen=[52.74, 63.32] # screen visual angle
#res=[768 ,1024] # resolution of screen
#pixperdeg=degscreen/res
degscreen=[10.32, 17.2] # array visual angle, no need to simulate the whole movie
res=[768/4 ,1024/4] # resolution of screen

fps=30

[X,Y]=np.meshgrid(np.linspace(-degscreen[1]/2, degscreen[1]/2, res[1]), 
np.linspace(-degscreen[0]/2, degscreen[0]/2, res[0]));

for o in np.arange(np.pi/180, 360*np.pi/180): # each orientation
    M=np.cos(o)*X +np.sin(o)*Y
    for sp in range (8,32): # each speed  
        movie=np.zeros((res[0],res[1], np.ceil((60 /7)*30)))
        st=np.min(M)
        fm_ct=0
        while (st<np.max(M)):
            img=np.zeros(M.shape)
            ind=np.where((M>st) & (M<st+6.7))
            img[ind]=1
            st=st+(sp/fps)
            movie[:,:, fm_ct]=img
            fm_ct=fm_ct+1
            
        movie=movie[:,:, :fm_ct-1]  
        movie=movie[0:r.gridx.shape[0],0:r.gridy.shape[1], :]         
            
        pt=[]
        for rf in e_rf:
            rflum= e2cm.retinalmovie2electrodtimeseries(rf, movie) 
            pt.append(e2cm.Movie2Pulsetrain(rflum)) 
                           
        ecm = r.ecm(e_all, pt)
        tm1 = ec2b.TemporalModel()
        #sr = np.zeros(ecm.shape)
        #xx = yy = 0
        fr = tm1.fast_response(ecm)
        ca = tm1.charge_accumulation(fr, ecm)
        sn = tm1.stationary_nonlinearity(ca)
        sr = tm1.slow_response(sn).data
        
    