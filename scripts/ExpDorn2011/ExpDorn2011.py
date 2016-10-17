
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
for x in np.arange(-2362, 2364, e_spacing):  
    for y in np.arange(-1312, 1314, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100) # electrode radiues
        hlist.append(179.6) 
        # electrode lift from retinal surface, 
        # epiretinal array - distance to the ganglion layer
        # subretinal array - distance to the bipolar layer
        
layers=['INL', 'NFL']      
e_all = e2cm.ElectrodeArray(rlist,xlist,ylist,hlist, ptype='subretinal')
del xlist, ylist, rlist, hlist 
        
# create retina, input variables include the sampling and how much of the retina is simulated, in microns   
# (0,0 represents the fovea) 
retinaname='1700by2900L80S150'
r = e2cm.Retina(axon_map=None, 
                sampling=150, ylo=-1700, yhi=1700, xlo=-2900, xhi=2900, axon_lambda=8)     
   
e_rf=[]
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy,e_spacing))
        
[ecs, cs]  = r.electrode_ecs(e_all, integrationtype='maxrule')    

tm = ec2b.TemporalModel()
        
# create movie
# original screen was [52.74, 63.32]  visual angle, res=[768 ,1024] # resolution of screen
# pixperdeg=degscreen/res
# no need to simulate the whole movie, just match it to the electrode array, xhi+xlo/294 (microns per degree)
        
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
            #  plt.plot(rflum)  plt.plot(pt[ct].data)   plt.plot(ptrain.data)
            pt.append(ptrain) 
        del movie
  
       
        rsample=(1/fps)*pt[0].tsample # factor by which movies resampled for presentation 
        boom
        brightness_movie = ec2b.pulse2percept(tm, ecs, r, pt, rsample)
                      
          
       # FILTERING BY ON OFF CELLS
       # foveal vision is ~60 cpd. 293μm on the retina corresponds to 1 degree, so the smallest receptive field probably covers 293/60 ~=5μm, 
       # we set the largest as being 10 times bigger than that numbers roughly based on 
       # Field GD & Chichilnisky EJ (2007) Information processing in the primate retina: circuitry and coding. Annual Review of Neuroscience 30:1-30
       # Chose 30 different sizes fairly randomly
        retinasizes=np.unique(np.ceil(np.array(np.linspace(5, 50, 15))/r.sampling))
        retinasizes = np.array([i for i in retinasizes if i >= 2])
        
        [onmovie, offmovie] = ec2b.onoffFiltering(brightness_movie.data, retinasizes)
        [normalmovie, prostheticmovie] =ec2b.onoffRecombine(onmovie, offmovie)   

        # save the various movies
        filename='Bar_S' + str(sp) + '_O' + str(o) + '_' + retinaname   
        n2sf.savemoviefiles(filename + '_orig', brightness_movie)
        n2sf.savemoviefiles(filename + 'on', onmovie)
        n2sf.savemoviefiles(filename + 'off', offmovie)
        n2sf.savemoviefiles(filename + 'normal', normalmovie) 
        n2sf.savemoviefiles(filename + 'prosthetic', prostheticmovie)
    
