# stuff.py
#
# Useful things:
#
# import oyster
# run stuff
# import importlib
# importlib.reload(oyster)

import oyster
import numpy as np
import matplotlib.pyplot as plt
import importlib

#importlib.reload(oyster)

#  Non-default Jasonius model parameters
nCells = 100      #  Total number of ganglion cells (8800)
nR = 801          #  Number of steps for each ganglion cell
maxR = 45         # max eccentricity
r0 = 4            #  minumum radius (optic disc size)

# Default model parameters:
center = [15,2]   #  p.center of optic disc
rot = 0*np.pi/180    #  angle of rotation (clockwise)
scale = 1         #  scale factor
bs = -1.9         #  superior 'b' parameter constant
bi = .5           #  inferior 'c' parameter constant

axon_lambda = 2      # best estimates are between 1 and 3.4 degrees


ang0 = np.hstack([np.linspace(60,180,nCells/2),
                np.linspace(-180,60,nCells/2)]);
                                 
r = np.linspace(r0,maxR,nR);

xa,ya = oyster.jansonius(ang0,r)

# plot it
plt.plot(xa,ya)
plt.axes().set_aspect('equal', 'datalim')
plt.show()

# grid of pixels
xlo = -4
xhi = 4
ylo = -4
yhi = 4
n = (101,101)


xg,yg =  np.meshgrid(np.linspace(xlo,xhi,n[0]),np.linspace(ylo,yhi,n[1]))
indices = np.indices(xg.shape)
# double loop through pixels

axonListId = ()
axon_xg = ()
axon_yg = ()
axon_dist = ()
axon_weight = ()
axon_id = ()

min_weight = .001

plt.figure(figsize = (10,10))
for id in range(0,n[0]*n[1]-1):
    
    #find the nearest axon to this pixel
    d = (xa-xg.flat[id])**2+ (ya-yg.flat[id])**2
    
    cur_ax_id = np.nanargmin(d)    
    [axPosId0,axNum] = np.unravel_index(cur_ax_id,d.shape)
    
    dist = 0
    cur_xg = xg.flat[id]
    cur_yg = yg.flat[id]
           
    axon_dist = axon_dist + ([0],) 
    axon_weight = axon_weight + ([1],)
    axon_xg = axon_xg + ([cur_xg],)
    axon_yg = axon_yg + ([cur_yg],)
    axon_id = axon_id + ([id],)
    
    # plt.plot(xa[:axPosId0,axNum],ya[:axPosId0,axNum],'.-')    
    
    #now go back toward the optic disc         
    for axPosId in range(axPosId0-1,-1,-1):
        dist = dist + np.sqrt((xa[axPosId+1,axNum]-xa[axPosId,axNum])**2
        + (ya[axPosId+1,axNum]-ya[axPosId,axNum])**2)
        
        weight = np.exp(-dist/axon_lambda)

        nearest_xg_id = np.abs(xg[0,:]-xa[axPosId,axNum]).argmin()
        nearest_yg_id = np.abs(yg[:,0]-ya[axPosId,axNum]).argmin()
        nearest_xg =xg[0,nearest_xg_id]
        nearest_yg =yg[nearest_yg_id,0]
        
        

        if nearest_xg != cur_xg or nearest_yg != cur_yg and weight>min_weight:
            cur_xg = nearest_xg
            cur_yg = nearest_yg

            axon_dist[id].append(dist)
            axon_weight[id].append(np.exp(-dist/axon_lambda))
            axon_xg[id].append(cur_xg)
            axon_yg[id].append(cur_yg)
            axon_id[id].append(np.ravel_multi_index((nearest_yg_id,nearest_xg_id),xg.shape)) 
    plt.plot(axon_xg[id],axon_yg[id])
            

id = 10

plt.plot(axon_xg[id],axon_yg[id])
# is the same as
plt.plot(xg.flat[axon_id[id]],yg.flat[axon_id[id]])

# made-up effective current field

#current_field = np.exp(-((xg-1)**2+yg**2)/1)

current_field = np.zeros(n)
current_field[49:51,49:51] = 1

plt.imshow(current_field,extent = (xlo,xhi,ylo,yhi))
# pass the current field through the oyster filter

out = np.zeros(n)
for id in range(0,n[0]*n[1]-1):
    out.flat[id]= np.dot(current_field.flat[axon_id[id]],axon_weight[id])    


plt.imshow(out,extent = (xlo,xhi,ylo,yhi))