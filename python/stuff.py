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

#to refresh changes:
#import importlib
#importlib.reload(oyster)

#  Non-default Jasonius model parameters
nCells = 500      #  Total number of ganglion cells (8800)
nR = 801          #  Number of steps for each ganglion cell
maxR = 45         # max eccentricity
r0 = 4            #  minumum radius (optic disc size)

# Default model parameters:
center = [15,2]   #  p.center of optic disc
rot = 0*np.pi/180    #  angle of rotation (clockwise)
scale = 1         #  scale factor
bs = -1.9         #  superior 'b' parameter constant
bi = .5           #  inferior 'c' parameter constant

axon_lambda = 2     # space constant for axonal streaks
                    # we think it's somewhere between 1 and 3.4 degrees

ang0 = np.hstack([np.linspace(60,180,nCells/2),
                np.linspace(-180,60,nCells/2)]);
                                 
r = np.linspace(r0,maxR,nR);

xa,ya = oyster.jansonius(ang0,r)

# Crop into a 30 x 30 deg disc
#id = x**2+y**2 > cropRad**2
#x[id] = np.NaN
#y[id] = np.NaN    

# plot it
plt.plot(xa,ya)
plt.axes().set_aspect('equal', 'datalim')
plt.show()

# generate xg,yg - the grid of pixel locations for the current field
xlo = -4
xhi = 4
ylo = -4
yhi = 4
n = (101,101)
xg,yg =  np.meshgrid(np.linspace(xlo,xhi,n[0]),np.linspace(ylo,yhi,n[1]))

axon_id, axon_weight = oyster.makeAxonStreaks(xg,yg,xa,ya, axon_lambda)

     
# made-up effective current field

current_field = np.exp(-((xg-1)**2+(yg+3)**2)/1)
current_field[60:65,20:25]= 1

current_field[50:60,80:90]= 1
current_field = current_field + np.exp(-((xg-2)**2+(yg-3)**2)/1)


plt.subplot(1,2,1)
plt.imshow(current_field,extent = (xlo,xhi,ylo,yhi),cmap = 'viridis')
# pass the current field through the oyster filter

out = np.zeros(n)
for id in range(0,n[0]*n[1]-1):
    out.flat[id]= np.dot(current_field.flat[axon_id[id]],axon_weight[id])    


plt.subplot(1,2,2)
plt.imshow(out,extent = (xlo,xhi,ylo,yhi),cmap = 'viridis')