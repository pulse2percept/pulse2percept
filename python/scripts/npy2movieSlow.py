# -*- npy2movieIone.py -*-
"""
Created on Tue Feb 23 14:55:00 2016

@author: Ariel Rokem
"""

import os
import os.path as op
import sys
import tempfile 
import matplotlib.animation as animation
import numpy as np
from pylab import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


fname = 'SE_retina_1700x2900L80_E46' 
loadname=fname + '.npy'
arr = np.load(loadname)

def ani_frame():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(arr[:,:, 0],cmap='gray',interpolation='nearest')
    tight_layout


def update_img(mov, n):
        tmp = arr[:,:, n]
        im.set_data(tmp)
        return im
        
ani = animation.FuncAnimation(fig,update_img,arr, arr.shape[2],interval=30)
writer = animation.writers['ffmpeg'](fps=30)

ani.save('demo.mp4',writer=writer,dpi=dpi)
return ani