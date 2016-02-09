# -*- coding: utf-8 -*-
"""
testSparseConv.py

Created on Fri Jan 22 00:20:17 2016

@author: Geoff Boynton
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import utils
import time

# time vector for stimulus (long)
maxT = .5; # seconds
nt = 100000;
t = np.linspace(0,maxT,nt)

# stimulus (10 Hz anondic and cathodic pulse train)
stim = np.zeros(nt)
stim[0:nt:10000] = 1
stim[100:nt:1000] = -1

plt.subplot(2,2,1)
plt.plot(t,stim)

# time vector for impulse response (shorter)
tt = t[t<.1]
ntt = len(tt)

# impulse reponse (kernel)
G = np.exp(-tt/.005)

plt.subplot(2,2,2)
plt.plot(tt,G)

# np.convolve
start_time = time.time()
outConv = np.convolve(stim,G)
outConv = outConv[0:len(t)]
tconv = time.time()-start_time
print("np.conv: elapsed time was %g seconds" % tconv)

plt.subplot(2,2,3)
plt.plot(t,outConv)

# utils.sparseconv
start_time = time.time()
outSparseconv = utils.sparseconv(G,stim)
outSparseconv = outSparseconv[0:len(t)]
tsparseconv = time.time()-start_time
print("sparseconv: elapsed time was %g seconds" %tsparseconv)

plt.subplot(2,2,4)

plt.plot(t,outSparseconv)

ratio = tconv/tsparseconv
print("Time ratio: %5.2f" %ratio)

# compare outputs - should be same within machine tolerance
np.max(abs(outConv-outSparseconv))
