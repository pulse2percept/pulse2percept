# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in 
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
#from __future__ import print_function
#import numpy as np
from scipy.misc import factorial
import numpy as np
import matplotlib.pyplot as plt
import utils

# this creates a waveform. Eventually this will be replaced by the output of current2effective current
dur=.5
tsample=.01/1000;

stim = utils.Parameters(freq=20, dur=.5,pulsedur=.075/1000, 
                        amp=20, tsample=.075/1000, amplitude=10) 
stim.t =np.arange(0, stim.dur, stim.tsample)# a vector of time
sawtooth=stim.freq * np.mod(stim.t, 1/stim.freq)
on=np.logical_and(sawtooth>(stim.pulsedur*stim.freq), 
                  sawtooth< (2 * stim.pulsedur * stim.freq))
on=on.astype(float);
off=sawtooth<stim.pulsedur * stim.freq
stim.tsform =stim.amplitude *( on.astype(float)-off.astype(float))

def gamma_gmb(n,tau,t):
    """y=Gamma(n,theta,t)
	returns a gamma function from [0:t];
	y=(t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1));
	which is the result of an n stage leaky integrator.
	6/27/95 gmb """

    flag=0

    if t[0]==0: 
        t=t[2:len(t)]
        flag=1
            
    y = ((tau * (1/t))**(1-n)*np.exp(-(1/(tau*(1/t)))))/(tau*np.ones(len(t))*factorial(n-1))
        
    if flag==1:
        y=np.concatenate([[0], y])
        
    return y
    

# model parameters
p=utils.Parameters(tau1 = .42/1000, tau2 = 45.25/1000, tau3 =  26.25/1000, 
                   e = 2.25, beta = .6
                   ,asymptote=14,   slope=.3, shift=47)
# p.tau1 = .42/1000  # fast leaky integrator, from Alan model, tends to be between .24 - .65
# p.tau2 = 45.25/1000   integrator for charge accumulation, has values between 38-57
# p.e = scaling factor for the effects of charge accumulation 2-3 for threshold or 8-10 for suprathreshold
# p.tau3 =  26.25/1000 # 24-33 FIX

# parameters for a stationary nonlinearity providing a continuous functino
# that nonlinearly rescales the response 
#based on Nanduri et al 2012, equation 3
#p.asymptote = 14 
# p.slope =.3
# p.shift =47 

# create all the leaky integrators we are going to use
# implement a fast leaky integrator
t=np.arange(0,20*p.tau1, stim.tsample)
G1=gamma_gmb(1, p.tau1, t) #% fast impulse response function


# implement a slightly slower leaky integrator, which is used to model adapatation due to charge accumulation over time
t = np.arange(0, 8*p.tau2,stim.tsample) # this could be shrunk slightly if need be
G2 = gamma_gmb(1, p.tau2, t)

# implement a final slow leaky integrator for the brain
t= np.arange(0, p.tau3*8, stim.tsample) # this is cropped as tightly as possible for speed sake
G3 = gamma_gmb(3,p.tau3,t)

# implement the nonlinear model
# charge integration in the cell
R1 = stim.tsample*np.convolve(G1,stim.tsform)

# use a rectified version of the stimulation time course
# to calculate charge accumulation, and subtract accumulated charge
rect_tsform=stim.tsform;
ind=stim.tsform<0
rect_tsform[ind]=0;
ca=stim.tsample *np.cumsum(rect_tsform.astype(float))
chargeaccumulated=p.e * stim.tsample*np.convolve(G2, ca)
R1=np.concatenate([R1, np.zeros(len(chargeaccumulated)-len(R1))])
R2=R1-chargeaccumulated
ind=R2<0
R2[ind]=0 # rectify again

# now we put in the stationary nonlinearity of Devyanis
R2norm=R2 / R2.max() # normalize
scFac=p.asymptote/(1+np.exp(-(R2/p.slope)+p.shift))
R3=R2norm*scFac #scaling factor varies with original amplitude

R4=stim.tsample* np.convolve(G3, R3)

R4 =stim.tsample*np.convolve(G3,R3)
