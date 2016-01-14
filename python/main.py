# main-*-
"""
The master function.
These snippets of code will eventually be made into separate functions
Created on Thu Jan 14 09:44:53 2016

@author: Ione Fine
"""


# electrode array parameters
array=utils.Parameters(electroderad=[260], sizex=[2000], sizey=[2000])
 # electroderad = electrode radius in microns
# arraysize= the size of the array in microns

# stimulus parameters

# stim.freq = 20  stimulation freq in Hz
# stim.dur = .5  duration of the entire pulse train in seconds
# stim.pulsedur  .45/1000  # duration of each pulse in the train
# stim.amp = 5   current amplitude on the electrode 30 uA
# stim.tsample  .01/1000 #smallest time unit in ms
# stim.freq=10; # frequency of pulses


# make an image of the current spread
# this will eventually be part of electrode2currentmap
ret=utils.Parameters(sample=25)  # the sampling of the retina in microns
 
 # calculate current spread in the array 
[x, y] = np.meshgrid(np.arange(-array.sizex[0]/2, array.sizex[0]/2, ret.sample), 
np.transpose(np.arange(-array.sizey[0]/2, array.sizey[0]/2, ret.sample)))  #microns
rad = np.sqrt(x**2+y**2)
cspread = np.ones((len(x), len(x)))
# this is not quite the right model for current spread (or at least we should check it 
# based on Arups equation, FIX)
tmp=rad[rad>array.electroderad[0]];
cspread[rad>array.electroderad[0]] = 2/np.pi * (np.arcsin(array.electroderad[0]/tmp))
ret=utils.Parameters(cspread=cspread)
# plot the time course
#plt.figure()
#plt.ylabel('deg')
#plt.xlabel('deg')
#plt.matshow(ret.cspread)# plot the current spread

