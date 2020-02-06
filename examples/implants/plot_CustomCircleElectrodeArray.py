"""
============================================================================
Running the Custom Electrode Array Circle
============================================================================

1. Creating the CustomCircleElectrodeArray class (inherited from ElectrodeArray)
------------------------------------------------------------------------------

"""


import pulse2percept as p2p
import collections as coll
import math
from pulse2percept.implants import ElectrodeArray, DiskElectrode, ProsthesisSystem
pi = math.pi

##############################################################################
# CustomCircleElectrodeArray is inherited from ElectrodeArray class and it can 
# use all the public methods of EelctrodeArray
# "CustomCircleElectrodeArray" is the name of the class
class CustomCircleElectrodeArray(ElectrodeArray): 

# "__init__" is constructor of CustomCircleElectrodeArray class and it takes 
# three parameters radius, x_axis, y_axis which represent the radius of the 
# CustomCircleElectrodeArray, the center of the circle on x_axis and y_axis
# respectively
# Once the class is instantiated, 10 electrodes will produced automatically
# on the circumstance of the electrode array
    def __init__(self, radius, x_axis, y_axis):
        self.electrodes = coll.OrderedDict()
        # generate 10 electrodes on the circumstance of the circle
        for x in range(0,10):
            self.add_electrode('A'+ str(x), DiskElectrode(x_axis + math.cos(2*pi/10*x)*radius, y_axis + math.sin(2*pi/10*x)*radius, 0, 100))

# "remove" function can be called to delete an electrode from the electrode
# array by entering the name of the electrode
    def remove(self, name):
        del self.electrodes[name]
    
    

##############################################################################
# A Electrode Array
# --------------------
# We can specify the arguments of the customized circle electrode array as follows:

radius = 1000 # radius of the electrode array circle 1000 in um
centered_x = 0 # center of the circle's x axis is at 0 in um
centered_y = 0 # center of the circle's y axis is at 0 in um

# By calling the constructor of the CustomCircleElectrodeArray class that inherited from 
# ElectrodeArray class We can initialize a customized electrode array circle by entering 
# these three parameters radius, centered_x, and centered_y in order
earray = CustomCircleElectrodeArray(radius, centered_x, centered_y)

##############################################################################
# By calling ProsthesisSystem with a ``earray`` source (we just created), 
# we can generate an implant
implant = ProsthesisSystem(earray)

##############################################################################
# We can visualize the generated electrode array built on the axon map by using
# the matplot (show the initialized custom electrode array circle)

import matplotlib.pyplot as plt
%matplotlib inline
p2p.viz.plot_implant_on_axon_map(implant)

##############################################################################
# demonstration of removing one of the electrode from the electrode array

removeElectrode = 'A1' # declaring a variable and assigning it to one of the electrode 
                       # name in order to delete the electrode

# use the earray object to call the remove function to remove 'A1'
earray.remove(removeElectrode) 

# recreate the implant with the new electrode array
implant = ProsthesisSystem(earray)

# replot the implant on the axon map after one electrode is removed
p2p.viz.plot_implant_on_axon_map(implant)