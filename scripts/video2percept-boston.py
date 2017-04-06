import numpy as np
import pickle
import pulse2percept as p2p

# Place an Argus I array on the retina
argus = p2p.implants.ArgusII(x_center=0, y_center=0, h=100)

sim = p2p.Simulation(argus)

# Set parameters of the optic fiber layer (OFL)
# In previous versions of the model, this used to be called the `Retina`
# object, which created a spatial grid and generated the axtron streak map.
sampling = 250       # spatial sampling of the retina (microns)
axon_lambda = 2        # constant that determines fall-off with axonal distance
sim.set_optic_fiber_layer(sampling=sampling, axon_lambda=axon_lambda,
                          x_range=[-2800, 2800], y_range=[-1700, 1700])

# Set parameters of the ganglion cell layer (GCL)
# In previous versions of the model, this used to be called `TemporalModel`.
t_gcl = 0.005 / 1000     # Sampling step (s) for the GCL computation
t_percept = 1.0 / 30.0   # Sampling step (s) for the perceptual output
sim.set_ganglion_cell_layer(tsample=t_gcl)

stim = p2p.stimuli.video2pulsetrain('../../data/boston-train.mp4', argus,
                                    framerate=30, coding='amplitude',
                                    valrange=[0, 60], max_contrast=False,
                                    rftype='square', invert=False)
pickle.dump(stim, open("stim-boston-amp.dat", "wb"))

percept = sim.pulse2percept(stim, t_percept=t_percept, tol=0.25)

pickle.dump(percept, open("percept-boston-square-amp.dat", "wb"))
p2p.files.save_percept("percept-boston-square-amp.avi", percept)
