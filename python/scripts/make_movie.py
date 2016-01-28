
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from utils import TimeSeries
import matplotlib.pyplot as plt


s1 = e2cm.Stimulus(freq=20, dur=0.5, pulse_dur=.075/1000.,interphase_dur=.075/1000., delay=0.,
                 tsample=.075/1000., current_amplitude=20, 
                 current=None, pulsetype='cathodicfirst')
                 
s2 = e2cm.Stimulus(freq=20, dur=0.5, pulse_dur=.075/1000.,interphase_dur=.075/1000., delay=9/1000,
                 tsample=.075/1000., current_amplitude=20, 
                 current=None, pulsetype='cathodicfirst')
                 
                 
plt.plot(s1.data)                
ea1 = e2cm.ElectrodeArray([250], [0], [0])
ea2 = e2cm.ElectrodeArray([100, 100], [-200, 200], [200, -200])

r = e2cm.Retina(axon_map='../axon.npz')
ecm = r.ecm(ea2, [s1, s2])
tm1 = ec2b.TemporalModel()
#sr = np.zeros(ecm.shape)
#xx = yy = 0
fr = tm1.fast_response(ecm)
ca = tm1.charge_accumulation(fr, ecm)
sn = tm1.stationary_nonlinearity(ca)
sr = tm1.slow_response(sn).data

# sr = TimeSeries(sn.tsample, sr)
# sr.resample(25)
# for i in range(sr.data.shape[-1]):
#     fig, ax = plt.subplots(1)
#     ax.matshow(sr.data[:, :, i], cmap='viridis', vmax=sr.data.max(),
#                vmin=sr.data.min())
#     fig.savefig('/Users/arokem/tmp/figures-tmp/fig%03d' % i)
#     plt.close("all")
