
import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from utils import TimeSeries
import matplotlib.pyplot as plt

s1 = e2cm.Stimulus(freq=20, pulse_dur=0.075/1000., tsample=0.075/1000.)
s2 = e2cm.Stimulus(freq=15, pulse_dur=0.075/1000., tsample=0.075/1000.)
ea = e2cm.ElectrodeArray([250, 100], [0, -800], [0, 800])
r = e2cm.Retina(axon_map='../axon.npz')
ecm = r.ecm(ea, [s1, s2])
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
