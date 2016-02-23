import numpy as np
import numpy.testing as npt

import electrode2currentmap as e2cm

def test_Electrode():
    sizex = 5000
    sizey = 2500
    sampling = 25
    retina = e2cm.Retina(sizex=sizex, sizey=sizey, sampling=sampling)
    e1 = e2cm.Electrode(retina, 200, 0, 0)
    npt.assert_(e1.scale.shape == (sizex // retina.sampling_deg,
                                   sizey // retina.sampling_deg))


def test_Movie2Pulsetrain():
    rflum = np.zeros(10, 10, 100)
    rflum[:, :, 50] = 1
    m2pt = Movie2Pulsetrain()
