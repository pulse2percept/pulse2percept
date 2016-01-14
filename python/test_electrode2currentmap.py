import numpy as np
import numpy.testing as npt

import electrode2currentmap as e2cm

def test_Electrode():
    sizex = 5000
    sizey = 2500
    sampling = 25
    e1 = e2cm.Electrode(200, 0, 0, sizex, sizey, sampling=sampling)
    npt.assert_(e1.scale.shape == (sizex//sampling, sizey//sampling))


def test_ElectrodeGrid():
    sizex = 5000
    sizey = 2500
    sampling = 25
    eg1 = e2cm.ElectrodeGrid([200, 400])
