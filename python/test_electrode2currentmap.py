import electrode2currentmap as e2cm
import numpy.testing as npt


def test_Electrode():
    sizex = 5000
    sizey = 2500
    sampling = 25
    e1 = e2cm.Electrode(200, 0, 0, sizex, sizey, sampling=sampling)
    npt.assert_(e1.scale.shape == (sizex//sampling, sizey//sampling))
