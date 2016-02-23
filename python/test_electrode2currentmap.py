import tempfile
import numpy as np
import numpy.testing as npt

import electrode2currentmap as e2cm


def test_Retina():
    retina_file = tempfile.NamedTemporaryFile().name
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, axon_map=retina_file)
    npt.assert_equal(retina.gridx.shape, ((yhi - ylo) / sampling,
                                          (xhi - xlo) / sampling))


def test_Movie2Pulsetrain():
    fps = 30.0
    amplitude_transform = 'linear'
    amp_max = 90
    freq = 20
    pulse_dur = .075/1000.
    interphase_dur = .075/1000.
    tsample = .005/1000.
    pulsetype = 'cathodicfirst'
    stimtype = 'pulsetrain'
    dtype = np.int8
    rflum = np.zeros((10, 10, 100))
    rflum[:, :, 50] = 1
    m2pt = e2cm.Movie2Pulsetrain(rflum)
    npt.assert_equal(m2pt.shape[:2], (10, 10))
