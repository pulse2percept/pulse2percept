import numpy.testing as npt
from pulse2percept.implants import ArgusI
from pulse2percept.models import Horsager2009Model


def test_Horsager2009Model():
    model = Horsager2009Model(xystep=5)
    model.build()
    npt.assert_equal(model.has_time, True)
    npt.assert_equal(model.predict_percept(ArgusI(stim=None)), None)
