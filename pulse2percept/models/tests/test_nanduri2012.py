import numpy.testing as npt
from pulse2percept.implants import ArgusII
from pulse2percept.models import Nanduri2012Model


def test_Nanduri2012Model():
    model = Nanduri2012Model(xystep=5)
    model.build()
    npt.assert_equal(model.has_time, True)
    npt.assert_equal(model.predict_percept(ArgusII(stim=None)), None)
