import numpy.testing as npt
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pulse2percept.implants.cortex import EllipsoidElectrode, LinearEdgeThread, NeuralinkThread, Neuralink

def test_EllipsoidElectrode():
    electrode = EllipsoidElectrode(0, 1, 2, 3, 4, 5, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.rx, 3)
    npt.assert_almost_equal(electrode.ry, 4)
    npt.assert_almost_equal(electrode.rz, 5)
    npt.assert_equal(electrode.name, 'A001')
    with pytest.raises(TypeError):
        EllipsoidElectrode([0], 1, 2)
    with pytest.raises(TypeError):
        EllipsoidElectrode(0, np.array([1, 2]), 2)
    with pytest.raises(TypeError):
        EllipsoidElectrode(0, 1, [2, 3])
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)


def test_LinearEdgeThread():
    thread = LinearEdgeThread()
    npt.assert_almost_equal(thread.x, 0)
    npt.assert_almost_equal(thread.y, 0)
    npt.assert_almost_equal(thread.z, 0)

    # elecs aren't actually at this spot, but are on the edge, a few microns off
    zs = []
    for e in thread.electrode_objects:
        npt.assert_almost_equal(e.x, thread.r + 7 // 2)
        npt.assert_almost_equal(e.y, 0)
        npt.assert_almost_equal(e.rot, thread.rot)
        zs.append(e.z)
    npt.assert_equal(np.allclose(np.diff(zs), thread.spacing), True)


    thread = LinearEdgeThread(orient=[1, 0, 0])
    xs = []
    for e in thread.electrode_objects:
        npt.assert_almost_equal(e.z, -thread.r - 7 // 2)
        npt.assert_almost_equal(e.y, 0)
        xs.append(e.x)
    npt.assert_equal(np.allclose(np.diff(xs), thread.spacing), True)

    thread = LinearEdgeThread(orient=[1, 1, 1], spacing=3*np.sqrt(3))
    locs = []
    for i, e in enumerate(thread.electrode_objects):
        npt.assert_almost_equal(e.x, 3*i + 4.618802, decimal=5)
        npt.assert_almost_equal(e.y, 3*i + 4.618802, decimal=5)
        npt.assert_almost_equal(e.z, 3*i - 4.618802, decimal=5)
        locs.append([e.x, e.y, e.z])
    npt.assert_equal(np.allclose(np.diff(locs, axis=0), 3), True)



def test_Neuralink():
    t1 = LinearEdgeThread(orient=[1, 0, 0])
    t2 = LinearEdgeThread(500, 500, orient=[0, 1, 0])
    nlink = Neuralink([t1, t2])

    # check that positions are the same
    npt.assert_equal(nlink['0-1'].x, t1['1'].x)
    npt.assert_equal(nlink['0-1'].y, t1['1'].y)
    npt.assert_equal(nlink['1-1'].x, t2['1'].x)
    npt.assert_equal(nlink['1-1'].y, t2['1'].y)

    


