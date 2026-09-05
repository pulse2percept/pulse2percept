from pulse2percept.plotting import (plot_argus_phosphenes,
                               plot_argus_simulated_phosphenes)
from pulse2percept.implants import ArgusI, ArgusII, AlphaAMS
from pulse2percept.models import (AxonMapModel, AxonMapSpatial,
                                  ScoreboardModel)
from pulse2percept.units import deg, um
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest
import matplotlib
matplotlib.use('Agg')

# Building an axon map writes a cache to a relative path; keep it in a
# temporary directory instead of wherever pytest was started from:
pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')


def test_plot_argus_phosphenes():
    df = pd.DataFrame([
        {'subject': 'S1', 'electrode': 'A1', 'image': np.random.rand(10, 10),
         'xrange': (-10, 10), 'yrange': (-10, 10)},
        {'subject': 'S1', 'electrode': 'B2', 'image': np.random.rand(10, 10),
         'xrange': (-10, 10), 'yrange': (-10, 10)},
    ])
    _, ax = plt.subplots()
    plot_argus_phosphenes(df, ArgusI(), ax=ax)
    plot_argus_phosphenes(df, ArgusII(), ax=ax)

    # Add axon map:
    _, ax = plt.subplots()
    plot_argus_phosphenes(df, ArgusI(), ax=ax, axon_map=AxonMapModel(ArgusI()))

    # Data must be a DataFrame:
    with pytest.raises(TypeError):
        plot_argus_phosphenes(np.ones(10), ArgusI())
    # DataFrame must have the required columns:
    with pytest.raises(ValueError):
        plot_argus_phosphenes(pd.DataFrame(), ArgusI())
    # Subjects must all be the same:
    with pytest.raises(ValueError):
        dff = pd.DataFrame([{'subject': 'S1'}, {'subject': 'S2'}])
        plot_argus_phosphenes(dff, ArgusI())
    # Works only for Argus:
    with pytest.raises(TypeError):
        plot_argus_phosphenes(df, AlphaAMS())
    # Works only for axon maps:
    with pytest.raises(TypeError):
        plot_argus_phosphenes(df, ArgusI(), ax=ax,
                              axon_map=ScoreboardModel(ArgusI()))
    # Manual subject selection
    plot_argus_phosphenes(df[df.electrode == 'B2'], ArgusI(), ax=ax)
    # If no implant given, the dataframe must name the device:
    with pytest.raises(ValueError):
        plot_argus_phosphenes(df, ax=ax)
    # ...and it must name a device that exists:
    df['implant_type_str'] = 'Arguz'
    with pytest.raises(ValueError):
        plot_argus_phosphenes(df, ax=ax)
    df['implant_type_str'] = 'ArgusII'
    # That column alone is enough; the placement ones are optional:
    plot_argus_phosphenes(df, ax=ax)
    df['implant_x'] = 0
    df['implant_y'] = 0
    df['implant_rot'] = 0
    plot_argus_phosphenes(df, ax=ax)


def test_argus_placement_never_moves_the_implant():
    """Placement is read from the data or the arguments, not written back"""
    from pulse2percept.plotting.argus import _argus_pose, _placed_electrodes
    df = pd.DataFrame([
        {'subject': 'S1', 'electrode': 'A1', 'image': np.random.rand(10, 10),
         'xrange': (-10, 10), 'yrange': (-10, 10),
         'implant_type_str': 'ArgusII', 'implant_x': -1331,
         'implant_y': -850, 'implant_rot': -28.4},
    ])
    argus = ArgusII()
    local = argus.electrode_array.coordinates()
    _, ax = plt.subplots()
    plot_argus_phosphenes(df, argus, ax=ax)
    npt.assert_array_equal(argus.electrode_array.coordinates(), local)

    # The dataset columns are what got used:
    xy, rot = _argus_pose(df, None, None)
    npt.assert_almost_equal(xy, (-1331, -850))
    npt.assert_almost_equal(rot, -28.4)
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    placed = _placed_electrodes(argus, xy, rot)
    npt.assert_almost_equal(np.array(list(placed.values())),
                            (R @ local[:, :2].T).T + xy)

    # Explicit arguments win over them, in model units:
    xy, rot = _argus_pose(df, (100, 200) * um, 10 * deg)
    npt.assert_almost_equal(xy, (100, 200))
    npt.assert_almost_equal(rot, 10)
    # ...and without either, the array is drawn about the fovea:
    npt.assert_equal(_argus_pose(df.drop(columns=['implant_x', 'implant_y',
                                                  'implant_rot']),
                                 None, None), ((0.0, 0.0), 0.0))
    # Position and rotation fall back independently:
    xy, rot = _argus_pose(df.drop(columns=['implant_rot']), None, None)
    npt.assert_almost_equal(xy, (-1331, -850))
    npt.assert_almost_equal(rot, 0.0)
    xy, rot = _argus_pose(df.drop(columns=['implant_x', 'implant_y']),
                          None, None)
    npt.assert_almost_equal(xy, (0.0, 0.0))
    npt.assert_almost_equal(rot, -28.4)


def test_argus_plot_does_not_flip_the_electrode_constants():
    """A left eye flips a copy, so plots do not depend on call history"""
    from pulse2percept.plotting import argus as argus_mod
    px1 = argus_mod.PX_ARGUS1.copy()
    px2 = argus_mod.PX_ARGUS2.copy()
    df = pd.DataFrame([
        {'subject': 'S1', 'electrode': 'A1', 'image': np.random.rand(10, 10),
         'xrange': (-10, 10), 'yrange': (-10, 10)},
    ])
    _, ax = plt.subplots()
    for implant in (ArgusI(eye='LE'), ArgusII(eye='LE'), ArgusII(eye='LE')):
        plot_argus_phosphenes(df, implant, ax=ax)
        npt.assert_array_equal(argus_mod.PX_ARGUS1, px1)
        npt.assert_array_equal(argus_mod.PX_ARGUS2, px2)


# Parametrize over the class, not over instances: arguments to `parametrize`
# are built at import time and shared across invocations, so a test that
# mutated one would leak state into the others.
@pytest.mark.parametrize('ImplantType', (ArgusI, ArgusII))
def test_plot_argus_simulated_phosphenes(ImplantType):
    implant = ImplantType()
    source = {'A1': [1, 0, 0], 'B2': [0, 1, 0], 'C3': [0, 0, 1]}
    model = ScoreboardModel(implant=implant).build()
    percepts = model.predict_percept(source)

    plot_argus_simulated_phosphenes(percepts, implant)

    # Add axon map:
    _, ax = plt.subplots()
    plot_argus_simulated_phosphenes(percepts, implant, ax=ax,
                                    axon_map=AxonMapModel(implant))


def test_the_plotted_implant_owns_the_axon_laterality(monkeypatch):
    """One implant in the picture, so one eye -- the caller's `argus`

    Asserted on where the bundles are grown from rather than on the drawn
    lines: this plot windows bundles to the array's own extent, and nothing
    survives that in a synthetic dataset.
    """
    df = pd.DataFrame([
        {'subject': 'S1', 'electrode': 'A1', 'image': np.random.rand(10, 10),
         'xrange': (-10, 10), 'yrange': (-10, 10)},
    ])
    grown = []
    unwrapped = AxonMapSpatial.grow_axon_bundles

    def spy(self, **kwargs):
        grown.append((self.implant.eye, tuple(self.loc_od)))
        return unwrapped(self, **kwargs)

    monkeypatch.setattr(AxonMapSpatial, 'grow_axon_bundles', spy)
    # A model bound to the other eye, which used to be what decided:
    axon_map = AxonMapModel(implant=ArgusII(eye='RE'), loc_od=(15.5, 1.5))
    _, ax = plt.subplots()
    plot_argus_phosphenes(df, ArgusII(eye='LE'), ax=ax, axon_map=axon_map)
    # The optic disc is nasal, so a left eye puts it at negative x:
    npt.assert_equal(grown, [('LE', (-15.5, 1.5))])
    # ... and the caller's model is left pointed where it was:
    npt.assert_equal(axon_map.implant.eye, 'RE')
    npt.assert_equal(tuple(axon_map.spatial.loc_od), (15.5, 1.5))
