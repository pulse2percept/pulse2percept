from pulse2percept.plotting import (plot_argus_phosphenes,
                               plot_argus_simulated_phosphenes)
from pulse2percept.implants import ArgusI, ArgusII, AlphaAMS
from pulse2percept.models import (AxonMapModel, AxonMapSpatial,
                                  ScoreboardModel)
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
    # If no implant given, dataframe must have additional columns:
    with pytest.raises(ValueError):
        plot_argus_phosphenes(df, ax=ax)
    df['implant_type_str'] = 'ArgusII'
    df['implant_x'] = 0
    df['implant_y'] = 0
    df['implant_rot'] = 0
    plot_argus_phosphenes(df, ax=ax)


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
