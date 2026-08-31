import numpy as np
import numpy.testing as npt
import pytest

import matplotlib
matplotlib.use('Agg')

import pulse2percept as p2p
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus


CANONICAL = ['play_stimulus_percept', 'plot_argus_phosphenes',
             'plot_argus_simulated_phosphenes', 'plot_stimulus_percept']


@pytest.mark.parametrize('name', CANONICAL)
def test_viz_reexports_plotting(name):
    """The old name wraps the new implementation rather than copying it"""
    npt.assert_equal(getattr(p2p.viz, name).__wrapped__,
                     getattr(p2p.plotting, name))
    # ... and is reachable through the old submodule as well:
    if name.startswith('plot_argus'):
        npt.assert_equal(getattr(p2p.viz.argus, name), getattr(p2p.viz, name))


def test_viz_warns_on_use():
    stim = ImageStimulus(np.random.rand(4, 6))
    percept = Percept(np.random.rand(3, 3, 1))
    # Importing pulse2percept must stay quiet; only calling warns:
    with pytest.warns(DeprecationWarning):
        axes = p2p.viz.plot_stimulus_percept(stim, percept)
    npt.assert_equal([ax.get_title() for ax in axes], ['Stimulus', 'Percept'])
    # The generic statistical helpers go away with the module:
    with pytest.warns(DeprecationWarning):
        p2p.viz.scatter_correlation(np.arange(10), np.arange(10))
