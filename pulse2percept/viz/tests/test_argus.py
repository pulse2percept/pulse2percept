import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ArgusI, ArgusII, AlphaAMS
from pulse2percept.viz import plot_argus_phosphenes


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
    plot_argus_phosphenes(df, ArgusI(), ax=ax, axon_map=AxonMapModel())

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
