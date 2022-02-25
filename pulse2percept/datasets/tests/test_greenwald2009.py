import pandas as pd
import numpy.testing as npt

from pulse2percept.datasets import load_greenwald2009


def test_load_greenwald2009():
    data = load_greenwald2009()

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    expected_columns = ['subject', 'implant', 'electrode', 'task', 'stim_class',
               'stim_dur', 'amp', 'brightness', 'pulse_dur', 'interphase_dur',
               'pulse_type', 'threshold']

    for expected_col in expected_columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (83, 12))
    npt.assert_equal(data.subject.unique(), ['S05', 'S06'])

    # Shuffle dataset
    data = load_greenwald2009(shuffle=True, random_state=5)
    npt.assert_equal(data.loc[0, 'subject'], 'S05')
    npt.assert_equal(data.loc[0, 'electrode'], 'C3')
    npt.assert_equal(data.loc[21, 'subject'], 'S06')
    npt.assert_equal(data.loc[21, 'electrode'], 'C4')

    # Select electrodes
    data = load_greenwald2009(electrodes=['B2', 'C3'])
    npt.assert_equal(data.shape, (56, 12))
    npt.assert_equal(data.electrode.unique(), ['B2', 'C3'])

    # Load electrode that doesn't exist
    data = load_greenwald2009(electrodes=['C4', 'Z1'])
    npt.assert_equal(data.shape, (27, 12))
    npt.assert_equal(data.electrode.unique(), 'C4')

    # Select by subject
    data = load_greenwald2009(subjects='S05')
    npt.assert_equal(data.shape, (48, 12))
    npt.assert_equal(data.subject.unique(), 'S05')

    # Select by subject with non-existent subject
    data = load_greenwald2009(subjects=['S06', 'X1'])
    npt.assert_equal(data.shape, (35, 12))
    npt.assert_equal(data.subject.unique(), 'S06')
