import pandas as pd
import numpy.testing as npt

from pulse2percept.datasets import load_nanduri2012


def test_load_nanduri2012():
    data = load_nanduri2012(shuffle=False)

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    columns = ['subject', 'implant', 'electrode', 'task', 'stim_class',
               'freq', 'amp_factor', 'ref_stim_class', 'ref_amp_factor',
               'ref_freq', 'brightness', 'size', 'pulse_dur',
               'interphase_dur', 'pulse_type', 'varied_param']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (128, 17))
    npt.assert_equal(data.subject.unique(), ['S06'])

    # Shuffle dataset (index will always be range(552), but rows are shuffled):
    data = load_nanduri2012(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'subject'], 'S06')
    npt.assert_equal(data.loc[0, 'electrode'], 'B1')
    npt.assert_equal(data.loc[94, 'subject'], 'S06')
    npt.assert_equal(data.loc[94, 'electrode'], 'A4')

    # Select electrodes:
    data = load_nanduri2012(electrodes='A2')
    npt.assert_equal(data.shape, (16, 17))
    npt.assert_equal(data.electrode.unique(), 'A2')
    npt.assert_equal(data.subject.unique(), 'S06')
    data = load_nanduri2012(electrodes=['A1', 'A9'])  # 'A9' doesn't exist
    npt.assert_equal(data.shape, (0, 17))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), 'S06')

    # Select task
    data = load_nanduri2012(task='rate')
    npt.assert_equal(data.shape, (88, 17))
    npt.assert_equal(data.task.unique(), 'rate')
    data = load_nanduri2012(task='size')
    npt.assert_equal(data.shape, (40, 17))
    npt.assert_equal(data.task.unique(), 'size')
