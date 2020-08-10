import pandas as pd
import numpy.testing as npt
import pytest
from unittest import mock
from importlib import reload

from pulse2percept.datasets import load_nanduri2012


def test_load_nanduri2012():
    data = load_nanduri2012(shuffle=False)

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    columns = ['subject', 'implant', 'electrode', 'task', 'stim_type',
               'stim_dur', 'stim_freq', 'stim_amp_factor', 'brightness', 'pulse_dur', 'pulse_type',
               'interphase_dur', 'delay_dur', 'source']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (95, 14))
    npt.assert_equal(data.subject.unique(), ['S06'])

    # Shuffle dataset (index will always be range(552), but rows are shuffled):
    data = load_nanduri2012(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'subject'], 'S06')
    npt.assert_equal(data.loc[0, 'electrode'], 'C1')
    npt.assert_equal(data.loc[0, 'stim_type'], 'fixed_duration')
    npt.assert_equal(data.loc[94, 'subject'], 'S06')
    npt.assert_equal(data.loc[94, 'electrode'], 'A2')
    npt.assert_equal(data.loc[94, 'stim_type'], 'fixed_duration')

    # Select electrodes:
    data = load_nanduri2012(electrodes='A1')
    npt.assert_equal(data.shape, (0, 14))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), 'S06')
    data = load_nanduri2012(electrodes=['A1', 'A9'])  # 'A9' doesn't exist
    npt.assert_equal(data.shape, (0, 14))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), 'S06')
