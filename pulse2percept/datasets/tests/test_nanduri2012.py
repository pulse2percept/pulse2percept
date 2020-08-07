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
               'interphase_dur', 'delay_dur', 'source', 'test']
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

    # Select subjects: *Pointless with one subject*
    data = load_nanduri2012(subjects='S06')
    npt.assert_equal(data.shape, (95, 14))
    npt.assert_equal(data.subject.unique(), 'S06')
    data = load_nanduri2012(subjects=['S06', 'S07'])  # 'S07' doesnt' exist
    npt.assert_equal(data.shape, (95, 14))
    npt.assert_equal(data.subject.unique(), 'S06')
    # data = load_nanduri2012(subjects=['S05', 'S06'])  # same as None
    # npt.assert_equal(data.shape, (552, 21))   -- Only one subject exists, can't test for two subjects that exist
    data = load_nanduri2012(subjects='S6')  # 'S6' doesn't exist
    npt.assert_equal(data.shape, (0, 14))
    npt.assert_equal(data.subject.unique(), [])

    # Select electrodes:
    data = load_nanduri2012(electrodes='A1')
    npt.assert_equal(data.shape, (0, 14))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), 'S06')
    data = load_nanduri2012(electrodes=['A1', 'A9'])  # 'A9' doesn't exist
    npt.assert_equal(data.shape, (0, 14))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), 'S06')

    # Select stimulus types:
    #data = load_nanduri2012(stim_types='single_pulse')
    #npt.assert_equal(data.shape, (0, 14))
    #npt.assert_equal(data.stim_type.unique(), 'single_pulse')
    #npt.assert_equal(list(data.subject.unique()), ['S05', 'S06'])
    data = load_nanduri2012(stim_types='fixed_duration')
    npt.assert_equal(data.shape, (95, 14))
    npt.assert_equal(list(data.stim_type.unique()),
                      ['fixed_duration'])
    npt.assert_equal(list(data.subject.unique()), ['S06'])

    # Subject + electrode + stim type:
    data = load_nanduri2012(subjects='S06', electrodes=['A2', 'C3'],
                             stim_types='fixed_duration')
    npt.assert_equal(data.shape, (12, 14))
    npt.assert_equal(data.subject.unique(), 'S06')
    npt.assert_equal(list(data.electrode.unique()), ['A2'])
