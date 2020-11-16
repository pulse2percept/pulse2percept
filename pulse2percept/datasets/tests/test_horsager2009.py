import pandas as pd
import numpy.testing as npt
import pytest
from unittest import mock
from importlib import reload

from pulse2percept.datasets import load_horsager2009


def test_load_horsager2009():
    data = load_horsager2009(shuffle=False)

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    columns = ['subject', 'implant', 'electrode', 'task', 'stim_type',
               'stim_dur', 'stim_freq', 'stim_amp', 'pulse_type', 'pulse_dur',
               'pulse_num', 'interphase_dur', 'delay_dur', 'ref_stim_type',
               'ref_freq', 'ref_amp', 'ref_amp_factor', 'ref_pulse_dur',
               'ref_interphase_dur', 'theta', 'source']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (608, 21))
    npt.assert_equal(data.subject.unique(), ['S05', 'S06'])

    # Shuffle dataset (index will always be range(607), but rows are shuffled):
    data = load_horsager2009(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'subject'], 'S06')
    npt.assert_equal(data.loc[0, 'electrode'], 'B3')
    npt.assert_equal(data.loc[0, 'stim_type'], 'fixed_duration')
    npt.assert_equal(data.loc[607, 'subject'], 'S06')
    npt.assert_equal(data.loc[607, 'electrode'], 'D1')
    npt.assert_equal(data.loc[607, 'stim_type'], 'latent_addition')

    # Select subjects:
    data = load_horsager2009(subjects='S05')
    npt.assert_equal(data.shape, (296, 21))
    npt.assert_equal(data.subject.unique(), 'S05')
    data = load_horsager2009(subjects=['S05', 'S07'])  # 'S07' doesnt' exist
    npt.assert_equal(data.shape, (296, 21))
    npt.assert_equal(data.subject.unique(), 'S05')
    data = load_horsager2009(subjects=['S05', 'S06'])  # same as None
    npt.assert_equal(data.shape, (608, 21))
    data = load_horsager2009(subjects='S6')  # 'S6' doesn't exist
    npt.assert_equal(data.shape, (0, 21))
    npt.assert_equal(data.subject.unique(), [])

    # Select electrodes:
    data = load_horsager2009(electrodes='A1')
    npt.assert_equal(data.shape, (114, 21))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), ['S06', 'S05'])
    data = load_horsager2009(electrodes=['A1', 'A9'])  # 'A9' doesn't exist
    npt.assert_equal(data.shape, (114, 21))
    npt.assert_equal(data.electrode.unique(), 'A1')
    npt.assert_equal(data.subject.unique(), ['S06', 'S05'])

    # Select stimulus types:
    data = load_horsager2009(stim_types='single_pulse')
    npt.assert_equal(data.shape, (80, 21))
    npt.assert_equal(data.stim_type.unique(), 'single_pulse')
    npt.assert_equal(list(data.subject.unique()), ['S05', 'S06'])
    data = load_horsager2009(stim_types=['single_pulse', 'fixed_duration'])
    npt.assert_equal(data.shape, (200, 21))
    npt.assert_equal(list(data.stim_type.unique()),
                     ['single_pulse', 'fixed_duration'])
    npt.assert_equal(list(data.subject.unique()), ['S05', 'S06'])

    # Subject + electrode + stim type:
    data = load_horsager2009(subjects='S05', electrodes=['A1', 'C3'],
                             stim_types='single_pulse')
    npt.assert_equal(data.shape, (16, 21))
    npt.assert_equal(data.subject.unique(), 'S05')
    npt.assert_equal(list(data.electrode.unique()), ['C3', 'A1'])
    npt.assert_equal(data.stim_type.unique(), 'single_pulse')
