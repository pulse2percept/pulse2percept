import pandas as pd
import numpy.testing as npt
import pytest
from unittest import mock
from importlib import reload

from pulse2percept import datasets


def test_load_horsager2009():
    data = datasets.load_horsager2009(shuffle=False)

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    columns = ['subject', 'implant', 'electrode', 'task', 'stim_type',
               'stim_dur', 'stim_freq', 'stim_amp', 'pulse_type', 'pulse_dur',
               'pulse_num', 'interphase_dur', 'delay_dur', 'ref_stim_type',
               'ref_freq', 'ref_amp', 'ref_amp_factor', 'ref_pulse_dur',
               'ref_interphase_dur', 'theta', 'source']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (552, 21))
    npt.assert_equal(data.subject.unique(), ['S05', 'S06'])

    # Shuffle dataset (index will always be range(552), but rows are shuffled):
    data = datasets.load_horsager2009(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'subject'], 'S06')
    npt.assert_equal(data.loc[0, 'electrode'], 'D1')
    npt.assert_equal(data.loc[0, 'stim_type'], 'latent_addition')
    npt.assert_equal(data.loc[551, 'subject'], 'S06')
    npt.assert_equal(data.loc[551, 'electrode'], 'A1')
    npt.assert_equal(data.loc[551, 'stim_type'], 'fixed_duration')
