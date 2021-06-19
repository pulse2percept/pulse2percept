"""`load_horsager2009`"""
from os.path import dirname, join
import numpy as np

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def load_horsager2009(subjects=None, electrodes=None, stim_types=None,
                      shuffle=False, random_state=0):
    """Load data from [Horsager2009]_

    Load the threshold data described in [Horsager2009]_. Average thresholds
    were extracted from the figures of the paper using WebplotDigitizer.

    ===================   =====================
    Retinal implants:                   Argus I
    Subjects:                                 2
    Number of samples:                      552
    Number of features:                      21
    ===================   =====================

    The dataset includes the following features:

    ====================  ================================================
    subject               Subject ID, S05-S06
    implant               Argus I
    electrode             Electrode ID, A1-F10
    task                  'threshold' or 'matching'
    stim_type             'single_pulse', 'fixed_duration',
                          'variable_duration', 'fixed_duration_supra',
                          'bursting_triplets', 'bursting_triplets_supra',
                          'latent_addition'
    stim_dur              Stimulus duration (ms)
    stim_freq             Stimulus frequency (Hz)
    stim_amp              Stimulus amplitude (uA)
    pulse_type            'cathodic_first'
    pulse_dur             Pulse duration (ms)
    interphase_dur        Interphase gap (ms)
    delay_dur             Stimulus delivered after delay (ms)
    ref_stim_type         Reference stimulus type ('single_pulse', ...)
    ref_freq              Reference stimulus frequency (Hz)
    ref_amp               Reference stimulus amplitude (uA)
    ref_amp_factor        Reference stimulus amplitude factor (xThreshold)
    ref_pulse_dur         Reference stimulus pulse duration (ms)
    ref_interphase_dur    Reference stimulus interphase gap (ms)
    theta                 Temporal model output at threshold (a.u.)
    source                Figure from which data was extracted
    ====================  ================================================

    Some stimulus types require a reference stimulus. For example,
    'bursting_triplets_supra' were delivered at 2x or 3x threshold of a
    reference bursting triplet pulse. The parameters of the reference stimulus
    are given in ``ref_*`` fields.

    Missing values are denoted by NaN.

    .. versionadded:: 0.6

    Parameters
    ----------
    subjects : str | list of strings | None, optional
        Select data from a subject or list of subjects. By default, all
        subjects are selected.
    electrodes : str | list of strings | None, optional
        Select data from a single electrode or a list of electrodes.
        By default, all electrodes are selected.
    stim_types : str | list of strings | None, optional
        Select data from a single stimulus type or a list of stimulus types.
        By default, all stimulus types are selected.
    shuffle : boolean, optional
        If True, the rows of the DataFrame are shuffled.
    random_state : int | numpy.random.RandomState | None, optional
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    data: pd.DataFrame
        The whole dataset is returned in a 552x21 Pandas DataFrame

    """
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")
    # Load data from CSV:
    module_path = dirname(__file__)
    file_path = join(module_path, 'data', 'horsager2009.csv')
    df = pd.read_csv(file_path)

    # Select subset of data:
    idx = np.ones_like(df.index, dtype=bool)
    if subjects is not None:
        if isinstance(subjects, str):
            subjects = [subjects]
        idx_subject = np.zeros_like(df.index, dtype=bool)
        for subject in subjects:
            idx_subject |= df.subject == subject
        idx &= idx_subject
    if electrodes is not None:
        if isinstance(electrodes, str):
            electrodes = [electrodes]
        idx_electrode = np.zeros_like(df.index, dtype=bool)
        for electrode in electrodes:
            idx_electrode |= df.electrode == electrode
        idx &= idx_electrode
    if stim_types is not None:
        if isinstance(stim_types, str):
            stim_types = [stim_types]
        idx_type = np.zeros_like(df.index, dtype=bool)
        for stim_type in stim_types:
            idx_type |= df.stim_type == stim_type
        idx &= idx_type
    df = df[idx]

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
