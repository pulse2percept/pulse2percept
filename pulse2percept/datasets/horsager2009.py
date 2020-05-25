"""`load_horsager2009`, `VariableDuration`"""
from os.path import dirname, join

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def load_horsager2009(shuffle=False, random_state=0):
    """Load data from [Horsager2009]_

    .. versionadded:: 0.6

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
    stim_type             'single_pulse', 'fixed_duration', ...
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

    Parameters
    ----------
    shuffle : boolean, optional, default: False
        If True, the rows of the DataFrame are shuffled.
    random_state : int | numpy.random.RandomState | None, optional, default: 0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    data: pd.DataFrame
        The whole dataset is returned in a 400x16 Pandas DataFrame
    """
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")
    # Load data from CSV:
    module_path = dirname(__file__)
    file_path = join(module_path, 'data', 'horsager2009.csv')
    df = pd.read_csv(file_path)

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
