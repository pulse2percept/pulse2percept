"""`load_nanduri2012`"""
from os.path import dirname, join
import numpy as np

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def load_nanduri2012(electrodes=None, shuffle=False, random_state=0):
    """Load data from [Nanduri2012]_

    Load the threshold data described in [Nanduri2012]_. Average thresholds
    were extracted from the figures of the paper using WebplotDigitizer.

    ===================   =====================
    Retinal implants:                   Argus I
    Subjects:                                 1
    Number of samples:                       95
    Number of features:                      14
    ===================   =====================

    The dataset includes the following features:

    ====================  ================================================
    subject               Subject ID, S06
    implant               Argus I
    electrode             Electrode ID, A2, A4, B1, C1, C4, D2, D3, D4
    task                  'rate'
    stim_type             'fixed_duration'
    stim_dur              Stimulus duration (ms)
    stim_freq             Stimulus frequency (Hz)
    stim_amp_factor       Stimulus amplitude ratio over threshold
    brightness            Patient rated brightness compared to reference
                          stimulus
    pulse_dur             Pulse duration (ms)
    pulse_type            'cathodic_first'
    interphase_dur        Interphase gap (ms)
    delay_dur             Stimulus delivered after delay (ms)
    source                Figure from which data was extracted
    ====================  ================================================

    .. versionadded:: 0.7

    Parameters
    ----------
    electrodes : str | list of strings | None, optional
        Select data from a single electrode or a list of electrodes.
        By default, all electrodes are selected.
    shuffle : boolean, optional
        If True, the rows of the DataFrame are shuffled.
    random_state : int | numpy.random.RandomState | None, optional
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    data: pd.DataFrame
        The whole dataset is returned in a 95x14 Pandas DataFrame

    """
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")
    # Load data from CSV:
    module_path = dirname(__file__)
    file_path = join(module_path, 'data', 'nanduri2012.csv')
    df = pd.read_csv(file_path)

    # Select subset of data:
    idx = np.ones_like(df.index, dtype=np.bool)
    if electrodes is not None:
        if isinstance(electrodes, str):
            electrodes = [electrodes]
        idx_electrode = np.zeros_like(df.index, dtype=np.bool)
        for electrode in electrodes:
            idx_electrode |= df.electrode == electrode
        idx &= idx_electrode
    df = df[idx]
    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
