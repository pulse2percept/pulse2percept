"""`load_greenwald2009`"""

from os.path import dirname, join

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def load_greenwald2009(subjects=None, electrodes=None, shuffle=False, random_state=0):
    """Load data from [Greenwald2009]_

        Load the brightness and size rating data described in [Greenwald2009]_. Datapoints
        were extracted from figure 4 of the paper using WebplotDigitizer.

        ===================   =====================
        Retinal implants:                   Argus I
        Subjects:                                 2
        Number of samples:                       83
        Number of features:                      12
        ===================   =====================

        The dataset includes the following features:

        ====================  ================================================
        subject               Subject ID, S06
        implant               Argus I
        electrode             Electrode ID, A2, A4, B1, C1, C4, D2, D3, D4
        task                  'rate'
        stim_class            "Greenwald2009"
        stim_dur              Stimulus duration (ms)
        amp                   Amplitude of the stimulation
        brightness            Patient reported brightness
        pulse_dur             Pulse duration (ms)
        interphase_dur        Interphase gap (ms)
        pulse_type            'cathodicFirst'
        threshold             Electrical stimulation threshold
        ====================  ================================================

        .. versionadded:: 0.7

        Parameters
        ----------
        subjects : str | list of strings | None, optional
            Select data from a single subject or a list of subjects.
            By default, all subjects are selected
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
            The whole dataset is returned in a 144x16 Pandas DataFrame

        """
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")

    # Load data from csv
    module_path = dirname(__file__)
    file_path = join(module_path, 'data', 'greenwald2009.csv')
    df = pd.read_csv(file_path)

    # Select a subset of data based on subject
    if subjects is not None:
        if isinstance(subjects, str):
            subjects = [subjects]

        df = df[df['subject'].isin(subjects)]

    # Select a subset of data based on electrode
    if electrodes is not None:
        if isinstance(electrodes, str):
            electrodes = [electrodes]

        df = df[df['electrode'].isin(electrodes)]

    # shuffle the data
    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
