"""`load_fornos2012`"""
from os.path import dirname, join

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def load_fornos2012(shuffle=False, subjects=None, figures=None, random_state=0):
    """Load data from [Fornos2012]_

        Load the brightness associated with joystick position data described in [Fornos2012]_.
        Datapoints were extracted from figures 3-7 of the paper.


        ===================   =====================
        Retinal implants:                  Argus II
        Subjects:                                 9
        Number of samples:                       45
        Number of features:                     158
        ===================   =====================

        The dataset includes the following features:

        ====================  ================================================
        Subject               Subject ID, S06
        Figure                Reference figure from [Fornos2012]_
        time value            Float time step in the range [-2.0, 75.7] in steps of 0.5. Note, this is a float value,
                              there is no explicit feature value of "time value"
        ====================  ================================================

        .. versionadded:: 0.7

        Parameters
        ----------
        shuffle : boolean, optional
            If True, the rows of the DataFrame are shuffled.
        random_state : int | numpy.random.RandomState | None, optional
            Determines random number generation for dataset shuffling. Pass an int
            for reproducible output across multiple function calls.
        subjects : str | list of strings | None, optional
            Select data from a single subject or a list of subjects.
            By default, all subjects are selected.
        figures : str | list of strings | None, optional
            Select data from a single figure or a list of figures.
            By default, all figures are selected
        Returns
        -------
        data: pd.DataFrame
            The whole dataset is returned in a 45x158 Pandas DataFrame

        """
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")

    # Load data from CSV:
    module_path = dirname(__file__)
    file_path = join(module_path, 'data', 'perez-fornos-2012.csv')
    df = pd.read_csv(file_path)

    if subjects is not None:
        if isinstance(subjects, str):
            subjects = [subjects]

        df = df[df['Subject'].isin(subjects)]

    if figures is not None:
        if isinstance(figures, str):
            figures = [figures]

        df = df[df['Figure'].isin(figures)]

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
