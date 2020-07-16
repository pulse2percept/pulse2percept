"""`fetch_beyeler2019`"""
from os.path import join, isfile
import numpy as np

from .base import get_data_dir, fetch_url

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False


def fetch_beyeler2019(data_path=None, shuffle=False, random_state=0,
                      download_if_missing=True):
    """Load the phosphene drawing dataset from [Beyeler2019]_

    Download the phosphene drawing dataset described in [Beyeler2019]_ from
    https://osf.io/6v2tb (263MB) to ``data_path``. By default, all datasets are
    stored in '~/pulse2percept_data/', but a different path can be specified.

    ===================   =====================
    Retinal implants:         Argus I, Argus II
    Subjects:                                 4
    Number of samples:                      400
    Number of features:                      16
    ===================   =====================

    The dataset includes the following features:

    ====================  ================================================
    subject               Subject ID, S1-S4
    electrode             Electrode ID, A1-F10
    image                 Phosphene drawing
    img_shape             x,y shape of the phosphene drawing
    date                  Experiment date
    stim_class            Stimulus type used to stimulate the array
    amp                   Pulse amplitude used (x Threshold)
    freq                  Pulse frequency used
    pdur                  Pulse duration used
    area                  Phosphene area (see [Beyeler2019]_ for details)
    orientation           Phosphene orientation (see [Beyeler2019]_)
    eccentricity          Phosphene elongation (see [Beyeler2019]_)
    compactness           Phosphene compactness (see [Beyeler2019]_)
    x_center, y_center    Phosphene center of mass (see [Beyeler2019]_)
    ====================  ================================================

    Parameters
    ----------
    data_path: string, optional
        Specify another download and cache folder for the dataset. By default
        all pulse2percept data is stored in '~/pulse2percept_data' subfolders.
    shuffle : boolean, optional
        If True, the rows of the DataFrame are shuffled.
    random_state : int | numpy.random.RandomState | None, optional, default: 0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
    download_if_missing : optional
        If False, raise an IOError if the data is not locally available
        instead of trying to download it from the source site.

    Returns
    -------
    data: pd.DataFrame
        The whole dataset is returned in a 400x16 Pandas DataFrame

    .. versionadded:: 0.6

    """
    if not has_h5py:
        raise ImportError("You do not have h5py installed. "
                          "You can install it via $ pip install h5py.")
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")
    # Create the local data directory if it doesn't already exist:
    data_path = get_data_dir(data_path)

    # Download the dataset if it doesn't already exist:
    file_path = join(data_path, 'beyeler2019.h5')
    if not isfile(file_path):
        if download_if_missing:
            url = 'https://osf.io/6v2tb/download'
            checksum = '211818c598c27d33d4e0cd5cdbac9e3ad23106031eb7b51c1a78ccaff000e156'
            fetch_url(url, file_path, remote_checksum=checksum)
        else:
            raise IOError("No local file %s found" % file_path)

    # Open the HDF5 file:
    f = h5py.File(file_path, 'r')

    # Fields names are 'subject.field_name', so we split by '.' to find the
    # subject ID:
    subjects = np.unique([k.split('.')[0] for k in f.keys()])

    # Create a DataFrame for every subject, then concatenate:
    dfs = []
    for subject in subjects:
        df = pd.DataFrame()
        df['subject'] = subject
        for key in f.keys():
            if subject not in key:
                continue
            # Find the field name - that's the DataFrame column:
            col = key.split('.')[1]
            if col == 'image':
                # Images need special treatment:
                # - Direct assign confuses Pandas, need a loop
                # - Save as black/white boolean
                df['image'] = [img.astype(np.bool) for img in f[key]]
            else:
                df[col] = f[key]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    f.close()

    # Combine 'img_shape_x' and 'img_shape_y' into 'img_shape' tuple
    df['img_shape'] = df.apply(lambda x: (x['img_shape_x'], x['img_shape_y']),
                               axis=1)
    df.drop(columns=['img_shape_x', 'img_shape_y'], inplace=True)

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
