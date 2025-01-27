"""`fetch_beyeler2019`, `subject_params`"""
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


subject_params = {
    'S1': {
        'subject_id': 'S1',
        'second_sight_id': 'TB',
        'implant_type_str': 'ArgusI',
        'implant_x': -1527,
        'implant_y': -556,
        'implant_rot': -64.74,
        'loc_od_x': 13.6,
        'loc_od_y': 0.0,
        'xmin': -36.9,
        'xmax': 36.9,
        'ymin': -36.9,
        'ymax': 36.9
    },
    'S2': {
        'subject_id': 'S2',
        'second_sight_id': '12-005',
        'implant_type_str': 'ArgusII',
        'implant_x': -1896,
        'implant_y': -542,
        'implant_rot': -44,
        'loc_od_x': 15.8,
        'loc_od_y': 1.4,
        'xmin': -30,
        'xmax': 30,
        'ymin': -22.5,
        'ymax': 22.5
    },
    'S3': {
        'subject_id': 'S3',
        'second_sight_id': '51-009',
        'implant_type_str': 'ArgusII',
        'implant_x': -1203,
        'implant_y': 280,
        'implant_rot': -35,
        'loc_od_x': 15.4,
        'loc_od_y': 1.57,
        'xmin': -32.5,
        'xmax': 32.5,
        'ymin': -24.4,
        'ymax': 24.4
    },
    'S4': {
        'subject_id': 'S4',
        'second_sight_id': '52-001',
        'implant_type_str': 'ArgusII',
        'implant_x': -1945,
        'implant_y': 469,
        'implant_rot': -34,
        'loc_od_x': 15.8,
        'loc_od_y': 1.51,
        'xmin': -32,
        'xmax': 32,
        'ymin': -24,
        'ymax': 24
    }
}


def fetch_beyeler2019(subjects=None, electrodes=None, data_path=None,
                      shuffle=False, random_state=0, download_if_missing=True):
    """Load the phosphene drawing dataset from [Beyeler2019]_

    Download the phosphene drawing dataset described in [Beyeler2019]_ from
    https://osf.io/28uqg (66MB) to ``data_path``. By default, all datasets are
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
    date                  Experiment date (YYYY/mm/dd)
    stim_class            Stimulus type used to stimulate the array
    amp                   Pulse amplitude used (x Threshold)
    freq                  Pulse frequency used (Hz)
    pdur                  Pulse duration used (ms)
    area                  Phosphene area (see [Beyeler2019]_ for details)
    orientation           Phosphene orientation (see [Beyeler2019]_)
    eccentricity          Phosphene elongation (see [Beyeler2019]_)
    compactness           Phosphene compactness (see [Beyeler2019]_)
    x_center, y_center    Phosphene center of mass (see [Beyeler2019]_)
    xrange, yrange        Screen size in deg (see [Beyeler2019]_)
    ====================  ================================================

    .. versionadded:: 0.6

    .. versionchanged:: 0.7

        Redirected download to 66MB version of the dataset that includes
        the fields ``x_center`` and ``y_center``.

    Parameters
    ----------
    subjects : str | list of strings | None, optional
        Select data from a subject or list of subjects. By default, all
        subjects are selected.
    electrodes : str | list of strings | None, optional
        Select data from a single electrode or a list of electrodes.
        By default, all electrodes are selected.
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
            url = 'https://osf.io/28uqg/download'
            checksum = '19817990a615d289cdc93b928c138f71977ea2cab54fd1b35d186f3b5a3c4ff5'
            fetch_url(url, file_path, remote_checksum=checksum)
        else:
            raise IOError(f"No local file {file_path} found")

    # Open the HDF5 file:
    f = h5py.File(file_path, 'r')

    # Create a DataFrame for every subject, then concatenate:
    dfs = []
    # Fields names are 'subject.field_name', so we split by '.' to find the
    # subject ID:
    for subject in np.unique([k.split('.')[0] for k in f.keys()]):
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
                df['image'] = [img.astype(bool) for img in f[key]]
            else:
                df[col] = f[key]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    f.close()

    # Combine 'img_shape_x' and 'img_shape_y' into 'img_shape' tuple
    df['img_shape'] = df.apply(lambda x: (x['img_shape_x'], x['img_shape_y']),
                               axis=1)
    df.drop(columns=['img_shape_x', 'img_shape_y'], inplace=True)

    # Verify integrity of the dataset:
    if len(df) != 400:
        raise ValueError(f"Try reloading the dataset: only {len(df)}/400 rows "
                         f"found")
    # Convert byte string to regular string, otherwise subset selection won't
    # work:
    try:
        df['subject'] = df['subject'].apply(lambda x: x.decode("utf-8"))
    except (UnicodeDecodeError, AttributeError):
        pass
    try:
        df['electrode'] = df['electrode'].apply(lambda x: x.decode("utf-8"))
    except (UnicodeDecodeError, AttributeError):
        pass

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
    df = df[idx]

    # Augment with implant type & location data:
    df['implant_type_str'] = ''
    df['implant_x'] = 0
    df['implant_y'] = 0
    df['implant_rot'] = 0
    df['xrange'] = None
    df['yrange'] = None
    for subject in df.subject.unique():
        df.loc[df.subject == subject,
               'implant_type_str'] = subject_params[subject]['implant_type_str']
        df.loc[df.subject == subject,
               'implant_x'] = subject_params[subject]['implant_x']
        df.loc[df.subject == subject,
               'implant_y'] = subject_params[subject]['implant_y']
        df.loc[df.subject == subject,
               'implant_rot'] = subject_params[subject]['implant_rot']
        df.loc[df.subject == subject, 'xrange'] = df.loc[df.subject == subject].apply(
            lambda row: (subject_params[subject]['xmin'], 
                         subject_params[subject]['xmax']), axis=1
        )
        df.loc[df.subject == subject, 'yrange'] = df.loc[df.subject == subject].apply(
            lambda row: (subject_params[subject]['ymin'], 
                         subject_params[subject]['ymax']), axis=1
        )

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)

    return df.reset_index(drop=True)
