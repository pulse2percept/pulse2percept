"""`load_data`, `fetch_url`, `fetch_beyeler2019`"""
import h5py
import hashlib
from urllib.request import urlretrieve
from os.path import join, dirname, exists, expanduser
from os import path, environ, makedirs
import csv
import numpy as np
import pandas as pd

def load_data(data_path):
    try:
        # Assume it's HDF5 and convert it to csv
        data=_hdf2df(data_path)
    except OSError:
        # Could not read file, must be csv:
        try:
            data=pd.read_csv(data_path)
        except ParserError:
            raise TypeError("File is neither csv nor hdf5")
    return data

def _hdf2df(hdf_file):
    """Converts the data from HDF5 to a Pandas DataFrame"""
    f = h5py.File(hdf_file, 'r')
    
    # Fields names are 'subject.field_name', so we split by '.'
    # to find the subject ID:
    subjects = np.unique([k.split('.')[0] for k in f.keys()])
    
    # Create a DataFrame for every subject, then concatenate:
    dfs = []
    for subject in subjects:
        df = pd.DataFrame()
        df['subject'] = subject
        for key in f.keys():
            if subject not in key:
                continue
            # Find the field name, that's the DataFrame column:
            col = key.split('.')[1]
            if col == 'image':
                # Images need special treatment:
                # - Direct assign confuses Pandas, need a loop
                # - Convert back to float so scikit_image can handle it
                df['image'] = [img.astype(np.float64) for img in f[key]]
            else:
                df[col] = f[key]
        dfs.append(df)
    dfs = pd.concat(dfs)
    f.close()
    
    # Combine 'img_shape_x' and 'img_shape_y' back into 'img_shape' tuple
    dfs['img_shape'] = dfs.apply(lambda x: (x['img_shape_x'], x['img_shape_y']), axis=1)
    dfs.drop(columns=['img_shape_x', 'img_shape_y'], inplace=True)
    return dfs.reset_index()    

def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()

def fetch_url(url, file_path, remote_checksum=None):
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by url, save into path using the file_path
    and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.
    Parameters
    ----------
    url : string
        url to the file.
    file_path: string
        Full path of the created file.
    """
    urlretrieve(url, file_path)
    checksum = _sha256(file_path)
    if remote_checksum != None and remote_checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))

def fetch_beyeler2019(data_path=None):
    # if the data file 'beyeler2019' is not exists
    # download it from the website 'https://osf.io/6v2tb/'
    if data_path is None:
        data_path = environ.get('PULSE2PERCEPT_DATA',
                               join('~', 'pulse2percept_data'))
    # check if the path exists
    data_path = expanduser(data_path)
    if not exists(data_path):
        makedirs(data_path)
    # check if the data file 'beyeler2019' exists under the path
    data_path = join(data_path, 'beyeler2019')
    if not path.isfile(data_path): 
        url='https://files.osf.io/v1/resources/dw9nz/providers/osfstorage/5e78ce214a60a506abbb4f58?action=download&direct&version=1'
        checksum='211818c598c27d33d4e0cd5cdbac9e3ad23106031eb7b51c1a78ccaff000e156'
        fetch_url(url,data_path,remote_checksum=checksum)
    # Convert the HDF5 data file to a Pandas DataFrame
    return load_data(data_path)