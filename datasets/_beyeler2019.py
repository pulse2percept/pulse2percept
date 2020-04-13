from .base import fetch_url
from os import environ, path, makedirs
from os.path import join, expanduser, exists
import h5py
import numpy as np
import pandas as pd

def fetch_beyeler2019(data_path=None):
    """load the beyeler2019 dataset from data_path
    
    Download it from 'https://osf.io/6v2tb/' if necessary
    
    Parameter
    ---------
    data_path: string, default: None
        Specify another path for the datasets. By default
        beyeler2019 is stored in '~/pulse2percept_data'
    
    Return
    ------
    data: Pandas DataFrame
    """
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
    f = h5py.File(data_path, 'r')
    
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