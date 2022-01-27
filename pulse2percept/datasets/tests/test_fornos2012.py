import pandas as pd
import numpy.testing as npt
import numpy as np

from pulse2percept.datasets import load_fornos2012


def test_load_fornos2012():
    data = load_fornos2012()

    # Test that data is a Pandas DataFrame
    npt.assert_equal(isinstance(data, pd.DataFrame), True)

    # Test shape of the dataframe
    npt.assert_equal(data.shape, (45, 3))

    # Test that all the expected columns are present
    columns = ['figure', 'subject', 'time_series']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    # Test each time series has the expected number of points
    expected_times_series_length = len(np.arange(start=-2.0, stop=75.5, step=0.5))
    for time_series in data['time_series']:
        npt.assert_equal(isinstance(time_series, np.ndarray), True)
        npt.assert_equal(len(time_series), expected_times_series_length)

    # Test shuffling the data
    data = load_fornos2012(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'figure'], 'fig7_S4')
    npt.assert_equal(data.loc[0, 'subject'], 'S4')
    npt.assert_equal(data.loc[40, 'figure'], 'fig5_S3')
    npt.assert_equal(data.loc[40, 'subject'], 'S3')

    # Test selecting by subjects
    data = load_fornos2012(subjects='S2')
    npt.assert_equal(data.shape, (5, 3))
    npt.assert_equal(data.subject.unique(), 'S2')
    data = load_fornos2012(subjects=['S2', 'S3'])
    npt.assert_equal(data.shape, (10, 3))
    npt.assert_equal(data.subject.unique(), ['S2', 'S3'])

    # Test selecting by figure
    data = load_fornos2012(figures='fig3_S7')
    npt.assert_equal(data.shape, (1, 3))
    npt.assert_equal(data.figure.unique(), 'fig3_S7')
    data = load_fornos2012(figures=['fig3_S7', 'fig4_S3'])
    npt.assert_equal(data.shape, (2, 3))
    npt.assert_equal(data.figure.unique(), ['fig3_S7', 'fig4_S3'])
