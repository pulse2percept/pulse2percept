import pandas as pd
import numpy.testing as npt
import numpy as np

from pulse2percept.datasets import load_fornos2012


def test_load_fornos2012():
    data = load_fornos2012()

    npt.assert_equal(isinstance(data, pd.DataFrame), True)

    time_steps = [str(time) for time in np.arange(start=-2.0, stop=75.5, step=0.5)]
    columns = ['Figure', 'Subject'] + time_steps

    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (45, 158))

    # Test shuffling the data
    data = load_fornos2012(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'Figure'], 'fig7_S4')
    npt.assert_equal(data.loc[0, 'Subject'], 'S4')
    npt.assert_equal(data.loc[40, 'Figure'], 'fig5_S3')
    npt.assert_equal(data.loc[40, 'Subject'], 'S3')

    # Test selecting by subjects
    data = load_fornos2012(subjects='S2')
    npt.assert_equal(data.shape, (5, 158))
    npt.assert_equal(data.Subject.unique(), 'S2')
    data = load_fornos2012(subjects=['S2', 'S3'])
    npt.assert_equal(data.shape, (10, 158))
    npt.assert_equal(data.Subject.unique(), ['S2', 'S3'])

    # Test selecting by figure
    data = load_fornos2012(figures='fig3_S7')
    npt.assert_equal(data.shape, (1, 158))
    npt.assert_equal(data.Subject.unique(), 'S7')
    data = load_fornos2012(figures=['fig3_S7', 'fig4_S3'])
    npt.assert_equal(data.shape, (2, 158))
    npt.assert_equal(data.Subject.unique(), ['S7', 'S3'])
