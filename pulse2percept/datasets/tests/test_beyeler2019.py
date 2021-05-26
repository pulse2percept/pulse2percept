import pandas as pd
import numpy.testing as npt
import pytest
from unittest import mock
from importlib import reload

from pulse2percept import datasets


def _is_beyeler2019_not_available():
    try:
        datasets.fetch_beyeler2019(download_if_missing=False)
        return False
    except IOError:
        return True


@pytest.mark.skipif(
    _is_beyeler2019_not_available(),
    reason='Download Beyeler2019 dataset to run this test'
)
def test_fetch_beyeler2019():
    data = datasets.fetch_beyeler2019(shuffle=False)

    npt.assert_equal(isinstance(data, pd.DataFrame), True)
    columns = ['subject', 'amp', 'area', 'compactness', 'date', 'eccentricity',
               'electrode', 'filename', 'freq', 'image', 'orientation', 'pdur',
               'stim_class', 'x_center', 'y_center', 'img_shape',
               'xrange', 'yrange']
    for expected_col in columns:
        npt.assert_equal(expected_col in data.columns, True)

    npt.assert_equal(data.shape, (400, 18))
    npt.assert_equal(data.subject.unique(), ['S1', 'S2', 'S3', 'S4'])
    npt.assert_equal(list(data[data.subject == 'S1'].img_shape.unique()[0]),
                     [192, 192])
    npt.assert_equal(list(data[data.subject != 'S1'].img_shape.unique()[0]),
                     [384, 512])

    # Subset selection:
    subset = datasets.fetch_beyeler2019(subjects='S2')
    npt.assert_equal(subset.shape, (110, 18))
    subset = datasets.fetch_beyeler2019(subjects=['S2', 'S3'])
    npt.assert_equal(subset.shape, (200, 18))
    subset = datasets.fetch_beyeler2019(subjects='invalid')
    npt.assert_equal(subset.shape, (0, 18))
    subset = datasets.fetch_beyeler2019(electrodes=['A1'])
    npt.assert_equal(subset.shape, (10, 18))
    subset = datasets.fetch_beyeler2019(electrodes=['A1', 'C3'])
    npt.assert_equal(subset.shape, (20, 18))
    subset = datasets.fetch_beyeler2019(electrodes='invalid')
    npt.assert_equal(subset.shape, (0, 18))

    # Shuffle dataset (index will always be range(400), but rows are shuffled):
    data = datasets.fetch_beyeler2019(shuffle=True, random_state=42)
    npt.assert_equal(data.loc[0, 'subject'], 'S3')
    npt.assert_equal(data.loc[0, 'electrode'], 'A2')
    npt.assert_equal(data.loc[399, 'subject'], 'S2')
    npt.assert_equal(data.loc[399, 'electrode'], 'D4')

    with mock.patch.dict("sys.modules", {"pandas": {}}):
        with pytest.raises(ImportError):
            reload(datasets)
            datasets.fetch_beyeler2019()
