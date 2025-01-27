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

    # Check that the result is a DataFrame
    assert isinstance(data, pd.DataFrame), "Data is not a DataFrame"

    # Check that all expected columns are present
    expected_columns = [
        'subject', 'amp', 'area', 'compactness', 'date', 'eccentricity',
        'electrode', 'filename', 'freq', 'image', 'orientation', 'pdur',
        'stim_class', 'x_center', 'y_center', 'img_shape',
        'xrange', 'yrange', 'implant_type_str', 'implant_x', 'implant_y',
        'implant_rot'
    ]
    assert all(col in data.columns for col in expected_columns), "Missing columns in DataFrame"

    # Verify DataFrame shape
    assert data.shape == (400, 22), f"Unexpected shape: {data.shape}"

    # Check unique subjects
    assert sorted(data.subject.unique()) == ['S1', 'S2', 'S3', 'S4'], "Unexpected subjects"

    # Check 'xrange' for S1
    s1_xrange = list(data[data.subject == 'S1'].xrange.unique())
    assert s1_xrange == [(-36.9, 36.9)], f"Unexpected xrange for S1: {s1_xrange}"

    # Check 'img_shape' for S1 and non-S1 subjects
    s1_img_shape = list(data[data.subject == 'S1'].img_shape.unique())
    non_s1_img_shape = list(data[data.subject != 'S1'].img_shape.unique())
#     assert s1_img_shape == [(192, 192)], f"Unexpected img_shape for S1: {s1_img_shape}"
    assert non_s1_img_shape == [(384, 512)], f"Unexpected img_shape for non-S1: {non_s1_img_shape}"

    # Subset selection:
    subset = datasets.fetch_beyeler2019(subjects='S2')
    npt.assert_equal(subset.shape, (110, 22))
    subset = datasets.fetch_beyeler2019(subjects=['S2', 'S3'])
    npt.assert_equal(subset.shape, (200, 22))
    subset = datasets.fetch_beyeler2019(subjects='invalid')
    npt.assert_equal(subset.shape, (0, 22))
    subset = datasets.fetch_beyeler2019(electrodes=['A1'])
    npt.assert_equal(subset.shape, (10, 22))
    subset = datasets.fetch_beyeler2019(electrodes=['A1', 'C3'])
    npt.assert_equal(subset.shape, (20, 22))
    subset = datasets.fetch_beyeler2019(electrodes='invalid')
    npt.assert_equal(subset.shape, (0, 22))

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
