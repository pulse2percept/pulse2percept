from pulse2percept import datasets
import pytest
import numpy.testing as npt

def _is_han2021_not_available():
    try:
        datasets.fetch_han2021(download_if_missing=False)
        return False
    except IOError:
        return True

@pytest.mark.skipif(
    _is_han2021_not_available(),
    reason='Download han2021 dataset to run this test'
)
def test_han2021():
    data = datasets.fetch_han2021()
    npt.assert_equal(len(data.keys()), 20)
    npt.assert_equal(data['stim4'].vid_shape, (180, 320, 3, 125))
    npt.assert_equal(data['stim5'].vid_shape, (180, 320, 3, 126))
    npt.assert_equal(data['stim6'].metadata['source_size'], (960, 540))
    npt.assert_equal(data['stim7'].metadata['source'], 'stim7.mp4')
    npt.assert_almost_equal(data['stim8'].data[200,100], 0.16078432)

    #check if resize works
    data2 = datasets.fetch_han2021(resize = (18, 32))
    npt.assert_equal(data2['sample1'].vid_shape, (18, 32, 3, 125))
    npt.assert_almost_equal(data2['sample1'].data[100, 50], 0.1577467)
    
    #check if as_gray worksk
    data3 = datasets.fetch_han2021(['stim1','stim2'], as_gray = True)
    npt.assert_equal(len(data3), 2)
    npt.assert_equal(data3['stim1'].vid_shape, (180, 320, 125))
    npt.assert_almost_equal(data3['stim2'].data[300,50], 0.08373686)

    #check if value error throws when inputing invalid name
    with pytest.raises(ValueError):
        datasets.fetch_han2021('sti')