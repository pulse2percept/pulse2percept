import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.utils import parallel
from unittest import mock
from imp import reload


def power_it(num, n=2):
    return num ** n


@pytest.mark.parametrize('engine', ('serial', 'joblib', 'dask'))
@pytest.mark.parametrize('scheduler', ('threading', 'multiprocessing'))
def test_parfor(engine, scheduler):
    my_array = np.arange(100).reshape(10, 10)
    i, j = np.random.randint(0, 9, 2)
    my_list = list(my_array.ravel())

    expected_00 = power_it(my_array[0, 0])
    expected_ij = power_it(my_array[i, j])

    with pytest.raises(ValueError):
        parallel.parfor(power_it, my_list, engine='unknown')
    with pytest.raises(ValueError):
        parallel.parfor(power_it, my_list, engine='dask', scheduler='unknown')

    # `backend` only relevant for dask, will be ignored for others
    # and should thus still give the right result
    calculated_00 = parallel.parfor(power_it, my_list, engine=engine,
                                    scheduler=scheduler,
                                    out_shape=my_array.shape)[0, 0]
    calculated_ij = parallel.parfor(power_it, my_list, engine=engine,
                                    scheduler=scheduler,
                                    out_shape=my_array.shape)[i, j]

    npt.assert_equal(expected_00, calculated_00)
    npt.assert_equal(expected_ij, calculated_ij)

    with mock.patch.dict("sys.modules", {'dask': {}}):
        reload(parallel)
        with pytest.raises(ImportError):
            parallel.parfor(power_it, my_list, engine='dask',
                            out_shape=my_array.shape)[0, 0]
    reload(parallel)
