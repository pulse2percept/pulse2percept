import utils
import numpy.testing as npt

def test_Parameters():
    my_params = utils.Parameters(foo='bar', list=[1, 2, 3])
    assert my_params.foo == 'bar'
    assert my_params.list == [1, 2, 3]
    assert str(my_params) == 'foo : bar\nlist : [1, 2, 3]'
    my_params.tuple = (1, 2, 3)
    assert my_params.tuple == (1, 2, 3)
