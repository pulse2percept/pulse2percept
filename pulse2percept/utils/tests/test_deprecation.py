import numpy.testing as npt
from pulse2percept.utils.deprecation import deprecated, is_deprecated
from pulse2percept.utils.testing import assert_warns_msg


@deprecated(alt_func='qwerty')
class MockClass1:
    pass


class MockClass2(object):

    @deprecated(deprecated_version=0.1, removed_version=0.2)
    def mymethod(self):
        pass


class MockClass3:

    @deprecated()
    def __init__(self):
        pass


class MockClass4:
    pass


@deprecated(deprecated_version=0.4)
def mock_function():
    return 10


def test_deprecated():
    assert_warns_msg(DeprecationWarning, MockClass1, 'Use ``qwerty`` instead')
    assert_warns_msg(DeprecationWarning, MockClass2().mymethod,
                     'since version 0.1, and will be removed in version 0.2')
    assert_warns_msg(DeprecationWarning, MockClass3, 'deprecated')
    assert_warns_msg(DeprecationWarning, mock_function, 'since version 0.4')


def test_is_deprecated():
    # Test if is_deprecated helper identifies wrapping via deprecated:
    # NOTE it works only for class methods and functions
    npt.assert_equal(is_deprecated(MockClass1.__init__), True)
    npt.assert_equal(is_deprecated(MockClass2().mymethod), True)
    npt.assert_equal(is_deprecated(MockClass3.__init__), True)
    npt.assert_equal(is_deprecated(MockClass4.__init__), False)
    npt.assert_equal(is_deprecated(mock_function), True)
