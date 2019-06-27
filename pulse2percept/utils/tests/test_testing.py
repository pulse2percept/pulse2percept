import warnings
from pulse2percept.utils.testing import assert_warns_msg


def _mock_warns(category=DeprecationWarning):
    warnings.warn(str(category), category=category)


def test_assert_warns_msg():
    for warning in [UserWarning, DeprecationWarning]:
        assert_warns_msg(warning, _mock_warns, str(warning), category=warning)
