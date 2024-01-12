import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

# use pytest.mark.slow because all neuropythy tests
# take a long time to run. This way, they will we skipped
# unless the user passes --runslow to pytest
@pytest.mark.slow
def test_slow_test():
    print("This should not run")
    npt.assert_equal(True, False)