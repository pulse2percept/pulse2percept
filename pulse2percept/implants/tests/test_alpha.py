import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import implants


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('r', (-45, 60))
def test_AlphaIMS(ztype, x, y, r):
    # Create an ArgusII and make sure location is correct
    # Height `h` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(1369) * 20
    # Convert rotation angle to rad
    rot = np.deg2rad(r)
    argus = implants.AlphaIMS(x=x, y=y, z=z, rot=rot)

    # TODO
