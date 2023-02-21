from pulse2percept.topography import Polimeni2006Map
import numpy as np
import numpy.testing as npt

def test_polimeni_v1():
    map = Polimeni2006Map(regions=['v1'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v1'](theta, radius)
    result_theta, result_radius = map.to_dva()['v1'](x, y)

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta)
    npt.assert_almost_equal(radius, result_radius)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
                    | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

def test_polimeni_v2():
    map = Polimeni2006Map(regions=['v2'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v2'](theta, radius)
    result_theta, result_radius = map.to_dva()['v2'](x, y)

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta)
    npt.assert_almost_equal(radius, result_radius)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
                    | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

def test_polimeni_v3():
    map = Polimeni2006Map(regions=['v3'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1,13)])

    x, y = map.from_dva()['v3'](theta, radius)
    result_theta, result_radius = map.to_dva()['v3'](x, y)

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta)
    npt.assert_almost_equal(radius, result_radius)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
                    | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

