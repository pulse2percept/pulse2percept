from pulse2percept.topography import Polimeni2006Map
from pulse2percept.utils import pol2cart, cart2pol
import numpy as np
import numpy.testing as npt


def test_polimeni_v1():
    map = Polimeni2006Map(regions=['v1'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v1'](*pol2cart(theta, radius))

    expected_x = [
        15902.589306126785, 22701.701209282972, 27105.66605223716,
        -51823.503877716503, -55086.034551492055, -57506.13668631103,
        -59506.3130414652, -61373.30525591315, -63293.183557664786,
        44457.872804343075, 45423.62865236582, 46405.82649122574
    ]
    expected_y = [
        5013.8355343520525, 12359.443849528407, 20342.321611408766,
        13847.746737780984, 6984.86446795094, 215.14551955541478,
        -6584.422109361202, -13588.223582967372, -20987.998282780045,
        -13939.6205118376, -6913.876163034446, -210.58956294736032
    ]

    # check that the conversion from dva to v1 was correct
    npt.assert_almost_equal(x, expected_x, 2)
    npt.assert_almost_equal(y, expected_y, 2)

    result_theta, result_radius = cart2pol(*map.to_dva()['v1'](x, y))

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta, 5)
    npt.assert_almost_equal(radius, result_radius, 5)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
        | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

    # check that points in the upper half of the visual field
    # correspond to the lower half of v1
    y_match = ((y > 0) & (theta < 0)) | ((y < 0) & (theta > 0))
    npt.assert_equal(y_match, True)


def test_polimeni_v2():
    map = Polimeni2006Map(regions=['v2'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v2'](*pol2cart(theta, radius))

    expected_x = [
        9689.877814630294, 20645.517280735134, 27062.815960747776,
        -51088.965767776444, -54358.02828705147, -57160.94659139381,
        -59613.0489766391, -61630.58330730293, -63307.69628722014,
        45017.75586687664, 46671.23009322621, 48277.67564932234
    ]
    expected_y = [
        20977.926765752603, 22151.218922100634, 20665.850562060903,
        23554.244234724538, 26453.386814786292, 29296.33682195241,
        -26724.6759835154, -23999.446003854897, -21316.642282519246,
        -23681.752491457257, -26265.911354121294, -28919.10673669272
    ]

    # check that the conversion from dva to v2 was correct
    npt.assert_almost_equal(x, expected_x, 2)
    npt.assert_almost_equal(y, expected_y, 2)

    result_theta, result_radius = cart2pol(*map.to_dva()['v2'](x, y))

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta, 5)
    npt.assert_almost_equal(radius, result_radius, 5)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
        | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

    # check that points in the upper half of the visual field
    # correspond to the lower half of v2
    y_match = ((y > 0) & (theta < (0))) | ((y < 0) & (theta > (0)))
    npt.assert_equal(y_match, True)


def test_polimeni_v3():
    map = Polimeni2006Map(regions=['v3'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v3'](*pol2cart(theta, radius))

    expected_x = [
        7208.613020331824, 18466.684974346612, 25257.495476836517,
        -50340.83769587597, -54148.17463378478, -57158.5303860779,
        -59641.052094685976, -61865.30148649867, -63893.55339523437,
        45582.11723185931, 47036.96603224987, 48291.23011297499
    ]
    expected_y = [
        25395.771704208673, 31847.85547750596, 35166.10649079788,
        33467.660650388934, 31491.043510450312, 29452.02958700397,
        -31445.110291796, -33567.49807200242, -35706.33963963459,
        -33578.30210820712, -31317.5638427424, -29075.77983553856
    ]

    # check that the conversion from dva to v3 was correct
    npt.assert_almost_equal(x, expected_x, 2)
    npt.assert_almost_equal(y, expected_y, 2)

    result_theta, result_radius = cart2pol(*map.to_dva()['v3'](x, y))

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta, 5)
    npt.assert_almost_equal(radius, result_radius, 5)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
        | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

    # check that points in the upper half of the visual field
    # correspond to the lower half of v3
    y_match = ((y > 0) & (theta < (0))) | ((y < 0) & (theta > (0)))
    npt.assert_equal(y_match, True)

def test_polimeni_y_inversion():
    map = Polimeni2006Map(regions=['v1','v2','v3'])

    # check that y-inversion is correct
    # point 0 is A and point 1 is B
    # A is almost directly above B in the visual field
    # so should be below B in v1 & v3
    # and above B in v2
    theta = np.array([np.pi/3, np.pi/6])
    radius = np.array([3, 3])
    v1x, v1y = map.from_dva()['v1'](*pol2cart(theta, radius))
    v2x, v2y = map.from_dva()['v2'](*pol2cart(theta, radius))
    v3x, v3y = map.from_dva()['v3'](*pol2cart(theta, radius))

    npt.assert_equal(v1y[0] < 0 and v1y[0] < v1y[1], True)
    npt.assert_equal(v2y[0] < 0 and v2y[0] > v2y[1], True)
    npt.assert_equal(v3y[0] < 0 and v3y[0] < v3y[1], True)

def test_polimeni_continuity():
    map = Polimeni2006Map(regions=['v1', 'v2', 'v3'])
    
    # check 8 points along the v2-v3 border
    v2v3_theta = np.array([1e-5 for _ in range(4)]+[np.pi+1e-5 for _ in range(4)])
    v2v3_radius = np.array([i for i in range(1, 5)]*2)

    v2x, v2y = map.from_dva()['v2'](*pol2cart(v2v3_theta, v2v3_radius))
    v3x, v3y = map.from_dva()['v3'](*pol2cart(v2v3_theta, v2v3_radius))

    npt.assert_almost_equal(v2x, v3x, 1)
    npt.assert_almost_equal(v2y, v3y, 1)
        
    # check 8 points along the v1-v2 border
    v1v2_theta = np.array([np.pi / 2 for _ in range(4)] + [-np.pi / 2 for _ in range(4)])
    v1v2_radius = np.array([i for i in range(1, 5)]*2)

    v1x, v1y = map.from_dva()['v1'](*pol2cart(v1v2_theta, v1v2_radius))
    v2x, v2y = map.from_dva()['v2'](*pol2cart(v1v2_theta, v1v2_radius))
    
    npt.assert_almost_equal(v1x, v2x, 1)
    npt.assert_almost_equal(v1y, v2y, 1)
    