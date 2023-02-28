from pulse2percept.topography import Polimeni2006Map
import numpy as np
import numpy.testing as npt


def test_polimeni_v1():
    map = Polimeni2006Map(regions=['v1'])
    theta = np.array([(i * np.pi - 0.1) / 6 for i in range(-5, 7)])
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v1'](theta, radius)

    expected_x = [
        15902.589306126785, 22701.701209282972, 27105.66605223716,
        -31823.503877716503, -35086.034551492055, -37506.13668631103,
        -39506.3130414652, -41373.30525591315, -43293.183557664786,
        44457.872804343075, 45423.62865236582, 46405.82649122574
    ]
    expected_y = [
        -5013.8355343520525, -12359.443849528407, -20342.321611408766,
        -13847.746737780984, -6984.86446795094, -215.14551955541478,
        6584.422109361202, 13588.223582967372, 20987.998282780045,
        13939.6205118376, 6913.876163034446, 210.58956294736032
    ]

    # check that the conversion from dva to v1 was correct
    npt.assert_almost_equal(x, expected_x)
    npt.assert_almost_equal(y, expected_y)

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

    expected_x = [
        9689.877814630294, 20645.517280735134, 27062.815960747776,
        -31088.965767776444, -34358.02828705147, -37160.94659139381,
        -39613.0489766391, -41630.58330730293, -43307.69628722014,
        45017.75586687664, 46671.23009322621, 48277.67564932234
    ]
    expected_y = [
        -20977.926765752603, -22151.218922100634, -20665.850562060903,
        -23554.244234724538, -26453.386814786292, -29296.33682195241,
        26724.6759835154, 23999.446003854897, 21316.642282519246,
        23681.752491457257, 26265.911354121294, 28919.10673669272
    ]

    # check that the conversion from dva to v2 was correct
    npt.assert_almost_equal(x, expected_x)
    npt.assert_almost_equal(y, expected_y)

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
    radius = np.array([i for i in range(1, 13)])

    x, y = map.from_dva()['v3'](theta, radius)

    expected_x = [
        7208.613020331824, 18466.684974346612, 25257.495476836517,
        -30340.83769587597, -34148.17463378478, -37158.5303860779,
        -39641.052094685976, -41865.30148649867, -43893.55339523437,
        45582.11723185931, 47036.96603224987, 48291.23011297499
    ]
    expected_y = [
        -25395.771704208673, -31847.85547750596, -35166.10649079788,
        -33467.660650388934, -31491.043510450312, -29452.02958700397,
        31445.110291796, 33567.49807200242, 35706.33963963459,
        33578.30210820712, 31317.5638427424, 29075.77983553856
    ]

    # check that the conversion from dva to v3 was correct
    npt.assert_almost_equal(x, expected_x)
    npt.assert_almost_equal(y, expected_y)

    result_theta, result_radius = map.to_dva()['v3'](x, y)

    # check that the inverse functions work correctly
    npt.assert_almost_equal(theta, result_theta)
    npt.assert_almost_equal(radius, result_radius)

    # check that points in the left visual field correspond to
    # the right hemisphere and vice versa
    sign_match = ((x > 0) & ((theta > (np.pi / 2)) | (theta < - (np.pi / 2)))) \
        | ((x < 0) & ((theta <= (np.pi / 2)) & (theta >= - (np.pi / 2))))
    npt.assert_equal(sign_match, True)

