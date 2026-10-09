import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.topography.cortex import Schira2010Map
from pulse2percept.units import Quantity, dva, mm, um

# Protocol S1 demo parameters (DemoSchiraEtal2009.m, magnification.m):
S1_PARAMS = dict(k=18, a=0.75, b=90, alpha1=1, alpha2=0.5, alpha3=0.4)

# Protocol S1 output, from the unmodified assembleV1V3Complex.m and
# bandedDoubleSech.m (Octave 11.3), times K. Units are mm in the authors'
# frame: right visual hemifield, upper field at +y. Rows are eccentricity
# (dva), polar angle (deg; 0 = horizontal meridian, + = upper), then (x, y)
# for V1, V2, V3.
S1_POINTS = np.array([
    [0.5, 30], [0.5, -60], [1, 0], [1.5, 90], [2, -90], [2, 45], [5, -20],
    [7.5, 70], [10, -45], [10, 0], [40, 15], [60, -75],
])
S1_BANDED = {  # shiftAmount = 0.4 (Protocol S1 default)
    'fovea': -78.560681850413,
    'v1': [[-72.669889622558, 2.747664216253],
           [-74.200494032569, -5.148719107306],
           [-67.495997453770, 0.000000000000],
           [-68.880000802458, 14.231949246978],
           [-65.358448227668, -16.219155664107],
           [-61.978622627584, 8.516157443264],
           [-49.461138998805, -4.767238542037],
           [-44.262746801418, 16.955126165393],
           [-39.583436599867, -11.126996588659],
           [-39.559008768327, 0.000000000000],
           [-20.647523066030, 3.123216516807],
           [-13.390356080849, -12.422823826175]],
    'v2': [[-81.815589907819, 8.079236109471],
           [-79.222466569218, -7.555563711727],
           [-79.138960722499, 16.027616418033],
           [-68.880000802458, 14.231949246978],
           [-65.358448227668, -16.219155664107],
           [-67.733833900370, 20.140440720454],
           [-52.735988035576, -28.494537321639],
           [-44.807872733547, 23.859399036650],
           [-39.745627587268, -26.752065126915],
           [-39.637441520711, 31.569066509714],
           [-15.338390188028, 22.579989844800],
           [-11.906216174689, -15.358580654078]],
    'v3': [[-85.742628534051, 9.158502710871],
           [-86.746456231381, -10.149134180055],
           [-79.138960722499, 16.027616418033],
           [-75.553472772557, 25.888054001662],
           [-70.830470588683, -29.637174005477],
           [-70.450546389039, 26.847268845701],
           [-53.088765448459, -32.226975410871],
           [-45.091580327245, 37.143816485984],
           [-39.454383901149, -35.161952981386],
           [-39.637441520711, 31.569066509714],
           [-14.536654928526, 24.343917397466],
           [-7.357417247249, -21.994199561970]],
}
S1_UNBANDED = {  # shiftAmount = 0
    'fovea': -86.174851370077,
    'v1': [[-77.637282339990, 3.613096707125],
           [-79.107058761926, -6.703365687206],
           [-71.122386934466, 0.000000000000],
           [-70.228899771969, 16.370114890310],
           [-66.237579883122, -18.109586627099],
           [-64.094497131063, 9.677052758222],
           [-50.552867330333, -5.111927479427],
           [-44.587205943351, 17.617325439250],
           [-40.014826455296, -11.529130013264],
           [-40.144759765460, 0.000000000000],
           [-20.762275710423, 3.162745910744],
           [-13.386374642512, -12.483520614854]],
    'v2': [[-83.290298901069, 10.759820572042],
           [-82.186520344616, -9.996199199660],
           [-78.296629961353, 18.192161087538],
           [-70.228899771969, 16.370114890310],
           [-66.237579883122, -18.109586627099],
           [-67.476179619436, 21.803410598525],
           [-52.278004977735, -29.051402709244],
           [-44.692375310322, 24.457396540912],
           [-39.532410132134, -27.082352782793],
           [-39.364387722396, 31.715264736520],
           [-15.269851276875, 22.583238677983],
           [-11.873063989455, -15.394828095241]],
    'v3': [[-85.238126502061, 11.781872429142],
           [-86.060735655289, -12.110299090836],
           [-78.296629961353, 18.192161087538],
           [-75.165162224685, 26.262921903187],
           [-70.496633159057, -29.898336153983],
           [-69.637420109860, 27.638393388079],
           [-52.608247922444, -32.543209001766],
           [-44.899755786942, 37.182661592557],
           [-39.242645341896, -35.205571528153],
           [-39.364387722396, 31.715264736520],
           [-14.476269152744, 24.328983864566],
           [-7.342779036872, -21.981624318367]],
}
# Protocol S1 V2 and V3 at eccentricity 0 for polar angles 90, 45, 0, -45,
# -90 deg (x in mm; y is 0). The fovea is a band, not a point:
S1_BAND_X = {
    'v2': [-78.560681850413, -80.178251332563, -81.959808986210,
           -80.178251332563, -78.560681850413],
    'v3': [-85.247574292073, -83.527275292156, -81.959808986210,
           -83.527275292156, -85.247574292073],
}


def _to_p2p(xy_mm, fovea_mm, left_offset, left_field):
    """Moves Protocol S1 output into p2p coordinates (um)

    Rigid only: V1 fovea to the origin, upper field to -y, and the right
    visual field mirrored into the left hemisphere.
    """
    x = 1000 * (np.asarray(xy_mm)[:, 0] - fovea_mm)
    y = -1000 * np.asarray(xy_mm)[:, 1]
    return (x, y) if left_field else (left_offset - x, y)


@pytest.mark.parametrize('lambda_, ref', [(0.4, S1_BANDED),
                                          (0, S1_UNBANDED)])
@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
@pytest.mark.parametrize('left_field', [True, False])
def test_schira2010_protocol_s1(region, lambda_, ref, left_field):
    vfmap = Schira2010Map(lambda_=lambda_, regions=[region], **S1_PARAMS)
    ecc, theta = S1_POINTS[:, 0], np.deg2rad(S1_POINTS[:, 1])
    x = ecc * np.cos(theta) * (-1 if left_field else 1)
    xum, yum = vfmap.from_dva()[region](x, ecc * np.sin(theta))
    expected = _to_p2p(ref[region], ref['fovea'], vfmap.left_offset,
                       left_field)
    npt.assert_allclose(xum, expected[0], atol=1e-6)
    npt.assert_allclose(yum, expected[1], atol=1e-6)


@pytest.mark.parametrize('region', ['v2', 'v3'])
def test_schira2010_foveal_band(region):
    """V2/V3 foveas are bands for lambda_ > 0, one point for lambda_ = 0"""
    vfmap = Schira2010Map(regions=[region], **S1_PARAMS)
    to_tissue = vfmap.from_dva()[region]
    # A point of the band per polar angle; (0, 0) has none, so is NaN:
    theta = np.deg2rad([90, 45, 1e-9, -45, -90])
    x, y = to_tissue(-1e-12 * np.cos(theta), 1e-12 * np.sin(theta))
    band = 1000 * (np.array(S1_BAND_X[region]) - S1_BANDED['fovea'])
    npt.assert_allclose(x, band, atol=1e-6)
    npt.assert_allclose(y, 0, atol=1e-6)
    npt.assert_equal(np.isnan(to_tissue(0, 0)), True)
    # Every point of the band is the fovea:
    npt.assert_allclose(vfmap.to_dva()[region](x, y), 0, atol=1e-6)
    # V1 has a single foveal tip:
    npt.assert_allclose(Schira2010Map(**S1_PARAMS).dva_to_v1(0, 0),
                        (vfmap.left_offset, 0), atol=1e-9)
    # Without banding, all foveas are the V1 foveal tip:
    flat = Schira2010Map(lambda_=0, regions=[region], **S1_PARAMS)
    npt.assert_allclose(flat.from_dva()[region](0, 0),
                        (flat.left_offset, 0), atol=1e-9)
    npt.assert_allclose(flat.from_dva()[region](
        -1e-12 * np.cos(theta), 1e-12 * np.sin(theta)), 0, atol=1e-3)


def test_schira2010_defaults():
    """Defaults are the published parameters; alphas and k from Protocol S1"""
    vfmap = Schira2010Map()
    npt.assert_equal((vfmap.a, vfmap.b, vfmap.lambda_), (1.05, 90, 0.4))
    npt.assert_equal((vfmap.k, vfmap.alpha1, vfmap.alpha2, vfmap.alpha3),
                     (18, 1, 0.5, 0.4))
    npt.assert_equal(vfmap.regions, ['v1'])
    npt.assert_equal(vfmap.get_param_units()['k'], mm)
    for name in ('a', 'b', 'lambda_'):
        npt.assert_equal(vfmap.get_param_units()[name], dva)
    npt.assert_equal(vfmap.get_param_units()['left_offset'], um)
    for name in ('alpha1', 'alpha2', 'alpha3'):
        npt.assert_equal(name in vfmap.get_param_units(), False)
    # Unitful parameters are stored in their units:
    npt.assert_almost_equal(Schira2010Map(k=18000 * um).k, 18)


def test_schira2010_shapes():
    """Scalars return scalars; arrays keep their shape"""
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'])
    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -0.5]])
    y = np.array([[0.5, 2.0, -1.0], [3.0, -6.0, 0.2]])
    for region in ('v1', 'v2', 'v3'):
        to_tissue = vfmap.from_dva()[region]
        to_visual = vfmap.to_dva()[region]
        xs, ys = to_tissue(5, -2)
        npt.assert_equal(np.ndim(xs) == 0 and np.ndim(ys) == 0, True)
        npt.assert_almost_equal((xs, ys),
                                np.ravel(to_tissue([5.0], [-2.0])))
        xc, yc = to_tissue(x, y)
        npt.assert_equal((xc.shape, yc.shape), (x.shape, y.shape))
        xb, yb = to_visual(xc, yc)
        npt.assert_equal((xb.shape, yb.shape), (x.shape, y.shape))
        npt.assert_equal(np.ndim(to_visual(xs, ys)[0]), 0)
        # Integer inputs match float inputs:
        npt.assert_almost_equal(to_tissue(5, -2), to_tissue(5.0, -2.0))
        npt.assert_almost_equal(to_visual(int(xs), int(ys)),
                                to_visual(float(int(xs)), float(int(ys))))


@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_schira2010_hemispheres(region):
    """Left field to right hemisphere; upper field to lower cortex"""
    vfmap = Schira2010Map(regions=[region])
    to_tissue = vfmap.from_dva()[region]
    x = np.array([3.0, 0.5, 10.0, 7.0])
    y = np.array([2.0, -1.0, 4.0, -9.0])
    xl, yl = to_tissue(-x, y)
    xr, yr = to_tissue(x, y)
    npt.assert_equal(xl > vfmap.left_offset / 2, True)
    npt.assert_equal(xr < vfmap.left_offset / 2, True)
    # The hemispheres mirror about x = left_offset / 2:
    npt.assert_allclose(xr, vfmap.left_offset - xl)
    npt.assert_allclose(yr, yl)
    npt.assert_equal(np.sign(yl), -np.sign(y))
    # The vertical meridian goes to the left hemisphere:
    npt.assert_equal(to_tissue(0, 5)[0] < vfmap.left_offset / 2, True)
    npt.assert_allclose(to_tissue(-0.0, 5), to_tissue(0.0, 5))


def test_schira2010_mirroring():
    """V2 is mirrored relative to V1 and V3"""
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'])
    # A is above B in the visual field:
    x, y = [-1.5, -2.6], [2.6, 1.5]
    npt.assert_equal(np.diff(vfmap.dva_to_v1(x, y)[1]) > 0, True)
    npt.assert_equal(np.diff(vfmap.dva_to_v2(x, y)[1]) < 0, True)
    npt.assert_equal(np.diff(vfmap.dva_to_v3(x, y)[1]) > 0, True)


@pytest.mark.parametrize('lambda_', [0.4, 0])
def test_schira2010_continuity(lambda_):
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'], lambda_=lambda_)
    ecc = np.array([0.1, 1, 5, 20, 80])
    # V1 and V2 share the vertical meridians:
    for sign in (1, -1):
        x, y = -ecc * 1e-12, sign * ecc
        npt.assert_allclose(vfmap.dva_to_v1(x, y), vfmap.dva_to_v2(x, y),
                            atol=1e-6)
    # V2 and V3 share both copies of the horizontal meridian:
    for theta in (0, -1e-12):
        x, y = -ecc * np.cos(theta), ecc * np.sin(theta)
        npt.assert_allclose(vfmap.dva_to_v2(x, y), vfmap.dva_to_v3(x, y),
                            atol=1e-6)
    # The two V2 copies of the horizontal meridian are distinct:
    upper = vfmap.dva_to_v2(-ecc, 0 * ecc)[1]
    lower = vfmap.dva_to_v2(-ecc, -1e-12 * ecc)[1]
    npt.assert_equal(upper < 0, True)
    npt.assert_equal(lower > 0, True)


def test_schira2010_scale():
    """Cortical distances from the V1 foveal tip scale with k"""
    x, y = np.array([-0.5, -3.0, -20.0]), np.array([0.2, -2.0, 15.0])
    for region in ('v1', 'v2', 'v3'):
        small = Schira2010Map(k=15, regions=[region]).from_dva()[region]
        large = Schira2010Map(k=30, regions=[region]).from_dva()[region]
        npt.assert_allclose(large(x, y), 2 * np.array(small(x, y)),
                            rtol=1e-12)


@pytest.mark.parametrize('lambda_', [0.4, 0])
@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_schira2010_forward_inverse(region, lambda_):
    vfmap = Schira2010Map(regions=[region], lambda_=lambda_)
    ecc, theta = np.meshgrid([0.05, 0.3, 1, 3, 10, 30, 89],
                             np.deg2rad([-90, -60, -20, -1, 1, 45, 90]))
    ecc, theta = ecc.ravel(), theta.ravel()
    for sign in (-1, 1):
        x, y = sign * ecc * np.cos(theta), ecc * np.sin(theta)
        xb, yb = vfmap.to_dva()[region](*vfmap.from_dva()[region](x, y))
        # Magnification falls with eccentricity, so dva error grows:
        npt.assert_array_less(np.hypot(xb - x, yb - y), 1e-6 * ecc + 1e-8)
    if region == 'v1':
        # Both V1 foveal tips:
        npt.assert_allclose(vfmap.v1_to_dva([0, vfmap.left_offset], [0, 0]),
                            0, atol=1e-6)


@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_schira2010_inverse_forward(region):
    """Tissue the inverse accepts maps back to itself within 1 um"""
    vfmap = Schira2010Map(regions=[region])
    gx, gy = np.meshgrid(np.linspace(-15000, 70000, 18),
                         np.linspace(-45000, 45000, 18))
    xd, yd = vfmap.to_dva()[region](gx, gy)
    inside = np.isfinite(xd)
    npt.assert_equal(np.isfinite(yd), inside)
    npt.assert_equal(0 < inside.sum() < inside.size, True)
    xb, yb = vfmap.from_dva()[region](xd[inside], yd[inside])
    npt.assert_allclose(np.hypot(xb - gx[inside], yb - gy[inside]), 0,
                        atol=1)


def test_schira2010_inverse_outside():
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'])
    # Interior points of each area (left field, 45 deg, 2 and 10 dva):
    x, y = np.array([-1.4, -7.1]), np.array([1.4, -7.1])
    tissue = {r: vfmap.from_dva()[r](x, y) for r in ('v1', 'v2', 'v3')}
    for region in ('v1', 'v2', 'v3'):
        for other in ('v1', 'v2', 'v3'):
            back = vfmap.to_dva()[region](*tissue[other])
            if region == other:
                npt.assert_allclose(back, (x, y), atol=1e-6)
            else:
                npt.assert_equal(np.isnan(back), True,
                                 err_msg=f'{other} in {region}')
    # Beyond 90 dva, between hemispheres, and far off the map:
    x = [vfmap.dva_to_v1(-89.9, 0)[0] + 2000, -9000, 30000]
    y = [0, 0, 80000]
    for region in ('v1', 'v2', 'v3'):
        npt.assert_equal(np.isnan(vfmap.to_dva()[region](x, y)), True)


def test_schira2010_nans():
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'])
    for region in ('v1', 'v2', 'v3'):
        x, y = vfmap.from_dva()[region]([np.nan, 2, 100], [1, np.nan, 0])
        npt.assert_equal(np.isnan(x), True)
        npt.assert_equal(np.isnan(y), True)
        x, y = vfmap.to_dva()[region]([np.nan, 5000], [0, np.nan])
        npt.assert_equal(np.isnan(x), True)
        npt.assert_equal(np.isnan(y), True)


def test_schira2010_units():
    vfmap = Schira2010Map(regions=['v1', 'v2', 'v3'])
    xdva, ydva = np.array([-5.0, 2.0]), np.array([-2.0, 3.0])
    for region in ('v1', 'v2', 'v3'):
        to_tissue = getattr(vfmap, f'dva_to_{region}')
        to_visual = getattr(vfmap, f'{region}_to_dva')
        x_um, y_um = to_tissue(xdva, ydva)
        npt.assert_allclose(to_tissue(xdva * dva, ydva * dva), (x_um, y_um))
        back = to_visual((x_um / 1000) * mm, y_um * um)
        npt.assert_allclose(back, (xdva, ydva), atol=1e-6)
        npt.assert_equal(isinstance(back[0], Quantity), False)


def test_schira2010_set_params():
    """The inverse follows parameters changed after construction"""
    vfmap = Schira2010Map()
    vfmap.v1_to_dva(*vfmap.dva_to_v1(-3, 2))
    vfmap.k = 25
    npt.assert_allclose(vfmap.v1_to_dva(*vfmap.dva_to_v1(-3, 2)), (-3, 2),
                        atol=1e-6)


def test_schira2010_plot_mm_ticks():
    fig, ax = plt.subplots()
    Schira2010Map().plot(ax=ax)
    npt.assert_equal(len(ax.lines) > 0, True)
    ax.set_xlim(-10000, 60000)
    ax.set_ylim(-5000, 5000)
    fig.canvas.draw()
    for axis in (ax.xaxis, ax.yaxis):
        labels = [float(t.get_text().replace('−', '-'))
                  for t in axis.get_ticklabels()]
        npt.assert_allclose(labels, axis.get_majorticklocs() / 1000)
    plt.close(fig)
