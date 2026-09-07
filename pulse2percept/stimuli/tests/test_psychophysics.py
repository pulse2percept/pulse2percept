import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p
from pulse2percept.stimuli import GratingStimulus, BarStimulus, psychophysics
from pulse2percept.units import (DimensionMismatchError, deg, dimensionless,
                                 dva, ms, rad, uA)
from pulse2percept.units import s as sec
from pulse2percept.vision import Scene


def test_GratingStimulus():
    shape = (5, 5)
    grating = GratingStimulus(shape, spatial_freq=0.25)
    npt.assert_equal(grating.shape, (np.prod(shape), 51))
    npt.assert_equal(grating.vid_shape, (shape[0], shape[1], 51))
    npt.assert_almost_equal(grating.data.min(), 0)
    npt.assert_almost_equal(grating.data.max(), 1)

    # Drifting to the left/right:
    nx = 3
    for direction in [0, 180]:
        # A grating with 1-px white bar, drifting 1 column per frame:
        grating = GratingStimulus((nx, nx), direction=direction,
                                  spatial_freq=1.0 / nx,
                                  temporal_freq=1.0 / nx, time=np.arange(nx))
        data = grating.data.reshape(grating.vid_shape)
        for i in range(nx):
            if direction == 0:
                npt.assert_almost_equal(data[:, i, (i + 1) % nx], 1)
            else:
                npt.assert_almost_equal(data[:, nx - i - 1, i], 1)

    # Contrast vs. mask:
    for mask in ['circle', None]:
        # Mask will have value 0.5, so contrast still defines the min/max:
        grating = GratingStimulus(shape, spatial_freq=0.25, contrast=0.45,
                                  mask=mask)
        npt.assert_almost_equal(grating.data.max() - grating.data.min(), 0.45)
        npt.assert_almost_equal(grating.data.min(), 0.275)
        npt.assert_almost_equal(grating.data.max(), 0.725)

    # Masks:
    for mask in ['circle', 'gauss']:
        grating = GratingStimulus(shape, mask=mask)
        npt.assert_almost_equal(grating.data[:2, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(grating.data[3:5, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(grating.data[-2:, :].ravel(), 0.5, decimal=2)


def test_BarStimulus():
    shape = (15, 15)
    bar = BarStimulus(shape)
    npt.assert_equal(bar.shape, (np.prod(shape), 51))
    npt.assert_equal(bar.vid_shape, (shape[0], shape[1], 51))
    npt.assert_almost_equal(bar.data.max(), 1)

    # Contrast vs. mask:
    for mask in ['circle', 'gauss', None]:
        # Mask will have value 0.5, so contrast still defines the min/max:
        bar = BarStimulus(shape, contrast=0.45, mask=mask)
        npt.assert_almost_equal(bar.data.max() - bar.data.min(), 0.45,
                                decimal=2)
        npt.assert_almost_equal(bar.data.min(), 0.275, decimal=2)
        npt.assert_almost_equal(bar.data.max(), 0.725, decimal=2)

    # Masks:
    for mask in ['circle', 'gauss']:
        bar = BarStimulus(shape, mask=mask)
        npt.assert_almost_equal(bar.data[:2, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(bar.data[3:5, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(bar.data[-2:, :].ravel(), 0.5, decimal=2)


def test_psychophysics_time_units():
    # `time` is a duration, so it may be given as one:
    for cls, kwargs in [(GratingStimulus, {}), (BarStimulus, {})]:
        bare = cls((4, 4), time=100, **kwargs)
        unitful = cls((4, 4), time=0.1 * sec, **kwargs)
        npt.assert_array_equal(bare.data, unitful.data)
        npt.assert_array_equal(bare.time, unitful.time)
        # An explicit list of time points works too:
        listed = cls((4, 4), time=[0, 20, 40] * ms, **kwargs)
        npt.assert_almost_equal(listed.time, [0, 20, 40])
        # These are visual stimuli: their pixels are gray levels, but their
        # time is still physical.
        npt.assert_equal(unitful.unit, dimensionless)
        npt.assert_equal(unitful.time_unit, ms)
        with pytest.raises(DimensionMismatchError):
            cls((4, 4), time=5 * uA, **kwargs)


def test_GratingStimulus_angle_units():
    """`direction` and `phase` are ordinary angles, not visual angle"""
    bare = GratingStimulus((4, 4), direction=45, phase=90, time=100)
    unitful = GratingStimulus((4, 4), direction=45 * deg, phase=90 * deg,
                              time=100)
    in_rad = GratingStimulus((4, 4), direction=np.pi / 4 * rad,
                             phase=np.pi / 2 * rad, time=100)
    npt.assert_allclose(unitful.data, bare.data, rtol=1e-12)
    npt.assert_allclose(in_rad.data, bare.data, rtol=1e-12)
    for kwargs in ({'direction': 10 * dva}, {'phase': 10 * dva},
                   {'phase': 10 * ms}):
        with pytest.raises(DimensionMismatchError):
            GratingStimulus((4, 4), time=100, **kwargs)


def test_psychophysics_namespace():
    # The module is reachable, but the optotype generators are not promoted to
    # the top-level namespace:
    npt.assert_equal(p2p.stimuli.psychophysics is psychophysics, True)
    for name in ('landolt_c', 'tumbling_e'):
        npt.assert_equal(hasattr(psychophysics, name), True)
        npt.assert_equal(hasattr(p2p.stimuli, name), False)


def _ink(scene):
    """Visual-field coordinates of the inked pixels, as ``(x, y)`` arrays"""
    img = scene.source.data.reshape(scene.source.img_shape)
    rows, cols = np.where(img < 0.5)
    return scene.pixel_to_dva(cols, rows)


def test_landolt_c():
    scene = psychophysics.landolt_c(gap=0.5 * dva, fov=15 * dva,
                                    shape=(512, 512))
    npt.assert_equal(isinstance(scene, Scene), True)
    npt.assert_equal(scene.shape, (512, 512))
    npt.assert_equal(scene.fov, (15.0, 15.0))
    npt.assert_equal(np.unique(scene.source.data).tolist(), [0.0, 1.0])
    meta = scene.source.metadata
    npt.assert_equal(meta['generator'], 'landolt_c')
    npt.assert_almost_equal(meta['gap'], 0.5)
    npt.assert_equal(meta['position'], (0.0, 0.0))
    npt.assert_almost_equal(meta['orientation'], 0.0)
    npt.assert_equal(meta['polarity'], 'dark')
    # A scalar fov is the horizontal one; the vertical follows from `shape`:
    npt.assert_equal(psychophysics.landolt_c(fov=10, shape=(256, 512)).fov,
                     (10.0, 5.0))


def test_landolt_c_geometry():
    gap, fov, shape = 1.0, 20.0, (512, 512)
    # One pixel of slack in each direction, since the extent is measured
    # between the centers of the outermost lit pixels:
    tol = 2 * fov / shape[0]
    scene = psychophysics.landolt_c(gap=gap, fov=fov, shape=shape)
    x, y = _ink(scene)
    # Outer diameter is 5 gaps, in both directions:
    npt.assert_almost_equal(x.max() - x.min(), 5 * gap, decimal=1)
    npt.assert_almost_equal(y.max() - y.min(), 5 * gap, decimal=1)
    npt.assert_array_less(abs(x.max() + x.min()), tol)
    npt.assert_array_less(abs(y.max() + y.min()), tol)
    # The stroke is one gap wide: with the opening pointing right, the column
    # through the center of the C crosses ink from -2.5 to -1.5 gaps:
    left = x[(abs(y) < tol) & (x < 0)]
    npt.assert_almost_equal(left.max() - left.min(), gap, decimal=1)
    # ... and the opening is one gap wide, measured across the gap direction:
    npt.assert_equal(np.any((x > 0) & (abs(y) < gap / 2)), False)
    # Ink resumes right at the slot edge, so the opening is exactly a gap wide:
    edge = abs(y[(x > 1.5 * gap) & (x < 2.5 * gap)]).min()
    npt.assert_almost_equal(edge, gap / 2, decimal=1)


@pytest.mark.parametrize('orientation,direction', [
    (0, (1, 0)), (90, (0, 1)), (180, (-1, 0)), (270, (0, -1)),
    # An arbitrary angle, and one wrapped past a full turn:
    (45, (np.sqrt(0.5), np.sqrt(0.5))), (360 + 90, (0, 1)),
])
def test_landolt_c_orientation(orientation, direction):
    gap = 1.0
    scene = psychophysics.landolt_c(gap=gap, orientation=orientation * deg,
                                    fov=20, shape=(512, 512))
    img = scene.source.data.reshape(scene.source.img_shape)
    rows, cols = np.where(img > 0.5)
    x, y = scene.pixel_to_dva(cols, rows)
    # The background pixels that fall inside the annulus are the opening, and
    # they sit on the side the orientation points to:
    radius = np.hypot(x, y)
    inside = (radius > 1.5 * gap) & (radius < 2.5 * gap)
    npt.assert_almost_equal([x[inside].mean() / radius[inside].mean(),
                             y[inside].mean() / radius[inside].mean()],
                            direction, decimal=2)


def test_landolt_c_position():
    gap, position = 0.5, (5.0, -3.0)
    scene = psychophysics.landolt_c(gap=gap, position=position * dva,
                                    fov=20 * dva, shape=(512, 512))
    x, y = _ink(scene)
    npt.assert_almost_equal([(x.min() + x.max()) / 2,
                             (y.min() + y.max()) / 2], position, decimal=1)
    # Eccentricity changes, angular size does not:
    npt.assert_almost_equal(x.max() - x.min(), 5 * gap, decimal=1)
    npt.assert_equal(scene.source.metadata['position'], position)


def test_landolt_c_polarity():
    kwargs = dict(gap=1, orientation=30 * deg, fov=12, shape=(128, 128))
    dark = psychophysics.landolt_c(polarity='dark', **kwargs).source.data
    light = psychophysics.landolt_c(polarity='light', **kwargs).source.data
    npt.assert_almost_equal(light, 1.0 - dark)


@pytest.mark.parametrize('kwargs,msg', [
    (dict(gap=0), "'gap'"),
    (dict(gap=-1), "'gap'"),
    (dict(gap=np.inf), "'gap'"),
    (dict(position=(0, 0, 0)), "'position'"),
    (dict(position=(np.nan, 0)), "'position'"),
    (dict(orientation=np.nan), "'orientation'"),
    (dict(polarity='inverted'), "'polarity'"),
    (dict(shape=(0, 10)), "'shape'"),
    (dict(shape=(10, 10, 10)), "'shape'"),
    (dict(shape=(10.5, 10)), "'shape'"),
    (dict(fov=0), "'fov'"),
    # The whole outer circle has to fit: 5 * gap = 5 dva across, centered 8 dva
    # out in a 10 dva field:
    (dict(gap=1, position=(8, 0), fov=10), 'half-FOV'),
    (dict(gap=1, position=(0, -8), fov=10), 'half-FOV'),
    # 0.1 dva across a 10-degree, 128-pixel frame is 1.3 pixels:
    (dict(gap=0.1, fov=10, shape=(128, 128)), 'resolve the opening'),
])
def test_landolt_c_invalid(kwargs, msg):
    with pytest.raises(ValueError) as excinfo:
        psychophysics.landolt_c(**kwargs)
    npt.assert_equal(msg in str(excinfo.value), True)


def _is_ink(scene, x, y):
    """Whether the pixel nearest visual-field point ``(x, y)`` is ink"""
    col, row = scene.dva_to_pixel(x, y)
    img = scene.source.data.reshape(scene.source.img_shape)
    return bool(img[int(round(float(row))), int(round(float(col)))] < 0.5)


def test_tumbling_e():
    scene = psychophysics.tumbling_e(stroke=0.5 * dva, fov=15 * dva,
                                     shape=(512, 512))
    npt.assert_equal(isinstance(scene, Scene), True)
    npt.assert_equal(scene.shape, (512, 512))
    npt.assert_equal(scene.fov, (15.0, 15.0))
    npt.assert_equal(np.unique(scene.source.data).tolist(), [0.0, 1.0])
    meta = scene.source.metadata
    npt.assert_equal(meta['generator'], 'tumbling_e')
    npt.assert_almost_equal(meta['stroke'], 0.5)
    npt.assert_equal(meta['position'], (0.0, 0.0))
    npt.assert_almost_equal(meta['orientation'], 0.0)
    npt.assert_equal(meta['polarity'], 'dark')
    npt.assert_equal(meta['fov'], (15.0, 15.0))
    # A scalar fov is the horizontal one; the vertical follows from `shape`:
    npt.assert_equal(psychophysics.tumbling_e(fov=10, shape=(256, 512)).fov,
                     (10.0, 5.0))


def test_tumbling_e_geometry():
    stroke, fov, shape = 1.0, 20.0, (512, 512)
    # One pixel of slack in each direction, since extents are measured between
    # the centers of the outermost inked pixels:
    tol = 2 * fov / shape[0]
    scene = psychophysics.tumbling_e(stroke=stroke, fov=fov, shape=shape)
    x, y = _ink(scene)
    # The 5 x 5 construction: the glyph is 5 strokes across, both ways, and
    # centered on the requested position:
    npt.assert_almost_equal(x.max() - x.min(), 5 * stroke, decimal=1)
    npt.assert_almost_equal(y.max() - y.min(), 5 * stroke, decimal=1)
    npt.assert_array_less(abs(x.max() + x.min()), tol)
    npt.assert_array_less(abs(y.max() + y.min()), tol)
    # A row through the middle of a gap crosses the spine only, which is one
    # stroke wide and ends 1.5 strokes left of center:
    spine = x[abs(y - stroke) < tol / 2]
    npt.assert_almost_equal(spine.max() - spine.min(), stroke, decimal=1)
    npt.assert_almost_equal(spine.max(), -1.5 * stroke, decimal=1)
    # A column through the free end of the bars crosses three one-stroke bars
    # separated by two one-stroke gaps:
    col = np.sort(y[abs(x - 2 * stroke) < tol / 2])
    jumps = np.diff(col)
    gaps = jumps[jumps > tol]
    npt.assert_equal(gaps.size, 2)
    npt.assert_almost_equal(gaps, [stroke, stroke], decimal=1)
    for bar in (col[col > 1.4 * stroke], col[abs(col) < 0.6 * stroke],
                col[col < -1.4 * stroke]):
        npt.assert_almost_equal(bar.max() - bar.min(), stroke, decimal=1)


@pytest.mark.parametrize('orientation', [0, 90, 180, 270, 45, 360 + 90, -30])
def test_tumbling_e_orientation(orientation):
    stroke = 1.0
    scene = psychophysics.tumbling_e(stroke=stroke, fov=20,
                                     orientation=orientation * deg,
                                     shape=(512, 512))
    # Probe points in the canonical right-facing frame, rotated by the
    # requested angle. Each sits a half stroke clear of a mask boundary.
    theta = np.deg2rad(orientation)

    def probe(u, v):
        return (u * np.cos(theta) - v * np.sin(theta),
                u * np.sin(theta) + v * np.cos(theta))

    # The bars reach the free end, the gaps beside the middle one do not, and
    # the spine runs the full height behind them:
    npt.assert_equal(_is_ink(scene, *probe(2 * stroke, 0)), True)
    npt.assert_equal(_is_ink(scene, *probe(2 * stroke, 2 * stroke)), True)
    npt.assert_equal(_is_ink(scene, *probe(2 * stroke, stroke)), False)
    npt.assert_equal(_is_ink(scene, *probe(2 * stroke, -stroke)), False)
    npt.assert_equal(_is_ink(scene, *probe(-2 * stroke, stroke)), True)
    npt.assert_equal(_is_ink(scene, *probe(-2 * stroke, 2 * stroke)), True)
    # Outside the glyph:
    npt.assert_equal(_is_ink(scene, *probe(3 * stroke, 0)), False)


@pytest.mark.parametrize('orientation,direction', [
    (0, (1, 0)), (90, (0, 1)), (180, (-1, 0)), (270, (0, -1)),
])
def test_tumbling_e_gap_side(orientation, direction):
    # The two gaps are the only background inside the glyph's bounding box,
    # and they sit on the side the bars point to:
    stroke = 1.0
    scene = psychophysics.tumbling_e(stroke=stroke, fov=20,
                                     orientation=orientation * deg,
                                     shape=(512, 512))
    img = scene.source.data.reshape(scene.source.img_shape)
    rows, cols = np.where(img > 0.5)
    x, y = scene.pixel_to_dva(cols, rows)
    inside = (abs(x) < 2.5 * stroke) & (abs(y) < 2.5 * stroke)
    centroid = np.array([x[inside].mean(), y[inside].mean()])
    along = np.asarray(direction, dtype=float)
    across = np.asarray([-direction[1], direction[0]], dtype=float)
    # The gaps span 4 strokes of the 5, offset a half stroke toward the bars:
    npt.assert_almost_equal(centroid @ along, 0.5 * stroke, decimal=1)
    npt.assert_almost_equal(centroid @ across, 0, decimal=1)


def test_tumbling_e_position():
    stroke, position = 0.5, (5.0, -3.0)
    scene = psychophysics.tumbling_e(stroke=stroke, position=position * dva,
                                     fov=20 * dva, shape=(512, 512))
    x, y = _ink(scene)
    npt.assert_almost_equal([(x.min() + x.max()) / 2,
                             (y.min() + y.max()) / 2], position, decimal=1)
    # Eccentricity changes, angular size does not:
    npt.assert_almost_equal(x.max() - x.min(), 5 * stroke, decimal=1)
    npt.assert_almost_equal(y.max() - y.min(), 5 * stroke, decimal=1)
    npt.assert_equal(scene.source.metadata['position'], position)


def test_tumbling_e_polarity():
    kwargs = dict(stroke=1, orientation=30 * deg, fov=12, shape=(128, 128))
    dark = psychophysics.tumbling_e(polarity='dark', **kwargs).source.data
    light = psychophysics.tumbling_e(polarity='light', **kwargs).source.data
    npt.assert_almost_equal(light, 1.0 - dark)


def test_tumbling_e_units():
    # Plain numbers follow the dva/degree conventions, so they have to agree
    # with the unit-aware call:
    plain = psychophysics.tumbling_e(stroke=0.5, position=(2, -1),
                                     orientation=90, fov=(12, 12),
                                     shape=(128, 128))
    quantity = psychophysics.tumbling_e(stroke=0.5 * dva,
                                        position=(2, -1) * dva,
                                        orientation=90 * deg,
                                        fov=(12, 12) * dva,
                                        shape=(128, 128))
    npt.assert_almost_equal(plain.source.data, quantity.source.data)
    npt.assert_equal(plain.fov, quantity.fov)


@pytest.mark.parametrize('kwargs,msg', [
    (dict(stroke=0), "'stroke'"),
    (dict(stroke=-1), "'stroke'"),
    (dict(stroke=np.inf), "'stroke'"),
    (dict(position=(0, 0, 0)), "'position'"),
    (dict(position=(np.nan, 0)), "'position'"),
    (dict(orientation=np.nan), "'orientation'"),
    (dict(polarity='inverted'), "'polarity'"),
    (dict(shape=(0, 10)), "'shape'"),
    (dict(shape=(10, 10, 10)), "'shape'"),
    (dict(shape=(10.5, 10)), "'shape'"),
    (dict(fov=0), "'fov'"),
    # The whole 5 x 5 dva square has to fit, and this one is centered 8 dva
    # out in a 10 dva field:
    (dict(stroke=1, position=(8, 0), fov=10), 'half-FOV'),
    (dict(stroke=1, position=(0, -8), fov=10), 'half-FOV'),
    # Rotated 45 deg, the square's axis-aligned extent grows to
    # 2.5 * sqrt(2) = 3.54 dva, past the 3.5 dva half-FOV the same E clears
    # upright:
    (dict(stroke=1, orientation=45, fov=7), 'half-FOV'),
    # 0.1 dva across a 10-degree, 128-pixel frame is 1.3 pixels:
    (dict(stroke=0.1, fov=10, shape=(128, 128)), 'resolve the bars'),
])
def test_tumbling_e_invalid(kwargs, msg):
    with pytest.raises(ValueError) as excinfo:
        psychophysics.tumbling_e(**kwargs)
    npt.assert_equal(msg in str(excinfo.value), True)


def test_tumbling_e_fits_snugly():
    # Upright, the bound is exactly 5 * stroke; the rotated-extent check must
    # not tighten that, nor loosen the 45-degree one:
    npt.assert_equal(psychophysics.tumbling_e(stroke=1, fov=5.2).fov,
                     (5.2, 5.2))
    npt.assert_equal(psychophysics.tumbling_e(stroke=1, orientation=45,
                                              fov=7.5).fov, (7.5, 7.5))
