import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p
from pulse2percept.stimuli import (BarStimulus, GratingStimulus,
                                   ImageStimulus, VideoStimulus,
                                   psychophysics)
from pulse2percept.units import (DimensionMismatchError, deg, dimensionless,
                                 dva, Hz, ms, rad, uA)
from pulse2percept.units import s as sec
from pulse2percept.vision import Scene


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
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
    for name in ('bar', 'grating', 'landolt_c', 'tumbling_e'):
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


def _raster(scene):
    """The scene's source as a ``(rows, cols, frames)`` array"""
    source = scene.source
    if isinstance(source, ImageStimulus):
        return source.data.reshape(source.img_shape)[..., np.newaxis]
    return source.data.reshape(source.vid_shape)


def _at(scene, x, y, frame=0):
    """Gray level of the pixel nearest the visual-field coordinate (x, y)"""
    col, row = scene.dva_to_pixel(x, y)
    return float(_raster(scene)[int(round(float(row))), int(round(float(col))),
                                frame])


def _lit(scene, frame=0):
    """Visual-field coordinates of the above-mean-gray pixels of a frame"""
    rows, cols = np.where(_raster(scene)[..., frame] > 0.5)
    return scene.pixel_to_dva(cols, rows)


def _peak_x(scene, frame=0):
    """x of the brightest pixel of the center row, in dva"""
    img = _raster(scene)[..., frame]
    row = img.shape[0] // 2
    col = int(np.argmax(img[row]))
    x, _ = scene.pixel_to_dva(col, row)
    return float(x)


def test_grating_static():
    scene = psychophysics.grating(spatial_freq=1 / dva, fov=10 * dva,
                                  shape=(128, 128))
    npt.assert_equal(isinstance(scene, Scene), True)
    npt.assert_equal(isinstance(scene.source, ImageStimulus), True)
    npt.assert_equal(isinstance(scene.source, VideoStimulus), False)
    npt.assert_equal(scene.shape, (128, 128))
    npt.assert_equal(scene.fov, (10.0, 10.0))
    npt.assert_almost_equal(_raster(scene).mean(), 0.5, decimal=2)
    meta = scene.source.metadata
    npt.assert_equal(meta['generator'], 'grating')
    npt.assert_almost_equal(meta['spatial_freq'], 1.0)
    npt.assert_almost_equal(meta['temporal_freq'], 0.0)
    npt.assert_equal(meta['fov'], (10.0, 10.0))
    # A scalar fov is the horizontal one; the vertical follows from `shape`:
    npt.assert_equal(psychophysics.grating(fov=10, shape=(256, 512)).fov,
                     (10.0, 5.0))


def test_grating_video():
    time = np.arange(0, 200, 10)
    scene = psychophysics.grating(spatial_freq=0.5 / dva, temporal_freq=5 * Hz,
                                  fov=20 * dva, shape=(64, 64), time=time)
    npt.assert_equal(isinstance(scene.source, VideoStimulus), True)
    npt.assert_equal(scene.source.vid_shape, (64, 64, time.size))
    npt.assert_almost_equal(scene.source.time, time)
    npt.assert_equal(scene.shape, (64, 64))
    # A single explicit time point is a one-frame video, not an image:
    one = psychophysics.grating(spatial_freq=0.2, shape=(16, 16),
                                fov=10, time=[7.0])
    npt.assert_equal(isinstance(one.source, VideoStimulus), True)
    npt.assert_almost_equal(one.source.time, [7.0])


@pytest.mark.parametrize('shape', [(64, 64), (256, 256)])
def test_grating_spatial_period(shape):
    """One cycle spans 1/spatial_freq dva at any raster resolution"""
    spatial_freq, fov = 0.5, 20.0
    scene = psychophysics.grating(spatial_freq=spatial_freq, fov=fov,
                                  shape=shape)
    row = _raster(scene)[shape[0] // 2, :, 0]
    # Interior local maxima, ties broken to the right so a two-pixel plateau
    # straddling a peak counts once:
    cols = np.where((row[1:-1] >= row[:-2]) & (row[1:-1] > row[2:]))[0] + 1
    x, _ = scene.pixel_to_dva(cols, np.zeros_like(cols))
    # 20 dva at 0.5 cycles/dva holds 10 cycles, with a peak at fixation, so
    # the ones at +/- 10 dva fall outside the frame:
    npt.assert_equal(cols.size, 9)
    npt.assert_allclose(np.diff(x), 1.0 / spatial_freq, atol=fov / shape[1])
    npt.assert_allclose(x[cols.size // 2], 0.0, atol=fov / shape[1])


def test_grating_temporal_phase():
    """Temporal phase follows the physical clock, not the frame index"""
    # 5 Hz: half a period is 100 ms (frame 4 here), a full one 200 ms.
    scene = psychophysics.grating(spatial_freq=0.5, temporal_freq=5 * Hz,
                                  fov=20, shape=(64, 64),
                                  time=np.arange(0, 201, 25))
    vid = _raster(scene)
    npt.assert_allclose(vid[..., 8], vid[..., 0], atol=1e-5)
    # Half a period inverts a full-contrast grating around mean gray:
    npt.assert_allclose(vid[..., 4], 1.0 - vid[..., 0], atol=1e-5)
    # Sampling the same physical instants twice as densely does not change
    # what happens at 100 ms, which a frame-index phase would:
    dense = psychophysics.grating(spatial_freq=0.5, temporal_freq=5 * Hz,
                                  fov=20, shape=(64, 64),
                                  time=np.arange(0, 201, 12.5))
    npt.assert_array_equal(_raster(dense)[..., 8], vid[..., 4])


def test_grating_sampling_invariance():
    """Shared timestamps give identical frames on different sampling grids"""
    kwargs = dict(spatial_freq=0.5, temporal_freq=7 * Hz, direction=30 * deg,
                  fov=20, shape=(64, 64))
    coarse = psychophysics.grating(time=np.arange(0, 300, 25), **kwargs)
    fine = psychophysics.grating(time=np.arange(0, 300, 10), **kwargs)
    t_coarse, t_fine = coarse.source.time, fine.source.time
    shared = np.intersect1d(t_coarse, t_fine)
    npt.assert_equal(shared.size > 1, True)
    for t in shared:
        npt.assert_array_equal(
            _raster(coarse)[..., int(np.argmin(abs(t_coarse - t)))],
            _raster(fine)[..., int(np.argmin(abs(t_fine - t)))])
    # The grids do disagree frame by frame, so the check above is not vacuous:
    npt.assert_equal(np.array_equal(_raster(coarse)[..., 1],
                                    _raster(fine)[..., 1]), False)


@pytest.mark.parametrize('direction,along_x', [(0, True), (90, False),
                                               (180, True), (270, False)])
def test_grating_direction(direction, along_x):
    """0 deg gives vertical bars varying along x, 90 deg horizontal ones"""
    scene = psychophysics.grating(spatial_freq=0.5, direction=direction * deg,
                                  fov=20, shape=(64, 64))
    img = _raster(scene)[..., 0]
    if along_x:
        # Varies along x only, so every row is the same:
        npt.assert_allclose(img - img[0], 0, atol=1e-6)
    else:
        npt.assert_allclose(img - img[:, :1], 0, atol=1e-6)


def test_grating_drift_speed():
    """The pattern moves along `direction` at temporal_freq/spatial_freq"""
    spatial_freq, temporal_freq = 0.05, 0.2
    # 4 dva/s for 250 ms, i.e. 1 dva. A period of 20 dva puts a single peak
    # in the 20-degree frame, so the brightest pixel is unambiguous.
    speed = temporal_freq / spatial_freq
    for direction, sign in ((0, 1), (180, -1)):
        scene = psychophysics.grating(spatial_freq=spatial_freq,
                                      temporal_freq=temporal_freq,
                                      direction=direction * deg, fov=20,
                                      shape=(256, 256),
                                      time=np.array([0.0, 250.0]))
        npt.assert_allclose(_peak_x(scene, 0), 0.0, atol=0.2)
        npt.assert_allclose(_peak_x(scene, 1), sign * speed * 0.25, atol=0.2)


def test_grating_phase():
    spatial_freq, fov, shape = 0.05, 20.0, (256, 256)
    kwargs = dict(spatial_freq=spatial_freq, fov=fov, shape=shape)
    # Phase 0 puts a luminance peak at fixation:
    npt.assert_allclose(_peak_x(psychophysics.grating(**kwargs)), 0.0,
                        atol=0.2)
    # Half a cycle inverts a full-contrast grating around mean gray:
    npt.assert_allclose(_raster(psychophysics.grating(phase=180 * deg,
                                                      **kwargs)),
                        1.0 - _raster(psychophysics.grating(**kwargs)),
                        atol=1e-6)
    # A quarter cycle moves the peak a quarter period against +x:
    quarter = psychophysics.grating(phase=90 * deg, **kwargs)
    npt.assert_allclose(_peak_x(quarter), -0.25 / spatial_freq, atol=0.2)
    npt.assert_allclose(_at(quarter, 0, 0), 0.5, atol=0.02)


def test_grating_contrast_and_mask():
    # 0.1-dva pixels on an odd raster put pixel centers exactly on the peaks
    # and troughs of a 1 cycle/dva grating:
    kwargs = dict(spatial_freq=1, fov=10.1, shape=(101, 101))
    for mask in ('circle', None):
        img = _raster(psychophysics.grating(contrast=0.45, mask=mask,
                                            **kwargs))
        # A circular aperture is 1 wherever it does not cut, so contrast
        # still sets the extremes:
        npt.assert_almost_equal(img.max(), 0.725, decimal=3)
        npt.assert_almost_equal(img.min(), 0.275, decimal=3)
    # A Gaussian aperture attenuates the pattern away from fixation, so
    # contrast only bounds the excursion there:
    img = _raster(psychophysics.grating(contrast=0.45, mask='gauss',
                                        **kwargs))
    npt.assert_equal(img.max() <= 0.725 + 1e-6, True)
    npt.assert_equal(img.min() >= 0.275 - 1e-6, True)
    # Zero contrast is a uniform mean-gray field:
    npt.assert_allclose(_raster(psychophysics.grating(contrast=0, **kwargs)),
                        0.5, atol=1e-6)
    # The corners lie outside a circular aperture, and 3 sigma out in a
    # Gaussian one:
    for mask, decimal in (('circle', 6), ('gauss', 2)):
        img = _raster(psychophysics.grating(mask=mask, **kwargs))
        npt.assert_almost_equal(img[0, 0, 0], 0.5, decimal=decimal)
        npt.assert_almost_equal(img[-1, -1, 0], 0.5, decimal=decimal)


def test_grating_units():
    bare = psychophysics.grating(spatial_freq=0.5, temporal_freq=4,
                                 direction=45, phase=90, fov=20,
                                 shape=(64, 64), time=[0, 20, 40])
    unitful = psychophysics.grating(spatial_freq=0.5 / dva,
                                    temporal_freq=4 * Hz, direction=45 * deg,
                                    phase=np.pi / 2 * rad, fov=20 * dva,
                                    shape=(64, 64), time=[0, 20, 40] * ms)
    npt.assert_array_equal(_raster(bare), _raster(unitful))
    npt.assert_array_equal(bare.source.time, unitful.source.time)
    # Milliseconds are the bare-number convention for time, seconds for speed:
    in_s = psychophysics.grating(spatial_freq=0.5, temporal_freq=4, fov=20,
                                 shape=(64, 64), time=[0, 0.02, 0.04] * sec)
    npt.assert_almost_equal(in_s.source.time, [0, 20, 40])
    # These are visual stimuli: pixels are gray levels, time is physical.
    npt.assert_equal(unitful.source.unit, dimensionless)
    npt.assert_equal(unitful.source.time_unit, ms)


@pytest.mark.parametrize('kwargs', [
    {'spatial_freq': 2 * Hz},
    {'temporal_freq': 2 / dva},
    {'direction': 10 * dva},
    {'phase': 10 * ms},
    {'contrast': 1 * uA},
    {'fov': 10 * ms},
    {'time': [0, 20] * uA},
])
def test_grating_dimensions(kwargs):
    with pytest.raises(DimensionMismatchError):
        psychophysics.grating(**{'shape': (16, 16), 'fov': 10,
                                 'spatial_freq': 0.2, **kwargs})


@pytest.mark.parametrize('kwargs,msg', [
    (dict(spatial_freq=0), "'spatial_freq'"),
    (dict(spatial_freq=-1), "'spatial_freq'"),
    (dict(spatial_freq=np.inf), "'spatial_freq'"),
    (dict(temporal_freq=np.nan), "'temporal_freq'"),
    (dict(temporal_freq=-1), "'temporal_freq'"),
    (dict(direction=np.inf), "'direction'"),
    (dict(phase=np.nan), "'phase'"),
    (dict(contrast=-0.1), "'contrast'"),
    (dict(contrast=1.5), "'contrast'"),
    (dict(mask='blur'), 'Unknown mask'),
    (dict(shape=(0, 16)), "'shape'"),
    (dict(fov=0), "'fov'"),
    # A duration is not a set of sample times: these generators assume no
    # frame rate.
    (dict(time=500), 'scalar'),
    (dict(time=[]), '1-D'),
    (dict(time=[[0, 10], [20, 30]]), '1-D'),
    (dict(time=[0, np.nan]), 'finite'),
    # Drift without sample times would silently freeze at t = 0:
    (dict(temporal_freq=4), "'temporal_freq' is 4 Hz"),
    # Well past the 0.8 cycles/dva Nyquist limit of a 0.625 dva pixel:
    (dict(spatial_freq=10), 'Nyquist'),
])
def test_grating_invalid(kwargs, msg):
    with pytest.raises(ValueError) as excinfo:
        psychophysics.grating(**{'shape': (16, 16), 'fov': 10,
                                 'spatial_freq': 0.2, **kwargs})
    npt.assert_equal(msg in str(excinfo.value), True)


def test_bar_static():
    scene = psychophysics.bar(width=1 * dva, offset=-3 * dva, fov=10 * dva,
                              shape=(128, 128))
    npt.assert_equal(isinstance(scene, Scene), True)
    npt.assert_equal(isinstance(scene.source, ImageStimulus), True)
    npt.assert_equal(isinstance(scene.source, VideoStimulus), False)
    npt.assert_equal(scene.shape, (128, 128))
    npt.assert_equal(scene.fov, (10.0, 10.0))
    # A single bright bar on a dark background, both around mean gray:
    npt.assert_almost_equal(_raster(scene).max(), 1.0)
    npt.assert_almost_equal(_raster(scene).min(), 0.0)
    x, _ = _lit(scene)
    npt.assert_allclose(x.mean(), -3.0, atol=0.1)
    meta = scene.source.metadata
    npt.assert_equal(meta['generator'], 'bar')
    npt.assert_almost_equal(meta['width'], 1.0)
    npt.assert_almost_equal(meta['offset'], -3.0)
    npt.assert_almost_equal(meta['speed'], 0.0)


@pytest.mark.parametrize('call,param,moving', [
    (psychophysics.grating, 'temporal_freq', 4 * Hz),
    (psychophysics.bar, 'speed', 8 * dva / sec),
])
def test_motion_requires_time(call, param, moving):
    """A nonzero drift rate or speed may not silently vanish into an image"""
    kwargs = dict(shape=(32, 32), fov=10)
    with pytest.raises(ValueError) as excinfo:
        call(**kwargs, **{param: moving})
    npt.assert_equal("'time' is None" in str(excinfo.value), True)
    # Zero motion is what a static stimulus means, so it stays legal:
    static = call(**kwargs, **{param: 0})
    npt.assert_equal(isinstance(static.source, ImageStimulus), True)
    # ... and the same motion is fine once there are sample times:
    video = call(**kwargs, time=np.arange(0, 200, 20), **{param: moving})
    npt.assert_equal(isinstance(video.source, VideoStimulus), True)


def test_bar_video():
    time = np.arange(0, 600, 10)
    scene = psychophysics.bar(width=1, speed=10 * dva / sec, fov=20,
                              shape=(64, 64), time=time)
    npt.assert_equal(isinstance(scene.source, VideoStimulus), True)
    npt.assert_equal(scene.source.vid_shape, (64, 64, time.size))
    npt.assert_almost_equal(scene.source.time, time)
    one = psychophysics.bar(width=2, shape=(16, 16), fov=10,
                            time=[7.0])
    npt.assert_equal(isinstance(one.source, VideoStimulus), True)
    npt.assert_almost_equal(one.source.time, [7.0])


@pytest.mark.parametrize('shape', [(64, 64), (256, 256)])
def test_bar_width_in_dva(shape):
    """The bar is `width` dva across at any raster resolution"""
    width, fov = 2.0, 20.0
    scene = psychophysics.bar(width=width, fov=fov, shape=shape)
    px = fov / shape[1]
    row = _raster(scene)[shape[0] // 2, :, 0]
    npt.assert_allclose(np.count_nonzero(row > 0.5) * px, width, atol=px)
    # ... and spans the whole frame the other way, since it is a stripe:
    col = _raster(scene)[:, shape[1] // 2, 0]
    npt.assert_equal(np.all(col > 0.5), True)


@pytest.mark.parametrize('direction,center', [
    (0, (3, 0)), (90, (0, 3)), (180, (-3, 0)), (270, (0, -3)),
])
def test_bar_direction(direction, center):
    """`direction` is measured counterclockwise from +x, with +y upwards"""
    scene = psychophysics.bar(width=2, direction=direction * deg, offset=3,
                              fov=20, shape=(128, 128))
    x, y = _lit(scene)
    npt.assert_allclose([x.mean(), y.mean()], center, atol=0.2)


def test_bar_displacement():
    """The bar's center is offset + speed * t, from its physical timestamps"""
    speed, offset = 8.0, -4.0
    time = np.array([0.0, 125.0, 250.0])
    scene = psychophysics.bar(width=2, speed=speed * dva / sec, offset=offset,
                              fov=20, shape=(256, 256), time=time)
    for frame, t in enumerate(time):
        x, _ = _lit(scene, frame)
        npt.assert_allclose(x.mean(), offset + speed * t / 1000, atol=0.1)
    # Reversing `direction` mirrors the whole trajectory through fixation,
    # since `offset` is measured along the motion axis; `speed` stays a
    # magnitude either way:
    back = psychophysics.bar(width=2, direction=180 * deg,
                             speed=speed * dva / sec, offset=offset, fov=20,
                             shape=(256, 256), time=time)
    x, _ = _lit(back, 2)
    npt.assert_allclose(x.mean(), -(offset + speed * 0.25), atol=0.1)


def test_bar_sampling_invariance():
    """Shared timestamps give identical frames on different sampling grids"""
    kwargs = dict(width=2, speed=8 * dva / sec, offset=-4, direction=30 * deg,
                  fov=20, shape=(64, 64))
    coarse = psychophysics.bar(time=np.arange(0, 300, 25), **kwargs)
    fine = psychophysics.bar(time=np.arange(0, 300, 10), **kwargs)
    t_coarse, t_fine = coarse.source.time, fine.source.time
    shared = np.intersect1d(t_coarse, t_fine)
    npt.assert_equal(shared.size > 1, True)
    for t in shared:
        npt.assert_array_equal(
            _raster(coarse)[..., int(np.argmin(abs(t_coarse - t)))],
            _raster(fine)[..., int(np.argmin(abs(t_fine - t)))])
    npt.assert_equal(np.array_equal(_raster(coarse)[..., 1],
                                    _raster(fine)[..., 1]), False)
    # A stationary bar is the moving one frozen at t = 0:
    static = psychophysics.bar(**{**kwargs, 'speed': 0})
    npt.assert_array_equal(_raster(static)[..., 0], _raster(fine)[..., 0])


def test_bar_edge_width():
    width, edge_width, fov, shape = 2.0, 1.0, 20.0, (512, 512)
    scene = psychophysics.bar(width=width, edge_width=edge_width, fov=fov,
                              shape=shape)
    px = fov / shape[1]
    row = _raster(scene)[shape[0] // 2, :, 0]
    # The plateau is `width` across, and the ramps add `edge_width` a side:
    npt.assert_allclose(np.count_nonzero(row > 1 - 1e-6) * px, width, atol=px)
    npt.assert_allclose(np.count_nonzero(row > 1e-6) * px,
                        width + 2 * edge_width, atol=2 * px)
    # Halfway down a raised-cosine ramp the profile sits at mean gray:
    npt.assert_allclose(_at(scene, width / 2 + edge_width / 2, 0), 0.5,
                        atol=0.05)
    # ... and the ramp descends monotonically to the background:
    ramp = row[shape[1] // 2:][:int(round((width / 2 + edge_width) / px)) + 2]
    npt.assert_equal(np.all(np.diff(ramp) <= 1e-6), True)


def test_bar_contrast_and_mask():
    kwargs = dict(width=2, fov=10.1, shape=(101, 101))
    for mask in ('circle', None):
        img = _raster(psychophysics.bar(contrast=0.45, mask=mask, **kwargs))
        npt.assert_almost_equal(img.max(), 0.725, decimal=6)
        npt.assert_almost_equal(img.min(), 0.275, decimal=6)
    # A Gaussian aperture attenuates bar and background together, so contrast
    # only bounds the excursion:
    img = _raster(psychophysics.bar(contrast=0.45, mask='gauss', **kwargs))
    npt.assert_equal(img.max() <= 0.725 + 1e-6, True)
    npt.assert_equal(img.min() >= 0.275 - 1e-6, True)
    npt.assert_allclose(_raster(psychophysics.bar(contrast=0, **kwargs)), 0.5,
                        atol=1e-6)
    # Corners are outside a circular aperture, hence mean gray:
    img = _raster(psychophysics.bar(mask='circle', **kwargs))
    npt.assert_almost_equal(img[0, 0, 0], 0.5)
    npt.assert_almost_equal(img[-1, -1, 0], 0.5)


def test_bar_units():
    bare = psychophysics.bar(width=2, edge_width=0.5, direction=45, speed=8,
                             offset=-3, fov=20, shape=(64, 64),
                             time=[0, 20, 40])
    unitful = psychophysics.bar(width=2 * dva, edge_width=0.5 * dva,
                                direction=np.pi / 4 * rad,
                                speed=8 * dva / sec, offset=-3 * dva,
                                fov=20 * dva, shape=(64, 64),
                                time=[0, 20, 40] * ms)
    npt.assert_array_equal(_raster(bare), _raster(unitful))
    npt.assert_array_equal(bare.source.time, unitful.source.time)
    npt.assert_equal(unitful.source.unit, dimensionless)
    npt.assert_equal(unitful.source.time_unit, ms)


@pytest.mark.parametrize('kwargs', [
    {'width': 2 * ms},
    {'edge_width': 2 * deg},
    {'speed': 8 * dva},
    {'offset': 3 * ms},
    {'direction': 10 * dva},
    {'contrast': 1 * uA},
    {'time': [0, 20] * uA},
])
def test_bar_dimensions(kwargs):
    with pytest.raises(DimensionMismatchError):
        psychophysics.bar(**{'shape': (16, 16), 'fov': 10, 'width': 2,
                             **kwargs})


@pytest.mark.parametrize('kwargs,msg', [
    (dict(width=0), "'width'"),
    (dict(width=-1), "'width'"),
    (dict(width=np.inf), "'width'"),
    (dict(edge_width=-1), "'edge_width'"),
    (dict(edge_width=np.nan), "'edge_width'"),
    (dict(speed=np.inf), "'speed'"),
    (dict(speed=-1), "'speed'"),
    (dict(offset=np.nan), "'offset'"),
    (dict(direction=np.nan), "'direction'"),
    (dict(contrast=-0.1), "'contrast'"),
    (dict(contrast=1.5), "'contrast'"),
    (dict(mask='blur'), 'Unknown mask'),
    (dict(shape=(0, 16)), "'shape'"),
    (dict(fov=0), "'fov'"),
    (dict(time=500), 'scalar'),
    (dict(time=[]), '1-D'),
    (dict(time=[0, np.nan]), 'finite'),
    # Motion without sample times would silently freeze at t = 0:
    (dict(speed=8), "'speed' is 8 dva/s"),
    # 0.1 dva across a 10-degree, 16-pixel frame is 0.16 pixels:
    (dict(width=0.1), 'resolve the bar'),
])
def test_bar_invalid(kwargs, msg):
    with pytest.raises(ValueError) as excinfo:
        psychophysics.bar(**{'shape': (16, 16), 'fov': 10, 'width': 2,
                             **kwargs})
    npt.assert_equal(msg in str(excinfo.value), True)


@pytest.mark.parametrize('cls,alt', [
    (GratingStimulus, 'psychophysics.grating'),
    (BarStimulus, 'psychophysics.bar'),
])
def test_psychophysics_legacy_deprecated(cls, alt):
    with pytest.warns(DeprecationWarning) as record:
        cls((16, 16), time=[0, 20])
    # `BarStimulus` builds a `GratingStimulus` internally, which must not warn
    # a second time:
    npt.assert_equal(len(record), 1)
    msg = str(record[0].message)
    for expected in (cls.__name__, alt, '0.11.0', '0.12.0'):
        npt.assert_equal(expected in msg, True)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_psychophysics_legacy_semantics():
    """The deprecated classes keep their raster/frame units through 0.11"""
    grating = GratingStimulus((4, 4), spatial_freq=0.25, temporal_freq=0.1)
    # `time=None` is still a 1-second video on an implicit 50 Hz grid:
    npt.assert_almost_equal(grating.time, np.arange(0, 1001, 20))
    npt.assert_equal(grating.vid_shape, (4, 4, 51))
    # Temporal frequency is still cycles/frame, so 0.1 repeats every 10
    # frames no matter what the time axis says:
    data = grating.data.reshape(grating.vid_shape)
    npt.assert_allclose(data[..., 10], data[..., 0], atol=1e-6)
    # Spatial frequency is still cycles/pixel, so 0.25 repeats every 4 pixels:
    npt.assert_allclose(data[:, 0, 0], data[:, 0, 0][0], atol=1e-6)
    bar = BarStimulus((16, 16), speed=1)
    npt.assert_almost_equal(bar.time, np.arange(0, 1001, 20))
    npt.assert_equal(bar.vid_shape, (16, 16, 51))
    # ... and they are videos, not scenes:
    npt.assert_equal(isinstance(bar, VideoStimulus), True)


def test_grating_spatial_nyquist():
    """A grating at or past two samples per cycle is refused, not aliased"""
    # 20 dva across 64 pixels is a 0.3125 dva pixel, whose Nyquist frequency
    # is 1.6 cycles/dva:
    nyquist = 0.5 / (20.0 / 64)
    kwargs = dict(fov=20, shape=(64, 64), direction=0 * deg)
    psychophysics.grating(spatial_freq=0.99 * nyquist, **kwargs)
    # Exactly at Nyquist the phase is unrecoverable, so equality is refused
    # along with everything above it:
    for spatial_freq in (nyquist, 1.01 * nyquist, 10 * nyquist):
        with pytest.raises(ValueError) as excinfo:
            psychophysics.grating(spatial_freq=spatial_freq, **kwargs)
        npt.assert_equal('Nyquist' in str(excinfo.value), True)


def test_grating_nyquist_is_directional():
    """Each axis is judged by its own angular pixel pitch"""
    # 40 dva over 64 columns is a 0.625 dva pixel (0.8 cycles/dva Nyquist);
    # 10 dva over 64 rows is a 0.15625 dva one (3.2 cycles/dva).
    kwargs = dict(fov=(40, 10), shape=(64, 64), spatial_freq=1.0)
    # A grating that varies only vertically is resolved at 1 cycle/dva ...
    psychophysics.grating(direction=90 * deg, **kwargs)
    # ... while the same frequency drawn horizontally aliases:
    with pytest.raises(ValueError) as excinfo:
        psychophysics.grating(direction=0 * deg, **kwargs)
    npt.assert_equal('horizontal component' in str(excinfo.value), True)
    # An oblique grating sits between the two limits: at 45 deg, 1 cycle/dva
    # has a 0.71 cycles/dva horizontal component, which 0.8 still resolves ...
    psychophysics.grating(direction=45 * deg, **kwargs)
    # ... but 1.2 cycles/dva does not:
    with pytest.raises(ValueError):
        psychophysics.grating(direction=45 * deg,
                              **{**kwargs, 'spatial_freq': 1.2})


def test_grating_temporal_nyquist():
    """Adjacent frames must advance the drift by less than half a cycle"""
    kwargs = dict(spatial_freq=0.5, fov=20, shape=(32, 32))
    # Samples 20 ms apart resolve anything slower than 25 Hz:
    time = np.arange(0, 200, 20)
    psychophysics.grating(temporal_freq=24 * Hz, time=time, **kwargs)
    for temporal_freq in (25, 100):
        with pytest.raises(ValueError) as excinfo:
            psychophysics.grating(temporal_freq=temporal_freq * Hz, time=time,
                                  **kwargs)
        npt.assert_equal('Nyquist' in str(excinfo.value), True)
    # The widest gap decides, not the average one:
    with pytest.raises(ValueError):
        psychophysics.grating(temporal_freq=24 * Hz, time=[0, 1, 2, 100],
                              **kwargs)
    # A single frame has no gap across which the drift could alias:
    psychophysics.grating(temporal_freq=1000 * Hz, time=[0.0], **kwargs)


def test_aperture_is_circular_in_dva():
    """Apertures are radial in visual angle, not stretched to the frame"""
    # A 40 x 10 dva field: the aperture is the largest circle that fits, so 5
    # dva in radius, rather than an ellipse filling the frame.
    scene = psychophysics.grating(spatial_freq=0.2, fov=(40, 10),
                                  shape=(64, 256), mask='circle')
    img = _raster(scene)[..., 0]
    rows, cols = np.where(abs(img - 0.5) > 1e-6)
    x, y = scene.pixel_to_dva(cols, rows)
    npt.assert_array_less(np.hypot(x, y), 5.0 + 1e-9)
    npt.assert_allclose(np.hypot(x, y).max(), 5.0, atol=0.2)
    # 15 dva out along x is inside an ellipse fitted to the frame, but far
    # outside the circle:
    npt.assert_almost_equal(_at(scene, 15, 0), 0.5)
    # A Gaussian aperture is isotropic too. A bar wide enough to fill the
    # frame leaves the aperture as the only thing shaping the gray levels:
    gauss = psychophysics.bar(width=100, fov=(40, 10), shape=(64, 256),
                              mask='gauss')
    npt.assert_allclose(_at(gauss, 3, 0), _at(gauss, 0, 3), atol=1e-5)
    npt.assert_array_less(_at(gauss, 4, 0), _at(gauss, 2, 0))
