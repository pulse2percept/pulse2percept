"""The new prediction API predicts exactly what the old one did

Every reference below was produced by the pre-refactor pipeline --
``implant.stim = source`` followed by ``model.predict_percept(implant)`` -- at
commit 62d5b4e, and is compared against the same setup written the new way:
``model = SomeModel(implant=implant)`` followed by
``model.predict_percept(source)``. The API changed; the numbers did not.

Percepts are compared by shape, time axis and a handful of whole-array
reductions rather than element by element, which keeps the references
readable. Sum, peak and sum of squares say how much brightness there is;
they are blind to where it sits, so a permuted electrode map or a shifted
retinotopy could preserve all three. The brightness moments say where it
sits: mean and mean-square pixel index along the percept's Y and X axes,
which no translation or reshuffling of the grid leaves alone.

``percept.data`` is float32, so the reductions carry slack that varies with
the platform's ``exp`` and the compiler's floating-point contraction.
``RTOL`` is loose enough to absorb that and still orders of magnitude tighter
than any change in the pipeline would be.
"""
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import ArgusII, GridImplant
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models import (AxonMapModel, BiphasicAxonMapModel,
                                  FadingTemporal, Model, ScoreboardModel,
                                  ScoreboardSpatial)
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulseTrain,
                                   ImageStimulus, VideoStimulus)
from pulse2percept.units import dva, xTh
from pulse2percept.vision import Scene, Scotoma

GRID = dict(xrange=(-8, 8), yrange=(-6, 6), step=1)
AXON = dict(n_axons=200, n_ax_segments=100, ignore_pickle=True)

#: What the pre-refactor pipeline predicted: shape, (t_first, t_last), sum,
#: max, sum of squares, and the brightness moments (mean_y, mean_sq_y,
#: mean_x, mean_sq_x) that `moments` below computes.
REFERENCE = {
    'scoreboard': ((13, 17, 1), None,
                   106.27331389391497, 29.729562759399414, 1590.841862258684,
                   (4.855147072796709, 24.43435468223049,
                    6.805482721437667, 47.70945073251632)),
    'axonmap': ((13, 17, 1), None,
                111.66620632618813, 23.478652954101562, 1327.6353996210921,
                (5.185161927644366, 27.987468482922242,
                 6.827698436941981, 47.849236089244805)),
    'spatiotemporal': ((13, 17, 4), (0.0, 60.0),
                       2.282379476566313, 0.32124122977256775,
                       0.28389843167895795,
                       (4.931046339872862, 24.86797727082938,
                        6.931046450353491, 48.59216420177322)),
    'encoded_image': ((13, 17, 1), None,
                      4068.7693935632706, 46.307098388671875,
                      105311.18907585883,
                      (8.147908231279617, 75.03071679781367,
                       8.276378690968135, 92.40379187416536)),
    'encoded_video': ((13, 17, 3), (0.0, 100.0),
                      13472.329383134842, 51.660400390625,
                      337278.64476578706,
                      (6.0661377111050685, 49.745639543040014,
                       7.921347896154394, 86.90694497105545)),
    'encoded_video_temporal': ((13, 17, 3), (50.0, 150.0),
                               39.186431967886165, 0.21076203882694244,
                               3.328352739130871,
                               (6.528613866884478, 56.270883389329825,
                                7.894180997742635, 86.76219135133657)),
    'biphasic': ((13, 17, 1), None,
                 3.807037961360792, 0.5194367010755934, 1.0132546632537747,
                 (5.0853018679140565, 27.047305434768763,
                  6.941999430159145, 49.25247073610444)),
    'dynaphos': ((7, 7, 6), (0.0, 100.0),
                 3.864256768792984e-06, 1.0946714610327035e-06,
                 3.800738015684018e-12,
                 (2.0, 4.0, 1.0, 1.0)),
    'scene_gaze': ((13, 17, 1), None,
                   1663.1452019751928, 36.72337341308594, 43973.7099062507,
                   (6.000000013081055, 40.855445551864825,
                    8.174004836496247, 71.63952372279992)),
    'scene_scotoma': ((41, 41, 3, 1), None,
                      2435.118678161456, 1.0, 1652.0926838311032,
                      (19.9999999996621, 544.5046493203229,
                       27.224079296239694, 833.4678215082496)),
}


#: Sized from the spread actually observed across the platforms CI runs on,
#: not from float32 epsilon: the AxonMap sum differs by 3e-6 between macOS and
#: Linux for percepts that are otherwise bit-identical within a platform.
RTOL = 1e-5


def moments(data):
    """Mean and mean-square pixel index of brightness along Y and X

    Raw moments rather than centered ones: a percept driven by a single
    electrode has essentially no spread, and a variance computed from it is
    cancellation noise that no tolerance can pin.
    """
    total = data.sum()
    out = []
    for axis in (0, 1):
        shape = [-1 if k == axis else 1 for k in range(data.ndim)]
        idx = np.arange(data.shape[axis], dtype=np.float64).reshape(shape)
        out += [(data * idx).sum() / total, (data * idx ** 2).sum() / total]
    return out


def assert_matches_reference(name, percept):
    shape, time, total, peak, sumsq, spatial = REFERENCE[name]
    data = np.asarray(percept.data, dtype=np.float64)
    npt.assert_equal(percept.data.shape, shape)
    if time is None:
        npt.assert_equal(percept.time, None)
    else:
        npt.assert_allclose([percept.time[0], percept.time[-1]], time,
                            rtol=1e-12)
    npt.assert_allclose(data.sum(), total, rtol=RTOL)
    npt.assert_allclose(data.max(), peak, rtol=RTOL)
    npt.assert_allclose((data ** 2).sum(), sumsq, rtol=RTOL)
    npt.assert_allclose(moments(data), spatial, rtol=RTOL)


def picture():
    return ImageStimulus(np.linspace(0, 1, 64).reshape((8, 8)))


def movie():
    rng = np.random.default_rng(0)
    return VideoStimulus(rng.random((6, 10, 3)), metadata={'fps': 20})


def scene_of(**kwargs):
    px = 41
    ramp = ImageStimulus(np.tile(np.linspace(0, 1, px), (px, 1)))
    return Scene(ramp, fov=(px, px), **kwargs)


def encoding_grid():
    return GridImplant(shape=(4, 4), spacing=500,
                       encoder=AmplitudeEncoder(amp_range=(0, 50)))


def test_scoreboard_prediction_is_unchanged():
    model = ScoreboardModel(implant=ArgusII(), rho=200, **GRID).build()
    assert_matches_reference('scoreboard',
                             model.predict_percept({'C5': 30, 'A1': 10}))


def test_axonmap_prediction_is_unchanged():
    model = AxonMapModel(implant=ArgusII(), rho=200, lam=500, **GRID,
                         **AXON).build()
    assert_matches_reference('axonmap',
                             model.predict_percept({'C5': 30, 'A1': 10}))


def test_spatiotemporal_prediction_is_unchanged():
    model = Model(implant=ArgusII(),
                  spatial=ScoreboardSpatial(rho=200, **GRID),
                  temporal=FadingTemporal(tau=100)).build()
    percept = model.predict_percept(
        {'C5': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)},
        t_percept=[0, 20, 40, 60])
    assert_matches_reference('spatiotemporal', percept)


def test_encoded_image_prediction_is_unchanged():
    model = ScoreboardModel(implant=ArgusII(), rho=200, **GRID).build()
    assert_matches_reference('encoded_image',
                             model.predict_percept(picture()))


def test_encoded_video_prediction_is_unchanged():
    # Argus II's own 6 Hz encoder and six-group raster, so this covers the
    # schedule as well as the encoding.
    model = ScoreboardModel(implant=ArgusII(), rho=200, **GRID).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        percept = model.predict_percept(movie())
    assert_matches_reference('encoded_video', percept)


def test_encoded_video_with_a_temporal_stage_is_unchanged():
    # The other half of the modulation/pulses distinction: with a temporal
    # stage the spatial one reads the delivered train instead.
    model = Model(implant=ArgusII(),
                  spatial=ScoreboardSpatial(rho=200, **GRID),
                  temporal=FadingTemporal(tau=100)).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        percept = model.predict_percept(movie())
    assert_matches_reference('encoded_video_temporal', percept)


def test_biphasic_axon_map_prediction_is_unchanged():
    implant = ArgusII()
    implant.thresholds = 80
    model = BiphasicAxonMapModel(implant=implant, rho=200, lam=500, **GRID,
                                 **AXON).build()
    percept = model.predict_percept(
        {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)})
    assert_matches_reference('biphasic', percept)


def test_dynaphos_prediction_is_unchanged():
    model = DynaphosModel(implant=Cortivis(), xrange=(-3, 3), yrange=(-3, 3),
                          step=1, dt=20).build()
    percept = model.predict_percept(
        {'11': BiphasicPulseTrain(300, 100, 0.17, stim_dur=100)})
    assert_matches_reference('dynaphos', percept)


def test_scene_with_gaze_prediction_is_unchanged():
    model = ScoreboardModel(implant=encoding_grid(), rho=200, **GRID).build()
    percept = model.predict_percept(scene_of(), gaze=(4, 0) * dva)
    assert_matches_reference('scene_gaze', percept)


def test_scene_with_scotoma_prediction_is_unchanged():
    model = ScoreboardModel(implant=encoding_grid(), rho=200, **GRID).build()
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    assert_matches_reference('scene_scotoma',
                             model.predict_percept(scene, vmax=50))


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, AxonMapModel])
def test_one_bound_model_predicts_many_stimuli(ModelClass):
    """A bound model is asked repeatedly, and keeps nothing between calls"""
    extra = AXON if ModelClass is AxonMapModel else {}
    model = ModelClass(implant=ArgusII(), rho=200, **GRID, **extra).build()
    sources = [{'C5': 30, 'A1': 10}, {'A1': 20}, picture(),
               {'C5': 30, 'A1': 10}]
    percepts = [model.predict_percept(s) for s in sources]

    # The first and last are the same stimulus, so they are the same percept:
    npt.assert_array_equal(percepts[0].data, percepts[-1].data)
    # ... and the ones in between really were different predictions:
    npt.assert_equal(np.allclose(percepts[0].data, percepts[1].data), False)
    npt.assert_equal(np.allclose(percepts[0].data, percepts[2].data), False)
    # Nothing was built a second time, and nothing was stored on the implant:
    npt.assert_equal(model.is_built, True)
    npt.assert_equal(hasattr(model.implant, 'stim'), False)


def test_rebinding_the_implant_invalidates_the_build():
    model = ScoreboardModel(implant=ArgusII(), rho=200, **GRID).build()
    npt.assert_equal(model.is_built, True)
    here = model.predict_percept({'C5': 30})

    model.implant = ArgusII(x=2000)
    npt.assert_equal(model.is_built, False)
    model.build()
    there = model.predict_percept({'C5': 30})
    # The same stimulus on a shifted array lands somewhere else, which is what
    # says the rebind reached the prediction rather than being ignored:
    npt.assert_equal(np.allclose(here.data, there.data), False)
    npt.assert_equal(np.argmax(here.data) != np.argmax(there.data), True)
