import numpy as np
import numpy.testing as npt
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pulse2percept.implants import DiskElectrode, ElectrodeArray, Implant
from pulse2percept.models import Model
from pulse2percept.models.retina import ScoreboardSpatial
from pulse2percept.models.temporal import FadingTemporal
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (BiphasicPulseTrain, ImageStimulus,
                                   MonophasicPulse, Stimulus, TraceEncoder,
                                   VideoStimulus)
from pulse2percept.topography import VisualFieldMap
from pulse2percept.units import Hz, dva, ms, s, um
from pulse2percept.plotting import (play_implant_percept,
                                    play_stimulus_percept,
                                    plot_implant_percept,
                                    plot_stimulus_percept)
from pulse2percept.plotting.comparison import (STIM_CMAP, _electrode_drive,
                                               _electrode_stim)


def video(n_frames=5, time=None, shape=(4, 6)):
    """A video whose frame ``i`` is uniformly ``i``"""
    frames = np.ones((*shape, n_frames)) * np.arange(n_frames)
    if time is None:
        time = np.arange(n_frames) * 33.0
    return VideoStimulus(frames, time=time)


def percept(n_frames=4, time=None, **kwargs):
    frames = np.ones((3, 3, n_frames)) * np.arange(n_frames)
    if time is None:
        time = np.arange(n_frames) * 50.0
    return Percept(frames, time=time, **kwargs)


def source_index(ani):
    """Which source frame each display frame shows"""
    return ani._layers[0].index


def test_plot_stimulus_percept():
    stim = ImageStimulus(np.random.rand(8, 10))
    axes = plot_stimulus_percept(stim, percept(n_frames=1))
    npt.assert_equal([ax.get_title() for ax in axes],
                     ['Stimulus', 'Percept'])
    npt.assert_equal(len(axes[0].images), 1)
    npt.assert_almost_equal(axes[0].images[0].get_array(),
                            stim.data.reshape(stim.img_shape))
    # The percept is drawn by ``Percept.plot``, which uses a pcolormesh:
    npt.assert_equal(len(axes[1].collections), 1)
    # Titles and plotting arguments are passed through:
    axes = plot_stimulus_percept(stim, percept(n_frames=1),
                                 titles=('In', 'Out'),
                                 stim_kwargs={'cmap': 'viridis'},
                                 percept_kwargs={'kind': 'hex'})
    npt.assert_equal([ax.get_title() for ax in axes], ['In', 'Out'])
    npt.assert_equal(axes[0].images[0].get_cmap().name, 'viridis')


def test_plot_stimulus_percept_axes():
    stim = ImageStimulus(np.random.rand(8, 10))
    _, axes = plt.subplots(nrows=2)
    plot_stimulus_percept(stim, percept(n_frames=1), axes=axes)
    npt.assert_equal(len(axes[0].images), 1)
    npt.assert_equal(len(axes[1].collections), 1)
    # Exactly two Axes, and they must be Axes:
    with pytest.raises(ValueError):
        plot_stimulus_percept(stim, percept(n_frames=1),
                              axes=plt.subplots(ncols=3)[1])
    with pytest.raises(TypeError):
        plot_stimulus_percept(stim, percept(n_frames=1), axes=['a', 'b'])


def test_plot_stimulus_percept_errors():
    """Only a visual source, and only one that has a single frame"""
    # A video and its percept have no frame that stands for both of them:
    with pytest.raises(TypeError):
        plot_stimulus_percept(video(), percept())
    # The electrical stimulus an encoder made is not the source picture:
    with pytest.raises(TypeError):
        plot_stimulus_percept(Stimulus({'A1': 1}), percept(n_frames=1))


def test_play_stimulus_percept_still_image():
    """A still image stays put while the percept fades"""
    stim = ImageStimulus(np.random.rand(8, 10))
    ani = play_stimulus_percept(stim, percept())
    npt.assert_equal(len(list(ani.frame_seq)), 4)
    npt.assert_equal(source_index(ani), [0, 0, 0, 0])
    npt.assert_equal(ani._layers[0].data.shape[-1], 1)
    html = ani.to_jshtml()
    npt.assert_equal('p2p-anim' in html, True)
    npt.assert_equal('"n": 4' in html, True)


def test_play_stimulus_percept_matching_rates():
    vid = video(n_frames=4, time=np.arange(4) * 50.0)
    ani = play_stimulus_percept(vid, percept(n_frames=4))
    npt.assert_equal(source_index(ani), [0, 1, 2, 3])
    npt.assert_equal(ani._layers[1].index, [0, 1, 2, 3])


def test_play_stimulus_percept_zero_order_hold():
    """A source on another time grid holds the frame that is up"""
    # Source at 30 Hz, percept every 50 ms:
    ani = play_stimulus_percept(video(n_frames=5), percept(n_frames=4))
    npt.assert_equal(source_index(ani), [0, 1, 3, 4])
    # A percept that outlasts its source holds the last frame:
    ani = play_stimulus_percept(video(n_frames=2), percept(n_frames=4))
    npt.assert_equal(source_index(ani), [0, 1, 1, 1])
    # A percept whose clock starts before the source holds the first one:
    ani = play_stimulus_percept(video(n_frames=3),
                                percept(n_frames=3, time=[-20.0, 0.0, 40.0]))
    npt.assert_equal(source_index(ani), [0, 0, 1])


def test_play_stimulus_percept_time_units():
    """Source and percept are lined up in physical time, not in raw numbers"""
    ani = play_stimulus_percept(video(n_frames=5),
                                percept(n_frames=3, time=[0, 0.05, 0.1],
                                        time_unit=s))
    npt.assert_equal(source_index(ani), [0, 1, 3])


def test_play_stimulus_percept_fps():
    """'fps' resamples the whole presentation, both panels with it"""
    ani = play_stimulus_percept(video(n_frames=5), percept(n_frames=4),
                                fps=40 * Hz)
    # 200 ms of percept at 40 Hz is eight display frames:
    npt.assert_equal(len(list(ani.frame_seq)), 8)
    npt.assert_equal(ani._layers[1].index, [0, 0, 1, 1, 2, 2, 3, 3])
    npt.assert_equal(source_index(ani), [0, 0, 1, 2, 3, 3, 4, 4])


def test_play_stimulus_percept_rgb():
    vid = VideoStimulus(np.random.rand(4, 6, 3, 3), time=[0, 50.0, 100.0])
    ani = play_stimulus_percept(vid, percept(n_frames=3))
    npt.assert_equal(ani._layers[0].data.shape, (4, 6, 3, 3))
    npt.assert_equal('p2p-anim' in ani.to_jshtml(), True)
    # An RGB percept carries its own colors, so it has no brightness range:
    rgb = Percept(np.random.rand(3, 3, 3, 2), time=[0, 50.0])
    npt.assert_equal('p2p-anim' in
                     play_stimulus_percept(vid, rgb).to_jshtml(), True)
    with pytest.raises(ValueError):
        play_stimulus_percept(vid, rgb, vmax=1)


def test_play_stimulus_percept_annotate_time():
    ani = play_stimulus_percept(video(), percept())
    npt.assert_equal(ani._labels[-1], 't = 150.00 ms')
    npt.assert_equal('t = 150.00 ms' in ani.to_jshtml(), True)
    ani = play_stimulus_percept(video(), percept(), annotate_time=False)
    npt.assert_equal(ani._labels, None)
    npt.assert_equal('t = 150.00 ms' in ani.to_jshtml(), False)


def test_play_stimulus_percept_leaves_data_alone():
    vid, perc = video(), percept()
    stim_data, stim_time = vid.data.copy(), vid.time.copy()
    perc_data, perc_time = perc.data.copy(), perc.time.copy()
    play_stimulus_percept(vid, perc, fps=60).to_jshtml()
    npt.assert_almost_equal(vid.data, stim_data)
    npt.assert_almost_equal(vid.time, stim_time)
    npt.assert_almost_equal(perc.data, perc_data)
    npt.assert_almost_equal(perc.time, perc_time)


def test_play_stimulus_percept_untimed_video(monkeypatch):
    """A source with no clock is refused, not frozen at its first frame"""
    vid = video()
    # ``Stimulus`` refuses to build a multi-frame stimulus without a time axis,
    # so make one report that it has none:
    monkeypatch.setattr(VideoStimulus, 'time', property(lambda self: None))
    with pytest.raises(ValueError):
        play_stimulus_percept(vid, percept())


def test_play_stimulus_percept_errors():
    # A percept without a time axis is a still image:
    with pytest.raises(ValueError):
        play_stimulus_percept(ImageStimulus(np.random.rand(4, 4)),
                              Percept(np.random.rand(3, 3, 1)))
    # The electrical stimulus an encoder made is not the source picture:
    with pytest.raises(TypeError):
        play_stimulus_percept(Stimulus({'A1': 1}), percept())


class LinearMap(VisualFieldMap):
    """1 dva = 1000 um"""

    def dva_to_lin(self, x, y):
        return (1000 * np.asarray(x, dtype=np.float32),
                1000 * np.asarray(y, dtype=np.float32))

    def lin_to_dva(self, x, y):
        return (np.asarray(x, dtype=np.float32) / 1000,
                np.asarray(y, dtype=np.float32) / 1000)

    def from_dva(self):
        return {'ret': self.dva_to_lin}

    def to_dva(self):
        return {'ret': self.lin_to_dva}


def line_model(**params):
    """Electrodes A, B, C at x = 0, 1000, 2000 um"""
    implant = Implant(ElectrodeArray({
        'A': DiskElectrode(0, 0, 0, 100),
        'B': DiskElectrode(1000, 0, 0, 100),
        'C': DiskElectrode(2000, 0, 0, 100)}))
    return ScoreboardSpatial(implant, visual_field_map=LinearMap(),
                             xrange=(-1, 3), yrange=(-1, 1), step=0.25,
                             **params).build()


def trace_stim(model):
    """A, B, then C, 100 ms each"""
    return TraceEncoder(model, step_dur=100 * ms).encode(
        np.array([(0, 0), (1, 0), (2, 0)]) * dva)


def stim_patches(ax):
    """The electrode collection the implant drew on ``ax``"""
    return next(c for c in ax.collections if hasattr(c, '_stim_patches'))


def fill(ax, name):
    coll = stim_patches(ax)
    return coll.get_facecolor()[coll._stim_patches[name]]


def overlay_pixel(ani, model, xy, frame, placed=True):
    """RGBA of the implant layer at ``xy`` (um) in display frame ``frame``"""
    layer = ani._layers[0]
    im = layer.image
    to_display = (stim_patches(im.axes).get_transform() if placed
                  else im.axes.transData)
    x, y = to_display.transform(xy)
    bbox = im.get_window_extent()
    h, w = layer.data.shape[:2]
    col = int((x - bbox.x0) / bbox.width * w)
    row = int((bbox.y1 - y) / bbox.height * h)
    return layer.data[row, col, :, layer.index[frame]]


def electrode_pixel(ani, model, name, frame):
    e = model.implant.electrode_array[name]
    return overlay_pixel(ani, model, (e.x, e.y), frame)


def test_plot_implant_percept_single_electrode():
    model = line_model()
    percept = model.predict_percept({'B': 20})
    axes = plot_implant_percept(model, percept)
    npt.assert_equal([ax.get_title() for ax in axes], ['Implant', 'Percept'])
    cmap = plt.get_cmap(STIM_CMAP)
    npt.assert_almost_equal(fill(axes[0], 'B'), cmap(1.0, alpha=0.8))
    # Undriven electrodes keep the implant's own fill:
    base = model.implant.electrode_array['A'].plot_kwargs['fc']
    npt.assert_almost_equal(fill(axes[0], 'A'), base)
    npt.assert_almost_equal(fill(axes[0], 'C'), base)
    # Only the driven electrode is labeled:
    npt.assert_equal([t.get_text() for t in axes[0].texts], [])
    axes = plot_implant_percept(model, percept, annotate=True)
    npt.assert_equal([t.get_text() for t in axes[0].texts], ['B'])


def test_plot_implant_percept_shared_scale():
    model = line_model()
    axes = plot_implant_percept(model, model.predict_percept({'A': 10,
                                                              'C': 20}))
    cmap = plt.get_cmap(STIM_CMAP)
    npt.assert_almost_equal(fill(axes[0], 'A'), cmap(0.5, alpha=0.8))
    npt.assert_almost_equal(fill(axes[0], 'C'), cmap(1.0, alpha=0.8))


def test_plot_implant_percept_cathodic():
    model = line_model()
    axes = plot_implant_percept(model, model.predict_percept({'A': -10,
                                                              'C': 20}))
    # Cathodic current is drive, too:
    cmap = plt.get_cmap(STIM_CMAP)
    npt.assert_almost_equal(fill(axes[0], 'A'), cmap(0.5, alpha=0.8))


def test_plot_implant_percept_placed_implant():
    model = line_model(implant_rotation=90, implant_position=(500, 0) * um)
    axes = plot_implant_percept(model, model.predict_percept({'C': 20}))
    _, ax = plt.subplots()
    model.plot(ax=ax, show_implant=True)
    # The colored implant is the placed one:
    coll, placed = stim_patches(axes[0]), stim_patches(ax)
    npt.assert_almost_equal(
        (coll.get_transform() - axes[0].transData).get_matrix(),
        (placed.get_transform() - ax.transData).get_matrix())
    # C at local (2000, 0) um sits at (500, 2000) um:
    xy = coll.get_transform().transform((2000, 0))
    npt.assert_almost_equal(axes[0].transData.inverted().transform(xy),
                            (500, 2000))


def test_plot_implant_percept_errors():
    model = line_model()
    # Two frames have no single time point:
    with pytest.raises(ValueError, match='play_implant_percept'):
        plot_implant_percept(model, model.predict_percept(trace_stim(model)))
    # A percept that does not record its stimulus:
    with pytest.raises(ValueError, match='metadata'):
        plot_implant_percept(model, Percept(np.zeros((3, 3, 1))))


def test_electrode_stim_nested():
    implant = line_model().implant
    model = Model(spatial=ScoreboardSpatial(implant, xrange=(-1, 3),
                                            yrange=(-1, 1), step=0.5),
                  temporal=FadingTemporal()).build()
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    percept = model.predict_percept({'A': pt})
    npt.assert_equal(isinstance(percept.metadata['stim'], Percept), True)
    stim = _electrode_stim(percept)
    npt.assert_equal(isinstance(stim, Stimulus), True)
    npt.assert_equal(stim.electrodes, ['A'])
    ani = play_implant_percept(model, percept)
    npt.assert_equal(len(ani._layers), 2)


def test_electrode_drive_zero_order_hold():
    stim = trace_stim(line_model())
    npt.assert_equal(stim.electrodes, ['A', 'B', 'C'])
    # Frames at 0, 100, 200 ms, held in between; off once the stimulus ends:
    drive = _electrode_drive(stim, [0, 50, 99, 100, 250, 300, 350])
    on = drive > 0
    npt.assert_equal(on[0], [1, 1, 1, 0, 0, 0, 0])
    npt.assert_equal(on[1], [0, 0, 0, 1, 0, 0, 0])
    npt.assert_equal(on[2], [0, 0, 0, 0, 1, 0, 0])
    # The modulation, not the pulse phases:
    npt.assert_almost_equal(drive[drive > 0], 100)
    # A timeless percept summarizes the whole stimulus:
    npt.assert_almost_equal(_electrode_drive(stim).ravel(), 100)


def test_electrode_drive_raw_waveform():
    # Pulses start every 50 ms; display frames are 20 ms apart:
    stim = Stimulus({'A': BiphasicPulseTrain(20, 10, 0.45, stim_dur=200)})
    drive = _electrode_drive(stim, np.arange(0, 200, 20))
    npt.assert_almost_equal(drive[0], [10, 0, 10, 0, 0, 10, 0, 10, 0, 0])
    # A cathodic pulse between display frames is still drive:
    stim = Stimulus({'A': MonophasicPulse(-20, 1, delay_dur=30,
                                          stim_dur=100)})
    npt.assert_almost_equal(_electrode_drive(stim, [0, 20, 40, 60])[0],
                            [0, 20, 0, 0])
    # A sample between display times counts toward the earlier one:
    stim = Stimulus([[0, 5, 0, 0]], time=[0, 10, 11, 40])
    npt.assert_almost_equal(_electrode_drive(stim, [0, 20])[0], [5, 0])
    # Values at the display time are interpolated:
    npt.assert_almost_equal(_electrode_drive(stim, [5, 15])[0], [5, 0])


def test_play_implant_percept_trace():
    model = line_model()
    percept = model.predict_percept(trace_stim(model))
    npt.assert_almost_equal(percept.time, [0, 100, 200])
    ani = play_implant_percept(model, percept, annotate=True)
    npt.assert_equal(len(ani._layers), 2)
    npt.assert_equal(ani._fmt, 'png')
    # One rendered state per electrode:
    npt.assert_equal(ani._layers[0].data.shape[-1], 3)
    # A, then B, then C:
    for frame, name in enumerate('ABC'):
        for other in 'ABC':
            alpha = electrode_pixel(ani, model, other, frame)[3]
            npt.assert_equal(alpha > 0, other == name)
    # Labels are baked into the implant layer:
    npt.assert_equal(len(ani._layers[0].image.axes.texts), 0)
    npt.assert_equal('<canvas' in ani.to_jshtml(), True)


def test_play_implant_percept_clocks():
    model = line_model()
    stim = trace_stim(model)
    # A percept sampled on its own clock, off the 100 ms modulation frames:
    percept = Percept(np.zeros((3, 3, 5)), time=[0, 50, 150, 250, 300],
                      metadata={'stim': stim})
    ani = play_implant_percept(model, percept)
    active = [[electrode_pixel(ani, model, name, frame)[3] > 0
               for name in 'ABC'] for frame in range(5)]
    npt.assert_equal(active, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                              [0, 0, 0]])
    # So does the display clock:
    ani = play_implant_percept(model, percept, fps=10 * Hz)
    active = [electrode_pixel(ani, model, 'B', frame)[3] > 0
              for frame in range(3)]
    npt.assert_equal(active, [False, True, False])


def test_play_implant_percept_shared_scale():
    model = line_model()
    stim = Stimulus({'A': [10, 0], 'C': [0, 20]}, time=[0, 100])
    percept = Percept(np.zeros((3, 3, 2)), time=[0, 100],
                      metadata={'stim': stim})
    ani = play_implant_percept(model, percept)
    cmap = plt.get_cmap(STIM_CMAP)
    # Both frames share one scale; A is not renormalized to its own peak:
    npt.assert_allclose(electrode_pixel(ani, model, 'A', 0)[:3],
                        cmap(0.5)[:3], atol=0.02)
    npt.assert_allclose(electrode_pixel(ani, model, 'C', 1)[:3],
                        cmap(1.0)[:3], atol=0.02)


def test_play_implant_percept_placed_implant():
    model = line_model(implant_rotation=90, implant_position=(500, 0) * um)
    stim = Stimulus({'C': [20, 20]}, time=[0, 100])
    percept = Percept(np.zeros((3, 3, 2)), time=[0, 100],
                      metadata={'stim': stim})
    ani = play_implant_percept(model, percept)
    npt.assert_equal(overlay_pixel(ani, model, (500, 2000), 0,
                                   placed=False)[3] > 0, True)
    # Not where C would sit in the device frame:
    npt.assert_equal(overlay_pixel(ani, model, (2000, 0), 0,
                                   placed=False)[3], 0)


def test_implant_percept_leaves_data_alone():
    model = line_model()
    stim = trace_stim(model)
    percept = model.predict_percept(stim)
    data, time = percept.data.copy(), percept.time.copy()
    stim_data, stim_time = stim.data.copy(), stim.time.copy()
    ani = play_implant_percept(model, percept, annotate=True)
    ani.to_jshtml()
    plot_implant_percept(model, model.predict_percept({'A': 20}))
    npt.assert_equal(percept.data, data)
    npt.assert_equal(percept.time, time)
    npt.assert_equal(stim.data, stim_data)
    npt.assert_equal(stim.time, stim_time)


def test_play_implant_percept_errors():
    model = line_model()
    with pytest.raises(ValueError, match='plot_implant_percept'):
        play_implant_percept(model, model.predict_percept({'A': 20}))
    percept = model.predict_percept(trace_stim(model))
    with pytest.raises(TypeError):
        play_implant_percept(model, percept, percept_kwargs={'kind': 'hex'})
