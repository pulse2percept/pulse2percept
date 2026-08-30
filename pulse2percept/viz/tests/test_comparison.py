import numpy as np
import numpy.testing as npt
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, Stimulus, VideoStimulus
from pulse2percept.units import Hz, s
from pulse2percept.viz import play_stimulus_percept, plot_stimulus_percept


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


def test_plot_stimulus_percept_video():
    """A video is summarized by its brightest frame, like a percept is"""
    vid = video()
    axes = plot_stimulus_percept(vid, percept())
    npt.assert_almost_equal(axes[0].images[0].get_array(),
                            vid._frames()[..., -1])


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
    plot_stimulus_percept(vid, perc)
    npt.assert_almost_equal(vid.data, stim_data)
    npt.assert_almost_equal(vid.time, stim_time)
    npt.assert_almost_equal(perc.data, perc_data)
    npt.assert_almost_equal(perc.time, perc_time)


def test_play_stimulus_percept_errors():
    # A percept without a time axis is a still image:
    with pytest.raises(ValueError):
        play_stimulus_percept(ImageStimulus(np.random.rand(4, 4)),
                              Percept(np.random.rand(3, 3, 1)))
    # The electrical stimulus an encoder made is not the source picture:
    with pytest.raises(TypeError):
        play_stimulus_percept(Stimulus({'A1': 1}), percept())
