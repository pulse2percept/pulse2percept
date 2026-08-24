""":py:class:`~pulse2percept.vision.Scene`"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from skimage.color import rgb2gray

from .scotoma import Scotoma
from ..percepts import Percept
from ..stimuli import ImageStimulus, VideoStimulus
from ..topography import Grid2D
from ..units import as_value, dimensionless, dva
from ..utils import PrettyPrint


def _resolve_fov(fov, n_rows, n_cols):
    """Normalize a user-supplied ``fov`` to ``(width, height)`` in dva

    A scalar is the horizontal FOV; the vertical one follows from the frame's
    aspect ratio, which is the same as assuming square angular pixels.
    """
    fov = np.asarray(as_value(fov, dva, 'fov'), dtype=float)
    if fov.ndim == 0:
        width = float(fov)
        height = width * n_rows / n_cols
    elif fov.shape == (2,):
        width, height = (float(f) for f in fov)
    else:
        raise ValueError(f"'fov' must be a scalar (horizontal FOV) or a "
                         f"(width, height) pair, not {fov.tolist()}.")
    for name, f in (('width', width), ('height', height)):
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"'fov' {name} must be a finite positive number "
                             f"of degrees, not {f}.")
    return (width, height)


def _gaze_points(gaze, n_frames):
    """Gaze as one (x, y) in dva, or one per frame"""
    if gaze is None:
        return np.zeros((1, 2))
    gaze = np.atleast_2d(np.asarray(as_value(gaze, dva, 'gaze'), dtype=float))
    if gaze.shape not in {(1, 2), (n_frames, 2)}:
        raise ValueError(f"'gaze' must be an (x, y) pair in dva, or one per "
                         f"frame ({n_frames} of them), not an array of shape "
                         f"{gaze.shape}.")
    if not np.all(np.isfinite(gaze)):
        # Left to reach the interpolator, this would come back as a blank
        # percept rather than as a question about where the eye was pointing:
        raise ValueError(f"'gaze' must be finite, not {gaze.tolist()}.")
    return gaze


def _clip_to_frame(points, shape):
    """Clip pixel coordinates onto the frame, and say which were on it

    A scene's field of view is its *outer* extent, so it reaches half a pixel
    past the outermost pixel centers (see ``pixel_to_dva``). Interpolation
    stops at those centers, and the border strip between them and the edge
    still belongs to the scene: a point there takes the value of the pixel it
    is inside. Past the outer edge there is no scene to sample, which is what
    ``inside`` is for -- nothing is extrapolated.
    """
    points = np.asarray(points, dtype=float)
    edges = np.asarray(shape[:2], dtype=float) - 0.5
    inside = np.all((points >= -0.5) & (points <= edges), axis=1)
    # A point off the frame is not interpolated at all, so anything goes here
    # as long as it is on the grid:
    on_grid = np.where(inside[:, np.newaxis], points, 0.0)
    return np.clip(on_grid, 0.0, edges - 0.5), inside


def _interpolate(grid, frames, points):
    """Sample ``frames`` at ``points``, carrying the trailing axes along

    One interpolator covers every channel and every frame: the grid is the
    leading two axes, and anything past them comes back as trailing axes of
    the result.
    """
    interpolator = RegularGridInterpolator(grid, frames, method='linear',
                                           bounds_error=False, fill_value=0)
    return interpolator(points)


def _drop_gray_axis(values):
    """Give back the (n_points, n_frames) a grayscale scene samples to

    Sampling runs on one (rows, cols, channels, frames) layout so that color
    and gray take the same path through the interpolator; a single channel is
    not a channel once it comes out the other side.
    """
    return values[:, 0] if values.shape[1] == 1 else values


class Scene(PrettyPrint):
    """What is visually present, and where native vision is lost

    A scene places a picture in the visual field: it says how much of the
    field the picture subtends, and optionally where in that field native
    vision is missing. That is enough for a model to work out what an
    implanted eye sees, without the caller converting anything by hand.

    Geometry follows one convention:

    *  ``fov`` is the *outer* angular extent of the frame, centered on it, so
       it reaches half an angular pixel past the outermost pixel centers.
    *  Pixel coordinates address pixel *centers*.
    *  Row 0 is the top of the frame and therefore the largest ``y``.

    The scotoma is *eye-centered*: it is fixed relative to the fovea, and so
    is an implant, which sits on the retina. Gaze moves the scene past both of
    them rather than moving either.

    A scene is read-only once built. ``fov`` is resolved against the source's
    frame shape, so swapping one out without the other would leave the
    geometry describing a picture that is no longer there.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    source : ImageStimulus, VideoStimulus, or image
        The picture itself. Anything that is not already a
        :py:class:`~pulse2percept.stimuli.ImageStimulus` or a
        :py:class:`~pulse2percept.stimuli.VideoStimulus` (a file name, a NumPy
        array) is handed to ``ImageStimulus``.
    fov : float or (width, height)
        How much of the visual field the source covers, in degrees of visual
        angle (e.g. ``40 * dva``). A scalar is the horizontal extent, and the
        vertical one follows from the frame's aspect ratio.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`, optional
        Where native vision is lost. If None, native vision is intact
        everywhere and the scene is simply what is out there.
    scotoma_fill : float, optional
        What complete loss looks like, as a display intensity in [0, 1].
        Defaults to black. What is lost has to look like *something*; this is
        the choice, and it belongs to the scene rather than to a model run.

    Examples
    --------
    A logo covering 40 degrees, seen with a central 16-degree scotoma:

    >>> from pulse2percept.stimuli import LogoBVL
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import Scene, Scotoma
    >>> scene = Scene(LogoBVL(), fov=40 * dva, scotoma=Scotoma.circle(8 * dva))
    >>> scene.fov
    (40.0, 32.0)

    """

    def __init__(self, source, fov, scotoma=None, scotoma_fill=0):
        if not isinstance(source, (ImageStimulus, VideoStimulus)):
            # A picture is the common case, and asking for the wrapper adds
            # nothing; a video has to be built by the caller because only they
            # know its frame times.
            source = ImageStimulus(source)
        if scotoma is not None and not isinstance(scotoma, Scotoma):
            raise TypeError(f"'scotoma' must be a Scotoma object, not "
                            f"{type(scotoma)}.")
        fill = float(as_value(scotoma_fill, dimensionless,
                              'scotoma_fill'))
        if not np.isfinite(fill) or fill < 0 or fill > 1:
            raise ValueError(f"'scotoma_fill' is a display intensity and must "
                             f"lie in [0, 1], not {scotoma_fill}.")
        self._source = source
        self._scotoma = scotoma
        self._scotoma_fill = fill
        n_rows, n_cols = self._frame_shape
        self._fov = _resolve_fov(fov, n_rows, n_cols)
        self._cached_frames = None

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'source': type(self.source).__name__, 'fov': self.fov,
                'shape': self.shape, 'scotoma': self.scotoma,
                'scotoma_fill': self.scotoma_fill}

    @property
    def source(self):
        """The picture itself, as an ImageStimulus or a VideoStimulus"""
        return self._source

    @property
    def scotoma(self):
        """Where native vision is lost, or None if it is intact"""
        return self._scotoma

    @property
    def scotoma_fill(self):
        """The display intensity complete loss shows as"""
        return self._scotoma_fill

    @property
    def _frame_shape(self):
        """The (rows, cols) of one frame of the source"""
        if isinstance(self.source, ImageStimulus):
            return tuple(self.source.img_shape[:2])
        return tuple(self.source.vid_shape[:2])

    @property
    def fov(self):
        """Field of view ``(width, height)``, in degrees of visual angle"""
        return self._fov

    @property
    def shape(self):
        """The ``(rows, cols)`` of one frame"""
        return self._frame_shape

    @property
    def time(self):
        """Frame times of the source, or None for a still scene"""
        return self.source.time

    @property
    def time_unit(self):
        """The unit ``time`` is counted in"""
        return self.source.time_unit

    @property
    def _angular_pixel(self):
        """Angular size ``(width, height)`` of one pixel, in dva"""
        n_rows, n_cols = self._frame_shape
        return (self._fov[0] / n_cols, self._fov[1] / n_rows)

    def pixel_to_dva(self, col, row):
        """Visual-field coordinates of a pixel center

        Parameters
        ----------
        col, row : float or array_like
            Pixel coordinates, where ``(0, 0)`` is the center of the top-left
            pixel. Fractional values address points between pixel centers.

        Returns
        -------
        x, y : np.ndarray
            Scene coordinates in degrees of visual angle, relative to the
            center of the frame. ``y`` grows upwards, so row 0 has the largest
            ``y``.

        """
        dx, dy = self._angular_pixel
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        x = (col + 0.5) * dx - self._fov[0] / 2
        y = self._fov[1] / 2 - (row + 0.5) * dy
        return x, y

    def dva_to_pixel(self, x, y):
        """Pixel coordinates of a point in the scene

        The inverse of :py:meth:`~pulse2percept.vision.Scene.pixel_to_dva`.

        Parameters
        ----------
        x, y : float or array_like
            Scene coordinates in degrees of visual angle, relative to the
            center of the frame.

        Returns
        -------
        col, row : np.ndarray
            Continuous pixel coordinates, where ``(0, 0)`` is the center of the
            top-left pixel. They are not rounded and not clipped to the frame:
            a point outside the FOV maps outside the pixel grid.

        """
        dx, dy = self._angular_pixel
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x + self._fov[0] / 2) / dx - 0.5
        row = (self._fov[1] / 2 - y) / dy - 0.5
        return col, row

    def _frames(self):
        """The source as a dense ``(rows, cols, channels, n_frames)`` array

        The one place that knows how an image or a video stores its pixels.
        Everything downstream treats a still as a one-frame movie, and an
        alpha channel is blended against black here, as it is everywhere else
        in p2p, because nothing further along knows what to do with it.
        """
        if self._cached_frames is not None:
            return self._cached_frames
        source = self.source
        if isinstance(source, ImageStimulus):
            frames = source.data.reshape(source.img_shape)[..., np.newaxis]
        else:
            frames = source.data.reshape(source.vid_shape)
        if frames.ndim == 3:
            # Grayscale: give it the channel axis the color path already has
            frames = frames[:, :, np.newaxis, :]
        if frames.shape[2] == 4:
            frames = np.clip(frames[:, :, :3] * frames[:, :, 3:4], 0, 1)
        elif frames.shape[2] not in (1, 3):
            raise ValueError(f"A scene must be grayscale, RGB or RGBA, not "
                             f"{frames.shape[2]}-channel.")
        frames = np.asarray(frames, dtype=np.float32)
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Scene values are display intensities and must "
                             f"lie in [0, 1], but this one spans "
                             f"[{frames.min():g}, {frames.max():g}].")
        self._cached_frames = frames
        return frames

    @property
    def n_frames(self):
        """How many frames the source has; 1 for a still scene"""
        return self._frames().shape[-1]

    def _sample_at(self, x, y, gaze=None):
        """What the scene shows at eye-centered visual-field positions

        The spatial half of turning a scene into stimulation, and the seam a
        color encoder would need: color channels are preserved here, and
        reducing them to one number is a separate step (see
        :py:meth:`_device_input`).

        Parameters
        ----------
        x, y : array_like
            Eye-centered visual-field coordinates in dva -- where the viewer
            is looking *from*, not where the picture is. ``gaze`` is what puts
            them in the scene: ``scene = visual field + gaze``.
        gaze : (x, y) or (n_frames, 2), optional
            Where the eye is pointing. One pair fixates for the whole source;
            one pair per frame moves the eye between frames.

        Returns
        -------
        values : (n_points, n_frames) or (n_points, 3, n_frames) array
            Zero wherever a position falls outside the scene: there is no
            picture there, and none is extrapolated.

        """
        frames = self._frames()
        gaze = _gaze_points(gaze, frames.shape[-1])
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        # The pixel grid is the only geometry the interpolator needs; the rest
        # of it lives in `dva_to_pixel`:
        grid = (np.arange(frames.shape[0], dtype=float),
                np.arange(frames.shape[1], dtype=float))
        sampled = []
        # One gaze samples every frame at once; a gaze per frame samples each
        # frame where the eye was pointing when it came up.
        for f, (gx, gy) in enumerate(gaze):
            col, row = self.dva_to_pixel(x + gx, y + gy)
            points, inside = _clip_to_frame(np.column_stack((row, col)),
                                            frames.shape)
            values = _interpolate(grid, frames if len(gaze) == 1
                                  else frames[..., f], points)
            values[~inside] = 0
            sampled.append(values)
        if len(sampled) == 1:
            return _drop_gray_axis(sampled[0])
        return _drop_gray_axis(np.stack(sampled, axis=-1))

    def _device_input(self, x, y, gaze=None):
        """One number per position per frame, for a device to encode

        The Scene-to-device seam. Today's stimulus encoders are luminance
        encoders, so this is where color stops.

        Returns
        -------
        gray : (n_points, n_frames) array
        """
        values = self._sample_at(x, y, gaze=gaze)
        if values.ndim == 2:
            return values
        # `rgb2gray` wants the channels last; the positions and the frames are
        # both just pixels to it:
        return rgb2gray(values.transpose((0, 2, 1)))

    def _native_rgb(self, gaze=None):
        """What is left of native vision, as ``(rows, cols, 3, n_frames)``

        The scene where vision is intact, ``scotoma_fill`` where it is
        completely lost, and a linear mix of the two where a graded scotoma
        says vision is partly there.
        """
        frames = self._frames()
        if frames.shape[2] == 1:
            frames = np.repeat(frames, 3, axis=2)
        if self.scotoma is None:
            return frames
        n_frames = frames.shape[-1]
        gaze = _gaze_points(gaze, n_frames)
        x_scene, y_scene = self._pixel_centers()
        fill = self.scotoma_fill
        out = np.empty(frames.shape, dtype=np.float32)
        for f in range(n_frames):
            # `scene = visual field + gaze`, run backwards: where each scene
            # pixel falls relative to the fovea, which is where the scotoma is.
            gx, gy = gaze[0] if len(gaze) == 1 else gaze[f]
            loss = self.scotoma(x_scene - gx, y_scene - gy)[..., np.newaxis]
            out[..., f] = (1 - loss) * frames[..., f] + loss * fill
        return out

    def _pixel_centers(self):
        """Scene coordinates of every pixel center, as ``(x, y)`` meshes"""
        n_rows, n_cols = self._frame_shape
        cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        return self.pixel_to_dva(cols, rows)

    def _grid(self):
        """A Grid2D on the scene's pixel centers, in scene coordinates

        Gaze moves where the eye-centered scotoma and percept land on the
        scene; it does not move the scene, so this is the one coordinate
        system a composed result is reported in.
        """
        n_rows, n_cols = self._frame_shape
        x_left, y_top = self.pixel_to_dva(0, 0)
        x_right, y_bottom = self.pixel_to_dva(n_cols - 1, n_rows - 1)
        step = (float(x_right - x_left) / (n_cols - 1) if n_cols > 1 else 1.0,
                float(y_top - y_bottom) / (n_rows - 1) if n_rows > 1 else 1.0)
        return Grid2D((float(x_left), float(x_right)),
                      (float(y_bottom), float(y_top)), step=step)

    def _native_percept(self, gaze=None):
        """Residual native vision as an ordinary RGB percept

        Which is what makes drawing and animating a scene the same problem as
        drawing and animating anything else p2p produces.
        """
        return Percept(self._native_rgb(gaze=gaze), space=self._grid(),
                       time=self.time, time_unit=self.time_unit)

    def plot(self, gaze=None, frame=0, ax=None, **kwargs):
        """Plot what is left of native vision

        The scene unchanged where vision is intact, and ``scotoma_fill`` where
        it is lost. A scotoma is eye-centered, so ``gaze`` decides where in the
        scene it falls.

        Parameters
        ----------
        gaze : (x, y), optional
            Where the eye is pointing: the scene location that falls on the
            fovea, in dva. Defaults to the origin.
        frame : int, optional
            Which frame of a video scene to draw. Ignored for a still scene.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. If None, uses the current axes.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        rgb = self._native_rgb(gaze=gaze)
        if not 0 <= frame < rgb.shape[-1]:
            raise ValueError(f"'frame' must be in 0..{rgb.shape[-1] - 1}, not "
                             f"{frame}.")
        still = Percept(rgb[..., frame:frame + 1], space=self._grid())
        return still.plot(ax=ax, **kwargs)

    def play(self, gaze=None, **kwargs):
        """Animate a video scene as it is natively seen

        Parameters
        ----------
        gaze : (x, y) or (n_frames, 2), optional
            Where the eye is pointing, in dva. One pair fixates throughout;
            one pair per frame moves the eye between frames.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.play`.

        Returns
        -------
        ani : :py:class:`~pulse2percept.utils.HTMLAnimation`

        """
        if self.time is None:
            raise ValueError("A still scene has nothing to play. Use plot().")
        return self._native_percept(gaze=gaze).play(**kwargs)
