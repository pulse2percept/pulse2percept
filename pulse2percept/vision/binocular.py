""":py:class:`~pulse2percept.vision.BinocularScene`"""
import numpy as np
import matplotlib.pyplot as plt

from .scene import Scene
from ..stimuli import ImageStimulus, VideoStimulus
from ..utils import PrettyPrint


def _share_visual_field(axes, scenes):
    """Draw both eyes over the same angular extent."""
    half = [max(scene.fov[i] for scene in scenes) / 2 for i in (0, 1)]
    for axis, extent in zip(('x', 'y'), half):
        for ax in axes:
            getattr(ax, f'set_{axis}lim')(-extent, extent)
            # `Percept._label_axes` fixes five ticks across its own range:
            getattr(ax, f'set_{axis}ticks')(np.linspace(-extent, extent,
                                                        num=5))


class BinocularScene(PrettyPrint):
    """The left and right monocular views, side by side

    A :py:class:`~pulse2percept.vision.Scene` is one eye's visual field.
    ``BinocularScene`` holds two of them, and says which eye each belongs to.
    The two sides are equal peers: neither is the implanted or the primary
    one, and they need not share a source, FOV, shape, scotoma or aperture.

    It does *not* model how the visual system combines them. There is no
    fusion, suppression, rivalry, stereopsis or ocular dominance here: this is
    what the left eye sees, and this is what the right eye sees. Plotting them
    is therefore an HMD-style pair of monocular views rather than one
    binocularly fused image.

    Models remain monocular in v0.11, so a prediction names the eye it is
    about::

        percept = model.predict_percept(binocular.left)

    .. versionadded:: 0.11.0

    Parameters
    ----------
    left, right : :py:class:`~pulse2percept.vision.Scene`
        What each eye sees. Stored as given, not copied.

    Examples
    --------
    A unilateral implant: a scotoma in the left eye, the fellow eye intact.

    >>> import numpy as np
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import BinocularScene, Scene, Scotoma
    >>> picture = np.zeros((8, 8))
    >>> binocular = BinocularScene(
    ...     left=Scene(picture, fov=40 * dva, scotoma=Scotoma.circle(8 * dva)),
    ...     right=Scene(picture, fov=40 * dva))
    >>> binocular.left.scotoma is None, binocular.right.scotoma is None
    (False, True)

    A bilateral loss that is symmetric about the vertical meridian, built with
    :py:meth:`~pulse2percept.vision.Scene.fellow_eye`:

    >>> left = Scene(picture, fov=40 * dva,
    ...              scotoma=Scotoma.circle(3 * dva, center=(6, 0) * dva))
    >>> binocular = BinocularScene(left, left.fellow_eye())
    >>> float(binocular.right.scotoma(-6, 0))
    1.0

    Imagery that is already packed left-right side by side, split into the
    two eyes by
    :py:meth:`~pulse2percept.vision.BinocularScene.from_side_by_side`:

    >>> stereo = np.zeros((10, 40))
    >>> stereo[:, 20:] = 1.0
    >>> binocular = BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    >>> binocular.left.shape, binocular.right.shape
    ((10, 20), (10, 20))
    >>> float(binocular.left.source.data.max())
    0.0

    """

    def __init__(self, left, right):
        for name, scene in (('left', left), ('right', right)):
            if not isinstance(scene, Scene):
                raise TypeError(f"'{name}' must be a Scene, not "
                                f"{type(scene)}.")
        self._left = left
        self._right = right

    @classmethod
    def from_side_by_side(cls, source, fov, **scene_kwargs):
        """Split one side-by-side stereo image into the two eyes

        A side-by-side stereo image packs both views into one frame, the left
        eye's in its left half and the right eye's in its right half. This
        splits that frame exactly halfway along its width and hands each half
        to a :py:class:`~pulse2percept.vision.Scene`.

        Nothing is flipped, resampled, cropped or interpolated: the two halves
        keep their pixel values and their grayscale/RGB/RGBA channels. This
        loads existing stereo imagery; it neither creates disparity nor infers
        depth, and the result remains two independent monocular views.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        source : ImageStimulus or image
            The packed stereo frame. Anything that is not already an
            :py:class:`~pulse2percept.stimuli.ImageStimulus`, such as a file
            name or a NumPy array, is handed to ``ImageStimulus``. Stereo
            video is not supported.
        fov : float or (width, height)
            How much of the visual field *one eye's* half covers, in degrees
            of visual angle. The packed frame's width has no visual-field
            meaning, so a scalar (horizontal) FOV gets its vertical extent
            from the aspect ratio of a half, not of the packed frame.
        **scene_kwargs :
            Passed unchanged to both
            :py:class:`~pulse2percept.vision.Scene` constructors, so the two
            eyes get identical scotoma, background, aperture and blend
            settings. Eye-specific geometry is neither inferred nor mirrored
            here; build the two scenes yourself if they should differ.

        Returns
        -------
        binocular : :py:class:`~pulse2percept.vision.BinocularScene`

        Examples
        --------
        >>> import numpy as np
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import BinocularScene
        >>> stereo = np.zeros((10, 40))
        >>> binocular = BinocularScene.from_side_by_side(stereo,
        ...                                              fov=(60, 40) * dva)
        >>> binocular.left.shape, binocular.left.fov
        ((10, 20), (60.0, 40.0))

        """
        if isinstance(source, VideoStimulus):
            raise TypeError("'source' must be a still stereo image, not a "
                            "VideoStimulus: stereo video is not supported.")
        if not isinstance(source, ImageStimulus):
            source = ImageStimulus(source)
        n_cols = source.img_shape[1]
        if n_cols % 2:
            raise ValueError(f"A side-by-side stereo image must split evenly "
                             f"into a left and a right half, so its width "
                             f"must be even, not {n_cols}.")
        halves = np.split(source.data.reshape(source.img_shape), 2, axis=1)
        return cls(*[Scene(ImageStimulus(half), fov=fov, **scene_kwargs)
                     for half in halves])

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'left': self.left, 'right': self.right}

    @property
    def left(self):
        """What the left eye sees, as a Scene"""
        return self._left

    @property
    def right(self):
        """What the right eye sees, as a Scene"""
        return self._right

    def plot(self, left_percept=None, right_percept=None, gaze=None, frame=0,
             rings=False, vmax=None, vmin=0, axes=None, **kwargs):
        """Plot the two eyes side by side

        Each panel is its own :py:meth:`~pulse2percept.vision.Scene.plot`: an
        eye given no percept shows its native or residual scene, and an eye
        given one shows that percept in its own field. The two are drawn
        independently and never combined.

        Parameters
        ----------
        left_percept, right_percept : \
:py:class:`~pulse2percept.percepts.Percept`, optional
            A brightness percept for that eye, or None to draw its scene.
        gaze : (x, y), optional
            Where both eyes are pointing, in dva. Defaults to the origin.
            One gaze is shared: vergence is not modeled.
        frame : int, optional
            Which frame of a video scene to draw. Ignored for still scenes.
        rings : bool, float, or sequence, optional
            Eccentricity rings, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`.
        vmax : float, optional
            The percept brightness that displays as white, shared by both
            eyes. Required whenever either percept is given.
        vmin : float, optional
            The percept brightness that displays as black. Defaults to 0.
        axes : sequence of two matplotlib.axes.Axes, optional
            Axes to draw the left and right eye on, in that order. If None,
            makes a new side-by-side pair.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.vision.Scene.plot`.

        Returns
        -------
        axes : (ax_left, ax_right)
            The two axes, always in left-eye, right-eye order.

        """
        fig = None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=kwargs.pop('figsize',
                                                              (10, 5)))
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError(f"'axes' must be one axes per eye (2 of them), "
                             f"not {len(axes)}.")
        drawn = []
        for ax, label, scene, percept in zip(axes, ('left', 'right'),
                                             (self.left, self.right),
                                             (left_percept, right_percept)):
            # A shared display range only reaches the eye it has a percept
            # for; `Scene.plot` refuses one it has nothing to map.
            scale = {'vmax': vmax, 'vmin': vmin} if percept is not None else {}
            drawn.append(scene.plot(gaze=gaze, frame=frame, ax=ax,
                                    rings=rings, percept=percept, **scale,
                                    **kwargs))
            drawn[-1].set_title(f'{label} eye')
        _share_visual_field(drawn, (self.left, self.right))
        if fig is not None:
            fig.tight_layout()
        return tuple(drawn)
