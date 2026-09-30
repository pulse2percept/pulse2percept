""":py:class:`~pulse2percept.vision.BinocularScene`"""
import numpy as np
import matplotlib.pyplot as plt

from .scene import (Scene, _check_prosthetic, _label_limits, _resolve_view,
                    _CONTEXT_ALPHA, _EYE_VIEW, _SCENE_VIEW)
from ..stimuli import ImageStimulus, VideoStimulus
from ..utils import PrettyPrint


def _share_visual_field(axes, scenes, view):
    """Sets both panels to the same limits: the largest FOV in the eye view,
    the union of both extents in the scene view"""
    if view == _EYE_VIEW:
        half_w, half_h = (max(scene.fov[i] for scene in scenes) / 2
                          for i in (0, 1))
        extent = (-half_w, half_w, -half_h, half_h)
    else:
        extent = (min(scene.extent[0] for scene in scenes),
                  max(scene.extent[1] for scene in scenes),
                  min(scene.extent[2] for scene in scenes),
                  max(scene.extent[3] for scene in scenes))
    for ax in axes:
        _label_limits(ax, extent)


class BinocularScene(PrettyPrint):
    """Left and right monocular views

    Holds one :py:class:`~pulse2percept.vision.Scene` per eye. Neither eye is
    primary, and the two may differ in source, FOV, shape, scotoma, and
    aperture.

    Binocular combination is not modeled (no fusion, suppression, rivalry,
    stereopsis, or ocular dominance). Plots show an HMD-style pair of
    monocular views, not a fused image.

    Models are monocular in v0.11, so predict per eye::

        percept = model.predict_percept(binocular.left)

    .. versionadded:: 0.11.0

    Parameters
    ----------
    left, right : :py:class:`~pulse2percept.vision.Scene`
        Scene for each eye. Stored as given, not copied.

    Examples
    --------
    Unilateral implant: scotoma in the left eye, fellow eye intact:

    >>> import numpy as np
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import BinocularScene, Scene, Scotoma
    >>> picture = np.zeros((8, 8))
    >>> binocular = BinocularScene(
    ...     left=Scene(picture, fov=40 * dva, scotoma=Scotoma.circle(8 * dva)),
    ...     right=Scene(picture, fov=40 * dva))
    >>> binocular.left.scotoma is None, binocular.right.scotoma is None
    (False, True)

    Bilateral loss, mirror-symmetric about the vertical meridian, using
    :py:meth:`~pulse2percept.vision.Scene.fellow_eye`:

    >>> left = Scene(picture, fov=40 * dva,
    ...              scotoma=Scotoma.circle(3 * dva, center=(6, 0) * dva))
    >>> binocular = BinocularScene(left, left.fellow_eye())
    >>> float(binocular.right.scotoma(-6, 0))
    1.0

    Side-by-side stereo imagery, split with
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
        """Build a binocular scene from a side-by-side stereo image

        The left half becomes the left-eye scene, the right half the
        right-eye scene. Neither half is flipped or resampled.

        Does not create disparity or infer depth.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        source : ImageStimulus or image
            Side-by-side stereo image. Other inputs accepted by
            :py:class:`~pulse2percept.stimuli.ImageStimulus` (e.g., file
            names, NumPy arrays) are converted first. Metadata is copied to
            both eyes. Stereo video is not supported.
        fov : float or ``(width, height)``
            Per-eye field of view in dva. Unless ``extent`` is passed, each
            eye's extent is inferred from one half of the image.
        **scene_kwargs :
            Passed to both :py:class:`~pulse2percept.vision.Scene`
            constructors. For per-eye settings (e.g., different scotomas),
            build the two scenes separately.

        Returns
        -------
        binocular : :py:class:`~pulse2percept.vision.BinocularScene`
            Left and right monocular scenes.

        Examples
        --------
        >>> import numpy as np
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import BinocularScene
        >>> stereo = np.zeros((10, 40))
        >>> binocular = BinocularScene.from_side_by_side(
        ...     stereo, fov=(60, 40) * dva)
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
        # Each half keeps the packed frame's metadata (including its
        # `source_shape`), as `ImageStimulus.crop` does:
        return cls(*[Scene(ImageStimulus(half, metadata=source.metadata),
                           fov=fov, **scene_kwargs)
                     for half in halves])

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'left': self.left, 'right': self.right}

    @property
    def left(self):
        """Left-eye Scene"""
        return self._left

    @property
    def right(self):
        """Right-eye Scene"""
        return self._right

    def plot(self, left_percept=None, right_percept=None, gaze=None, frame=0,
             rings=False, meridians=False, vmax=None, vmin=None, axes=None,
             view=_SCENE_VIEW, context_alpha=_CONTEXT_ALPHA, **kwargs):
        """Plot the two eyes side by side

        Each panel is a separate :py:meth:`~pulse2percept.vision.Scene.plot`
        call: without a percept, the eye's native or residual scene; with
        one, that percept in the eye's field. Eyes are never combined.

        Parameters
        ----------
        left_percept, right_percept : \
:py:class:`~pulse2percept.percepts.Percept`, optional
            Brightness percept for that eye, or None to draw its scene.
        gaze : (x, y), optional
            Scene location (dva) on both foveas. Defaults to the origin.
            Shared by both eyes (vergence is not modeled).
        frame : int, optional
            Frame of a video scene to draw. Ignored for still scenes.
        rings, meridians : bool, float, or sequence, optional
            Visual-field grid centered on each eye's fovea, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`.
        vmax : float, optional
            Percept brightness shown as white, shared by both eyes. Defaults
            to the maximum over both percepts.
        vmin : float, optional
            Percept brightness shown as black. Defaults to 0.
        axes : sequence of two matplotlib.axes.Axes, optional
            Axes to draw the left and right eye on, in that order. If None,
            makes a new side-by-side pair.
        view : {'scene', 'eye'}, optional
            As in :py:meth:`~pulse2percept.vision.Scene.plot`. Both panels
            share limits: the union of both ``extent`` values in the scene
            view (default), the largest FOV in the eye view.
        context_alpha : float, optional
            Opacity of the source outside each FOV in the scene view, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.vision.Scene.plot`.

        Returns
        -------
        axes : (ax_left, ax_right)
            Left-eye and right-eye axes, in that order.

        """
        view = _resolve_view(view)
        fig = None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=kwargs.pop('figsize',
                                                              (10, 5)))
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError(f"'axes' must be one axes per eye (2 of them), "
                             f"not {len(axes)}.")
        percepts = [p for p in (left_percept, right_percept) if p is not None]
        for percept in percepts:
            _check_prosthetic(percept)
        if vmax is None and percepts:
            # One scale for both eyes, so brightness is comparable:
            vmax = max(np.max(p.data) for p in percepts)
            if vmin is None and vmax == 0:
                # All-zero percepts: let `Scene.plot` draw them black:
                vmax = None
        drawn = []
        for ax, label, scene, percept in zip(axes, ('left', 'right'),
                                             (self.left, self.right),
                                             (left_percept, right_percept)):
            # Pass vmin/vmax only with a percept; `Scene.plot` rejects them
            # otherwise:
            scale = {'vmax': vmax, 'vmin': vmin} if percept is not None else {}
            drawn.append(scene.plot(gaze=gaze, frame=frame, ax=ax,
                                    rings=rings, meridians=meridians,
                                    percept=percept, view=view,
                                    context_alpha=context_alpha, **scale,
                                    **kwargs))
            drawn[-1].set_title(f'{label} eye')
        _share_visual_field(drawn, (self.left, self.right), view)
        if fig is not None:
            fig.tight_layout()
        return tuple(drawn)
