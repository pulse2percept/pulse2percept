""":py:class:`~pulse2percept.vision.BinocularScene`"""
import numpy as np
import matplotlib.pyplot as plt

from .scene import Scene
from ..utils import PrettyPrint


def _share_visual_field(axes):
    """Put both axes on the same degrees-per-inch scale

    Each `Scene.plot` scales its axes to its own FOV, which would draw a
    30-degree and a 50-degree field at the same size on screen. Widening both
    to the union of the two ranges keeps one degree the same length in either
    panel. Presentation only: each image keeps its own extent.
    """
    for axis in ('x', 'y'):
        limits = [getattr(ax, f'get_{axis}lim')() for ax in axes]
        lo = min(min(lim) for lim in limits)
        hi = max(max(lim) for lim in limits)
        for ax in axes:
            getattr(ax, f'set_{axis}lim')(lo, hi)
            # `Percept._label_axes` fixes five ticks across its own range:
            getattr(ax, f'set_{axis}ticks')(np.linspace(lo, hi, num=5))


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

    """

    def __init__(self, left, right):
        for name, scene in (('left', left), ('right', right)):
            if not isinstance(scene, Scene):
                raise TypeError(f"'{name}' must be a Scene, not "
                                f"{type(scene)}.")
        self._left = left
        self._right = right

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
        _share_visual_field(drawn)
        if fig is not None:
            fig.tight_layout()
        return tuple(drawn)
