""":py:func:`~pulse2percept.vision.compose_hybrid_vision`"""
from .scene import Scene
from .scotoma import Scotoma
from ..stimuli import ImageStimulus, VideoStimulus


def compose_hybrid_vision(scene, prosthetic, scotoma, vmax, vmin=0, gaze=None,
                          scotoma_fill=0):
    """Compose native and prosthetic vision into one RGB percept

    What someone with an eye-centered scotoma and a retinal implant sees:
    intact vision outside the scotoma, and the modeled percept inside it.

    .. deprecated:: 0.11.0

        Composition is what a :py:class:`~pulse2percept.vision.Scene` with a
        scotoma does on its own, so
        :py:meth:`~pulse2percept.models.Model.predict_percept` returns the
        composed percept directly. This function is the same operation spelled
        out by hand, and is going away.

    Parameters
    ----------
    scene : ImageStimulus or VideoStimulus
        What is out there to be seen. Must state a ``fov``.
    prosthetic : :py:class:`~pulse2percept.percepts.Percept`
        The modeled percept, in arbitrary brightness units.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`
        Where native vision is lost, and how much of it.
    vmax : float
        The brightness that displays as white.
    vmin : float, optional
        The brightness that displays as black.
    gaze : (x, y) or (n_frames, 2), optional
        Where the eye is pointing, in degrees of visual angle.
    scotoma_fill : float, optional
        What complete loss looks like where there is no percept, as a display
        intensity in [0, 1].

    Returns
    -------
    percept : :py:class:`~pulse2percept.percepts.Percept`
        An RGB percept of shape ``(Y, X, 3, T)`` on the scene's pixel grid.

    """
    if not isinstance(scene, (ImageStimulus, VideoStimulus)):
        raise TypeError(f"'scene' must be an ImageStimulus or a "
                        f"VideoStimulus, not {type(scene)}.")
    if scene.fov is None:
        raise ValueError("A scene must state the 'fov' it subtends before it "
                         "can be composed with an eye-centered scotoma.")
    if not isinstance(scotoma, Scotoma):
        raise TypeError(f"'scotoma' must be a Scotoma, not {type(scotoma)}.")
    placed = Scene(scene, fov=scene.fov, scotoma=scotoma,
                   scotoma_fill=scotoma_fill)
    return placed._compose(prosthetic, vmax, vmin=vmin, gaze=gaze)
