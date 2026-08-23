""":py:class:`~pulse2percept.implants.ProsthesisSystem`,
   :py:class:`~pulse2percept.implants.RectangleImplant`"""
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from scipy.interpolate import RegularGridInterpolator

from .electrodes import Electrode, DiskElectrode
from .electrode_arrays import ElectrodeArray, ElectrodeGrid
from .rasters import Raster
from ..stimuli import (BiphasicPulseTrain, Stimulus, ImageStimulus,
                       StimulusEncoder, VideoStimulus)
from ..stimuli.base import _describe_unit
from ..stimuli.pulse_trains import _as_threshold_amp
from ..units import DimensionMismatchError, as_value, uA, um
from ..utils import PrettyPrint


class ProsthesisSystem(PrettyPrint):
    """Visual prosthesis system

    A visual prosthesis combines an electrode array and (optionally) a
    stimulus. This is the base class for prosthesis systems such as
    :py:class:`~pulse2percept.implants.ArgusII` and
    :py:class:`~pulse2percept.implants.AlphaIMS`.

    .. versionadded:: 0.6

    Parameters
    ----------
    earray : :py:class:`~pulse2percept.implants.ElectrodeArray` or
             :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to the retina.
    stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
        A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
        object (e.g., scalar, NumPy array, pulse train). It must be electrical
        (see
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stimulus_unit`), or
        an image or a video that the implant's ``encoder`` turns into one.

        .. versionchanged:: 0.10.0
            A stimulus that is neither a current nor something the implant's
            ``encoder`` can turn into one is refused here, instead of at the
            point where a model tries to read it.
    eye : 'LE' or 'RE'
        A string indicating whether the system is implanted in the left ('LE')
        or right eye ('RE')
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.
        Safety is an electrical property, so this also requires the stimulus
        to be measured in units of current.
    encoder : :py:class:`~pulse2percept.stimuli.StimulusEncoder`, optional
        How the device turns a picture into stimulation. If given, assigning an
        image or a video to
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim` encodes it
        first, so that ``implant.stim`` always ends up electrical. If None,
        such a stimulus is refused, since there is no principled default
        mapping from a gray level to an amplitude or a frequency.

        .. versionadded:: 0.10.0
    raster : :py:class:`~pulse2percept.implants.Raster`, optional
        How the stimulator takes turns between electrodes that it cannot drive
        at the same time. If None, every electrode may fire at once. Assigning
        one binds it to this implant (see
        :py:meth:`~pulse2percept.implants.Raster.bind`), and it is the raster
        the ``encoder`` schedules against.

        .. versionadded:: 0.10.0
    max_current : float, optional
        The total current (uA) the stimulator can source at any one instant,
        summed over all electrodes. If given, assigning a stimulus that exceeds
        it raises. If None, no such check is performed.

        May be given as a plain number of microamps or as a unitful quantity
        (e.g. ``0.1 * mA``); see :py:mod:`pulse2percept.units`.

        .. versionadded:: 0.10.0

    Examples
    --------
    A system in the left eye made from a single
    :py:class:`~pulse2percept.implants.DiskElectrode` with radius
    r=100um sitting at x=200um, y=-50um, z=10um:

    >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
    >>> implant = ProsthesisSystem(DiskElectrode(200, -50, 10, 100), eye='LE')

    .. note::

        A stimulus can also be assigned later (see
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim`).

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_earray', '_stim', '_eye', 'safe_mode', 'preprocess',
                 '_encoder', '_raster', '_max_current', '_thresholds')

    #: The physical quantity this system delivers, mirroring
    #: :py:attr:`~pulse2percept.models.BaseModel.stimulus_unit`. An electrical
    #: prosthesis delivers current, so a stimulus that is not a current is not
    #: a stimulus it can deliver, and is refused when it is assigned rather
    #: than when a model eventually tries to read it. Only the *dimension* has
    #: to match: `Stimulus` has already converted an amplitude given as
    #: ``0.05 * mA`` into 50 uA by the time it gets here. A device that
    #: delivered something else (light, say) would override this.
    stimulus_unit = uA

    def __init__(self, earray, stim=None, eye='RE', preprocess=False,
                 safe_mode=False, encoder=None, raster=None, max_current=None):
        self.earray = earray
        self.eye = eye
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.encoder = encoder
        self.raster = raster
        self.max_current = max_current
        # Beware of race condition: the stimulus goes last, because assigning
        # it may have to encode a picture, which needs the encoder, the raster
        # and the electrode array to be in place already.
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = {
            'earray': self.earray, 'stim': self.stim, 'safe_mode': self.safe_mode,
            'preprocess': self.preprocess
        }
        if hasattr(self, "eye"):
            params['eye'] = self.eye
        if self.encoder is not None:
            params['encoder'] = self.encoder
        if self.raster is not None:
            params['raster'] = self.raster
        if self.max_current is not None:
            params['max_current'] = self.max_current
        if self.thresholds:
            params['thresholds'] = self.thresholds
        return params

    @property
    def encoder(self):
        """Stimulus encoder

        How this device turns a picture into stimulation. Most implants do not
        set one in their constructor, so the slot backing it may never have
        been written to; no encoder means an image or a video assigned to
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim` is refused
        rather than silently given a modulation scheme nobody chose.
        """
        return getattr(self, '_encoder', None)

    @encoder.setter
    def encoder(self, encoder):
        """Encoder setter (called upon ``self.encoder = encoder``)"""
        if encoder is not None and not isinstance(encoder, StimulusEncoder):
            raise TypeError(f"'encoder' must be a StimulusEncoder object, not "
                            f"{type(encoder)}.")
        self._encoder = encoder

    @property
    def raster(self):
        """Raster pattern

        Most implants do not set this in their constructor, so the slot backing
        it may never have been written to; an unset raster means all electrodes
        may fire at once.

        Assigning a raster binds it to this implant (see
        :py:meth:`~pulse2percept.implants.Raster.bind`), which is what lets a
        geometry-dependent pattern such as
        :py:class:`~pulse2percept.implants.CheckerboardRaster` work itself out
        and what lets :py:meth:`~pulse2percept.implants.Raster.plot` be called
        without an argument.
        """
        return getattr(self, '_raster', None)

    @raster.setter
    def raster(self, raster):
        """Raster setter (called upon ``self.raster = raster``)"""
        if raster is not None:
            if not isinstance(raster, Raster):
                raise TypeError(f"'raster' must be a Raster object, not "
                                f"{type(raster)}.")
            # Before the slot is written to, so that a raster that cannot be
            # laid out on this array leaves the implant as it was:
            raster.bind(self)
        self._raster = raster

    @property
    def max_current(self):
        """Total instantaneous current (uA) the stimulator can source"""
        return getattr(self, '_max_current', None)

    @max_current.setter
    def max_current(self, max_current):
        """Current limit setter (called upon ``self.max_current = ...``)"""
        max_current = as_value(max_current, uA, 'max_current')
        if max_current is not None and max_current <= 0:
            raise ValueError("'max_current' must be positive.")
        self._max_current = max_current

    @property
    def thresholds(self):
        """Perceptual threshold current (uA) for each electrode

        Calibration of the electrode/subject interface, relating stimulation
        expressed in :py:data:`~pulse2percept.units.xTh` to the current that
        realizes it. Assign one current for the whole array, a dict for
        per-electrode values (a partial dict leaves the rest uncalibrated), or
        None to clear.

        Changing thresholds recalibrates any
        :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` already assigned
        to :py:attr:`stim`: ``xTh`` trains keep their multiple of threshold,
        current-valued trains keep their current. Sampled stimuli are
        unchanged.

        .. versionadded:: 0.10.0

        Examples
        --------
        >>> from pulse2percept.implants import ArgusII
        >>> from pulse2percept.stimuli import BiphasicPulseTrain
        >>> from pulse2percept.units import uA, xTh
        >>> implant = ArgusII()
        >>> implant.stim = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
        ...                 'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
        >>> implant.thresholds = {'A1': 80 * uA, 'A2': 120 * uA}
        >>> float(abs(implant.stim['A1']).max())
        160.0
        >>> float(abs(implant.stim['A2']).max())
        240.0

        """
        return dict(getattr(self, '_thresholds', None) or {})

    @thresholds.setter
    def thresholds(self, thresholds):
        """Threshold setter (called upon ``self.thresholds = ...``)"""
        previous = getattr(self, '_thresholds', {})
        self._thresholds = self._normalize_thresholds(thresholds)
        try:
            self._recalibrate_stim()
        except Exception:
            # Thresholds and stimulus move together or not at all:
            self._thresholds = previous
            raise

    def _normalize_thresholds(self, thresholds):
        """Return thresholds as a mapping of electrode names to uA."""
        if thresholds is None:
            return {}
        if not isinstance(thresholds, dict):
            threshold = _as_threshold_amp(thresholds, 'thresholds')
            return {name: threshold for name in self.electrode_names}
        normalized = {}
        for name, threshold in thresholds.items():
            if name not in self.electrodes:
                raise ValueError(f'Electrode "{name}" not found in implant.')
            threshold = _as_threshold_amp(threshold, f'thresholds[{name!r}]')
            # Omitting an electrode already means "uncalibrated", so None is
            # not stored as a second way of saying it:
            if threshold is not None:
                normalized[name] = threshold
        return normalized

    def _recalibrate_stim(self):
        """Recalibrate the stored stimulus for the thresholds now in force"""
        stim = getattr(self, '_stim', None)
        if stim is None:
            return
        calibrated = self._calibrated(stim)
        if calibrated is stim:
            return
        # Checked before it is stored, so a rejected stimulus is not the one
        # left behind:
        self.check_stim(calibrated)
        self._stim = calibrated

    def _calibrated(self, stim):
        """``stim`` with this implant's thresholds applied to its pulse trains

        Rebuilds the pulse trains the stimulus retains rather than rewriting
        its samples, generating no waveform, and returns ``stim`` itself when
        nothing needs rebuilding. An electrode the thresholds leave out gets
        None, so this clears a calibration as readily as it applies one.
        """
        thresholds = getattr(self, '_thresholds', None) or {}
        sources = stim._structured_sources()
        if sources is None:
            return stim
        rebuilt, changed = {}, False
        for name, source in sources:
            train = (source._with_threshold(thresholds.get(name))
                     if type(source) is BiphasicPulseTrain else source)
            changed = changed or train is not source
            rebuilt[name] = train
        if not changed:
            return stim
        if len(sources) == 1 and sources[0][1] is stim:
            # The stimulus *is* the pulse train, and must stay that kind of
            # object rather than become a collection of one:
            return rebuilt[sources[0][0]]
        return Stimulus(rebuilt,
                        metadata=deepcopy(stim.metadata.get('user')))

    def _require_deliverable_stim(self, stim):
        """Refuse a stimulus this system cannot physically deliver

        The implant's half of the boundary that
        :py:func:`~pulse2percept.models.base._require_stim_dimension` guards on
        the model's side. Both are needed: a model must not read gray levels as
        microamps whatever produced them, and an implant that accepted a
        picture would only fail once a model got hold of it, several steps
        away from the assignment that was actually wrong.

        Checked on the *final* stimulus, after ``preprocess`` and after any
        reshaping onto the electrode grid, so that a preprocessing step that
        turns an image into current is exactly as valid as encoding it first.
        """
        if stim.unit.dimension == self.stimulus_unit.dimension:
            return
        raise DimensionMismatchError(
            f"{type(self).__name__} delivers "
            f"{_describe_unit(self.stimulus_unit)}, but this stimulus is "
            f"measured in {_describe_unit(stim.unit)}. Give the implant an "
            f"'encoder' (pulse2percept.stimuli.AmplitudeEncoder or "
            f"FrequencyEncoder) so that image or video input is encoded on "
            f"assignment, encode it yourself first, or give the implant a "
            f"'preprocess' function that does.")

    @staticmethod
    def _require_current_stim(stim, check):
        """Refuse to run an electrical safety check on something that isn't

        Called by each check rather than by ``check_stim``, because a safety
        check is a claim about electricity and each one has to say so for
        itself: ``check_stim`` is public, and is called directly by tests and
        by subclasses on stimuli that never went through the setter.
        """
        if stim.unit.dimension != uA.dimension:
            raise DimensionMismatchError(
                f"Safety check '{check}' needs an electrical stimulus to "
                f"check, and this one is measured in "
                f"{_describe_unit(stim.unit)}. Encode it into current first "
                f"(see pulse2percept.stimuli.StimulusEncoder), or give the "
                f"implant a 'preprocess' function that does.")

    @classmethod
    def _require_charge_balanced(cls, stim):
        cls._require_current_stim(stim, 'safe_mode')
        # `is False` rather than `not`: the property answers None when the
        # question does not apply, which the guard above has already ruled out
        # here but which must never be read as "unbalanced".
        if stim.is_charge_balanced is False:
            raise ValueError("Safety check: Stimulus must be charge-balanced.")

    def _require_within_current_limit(self, stim):
        # Before the empty-data fast path: an empty dimensionless stimulus is
        # just as much the wrong kind of thing as a full one.
        self._require_current_stim(stim, 'max_current')
        # What the stimulator has to source at an instant is the sum over every
        # electrode active at that instant, whatever the sign of each:
        if stim.data.size == 0:
            return
        total = np.abs(stim.data).sum(axis=0)
        peak = total.max()
        if peak > self.max_current:
            worst = int(np.argmax(total))
            n_active = int(np.count_nonzero(stim.data[:, worst]))
            raise ValueError(
                f"Safety check: stimulus draws {peak:.1f} uA at once "
                f"({n_active} electrodes active), which exceeds "
                f"max_current={self.max_current:.1f} uA. Give the implant a "
                f"'raster' so that fewer electrodes fire at the same time, or "
                f"lower the amplitude.")

    def check_stim(self, stim):
        """Quality-check the stimulus

        This method is executed every time a new value is assigned to ``stim``.

        If ``safe_mode`` is set to True, this function will only allow stimuli
        that are charge-balanced. If ``max_current`` is set, it will only allow
        stimuli whose total instantaneous current stays within it.

        Both are questions about electricity, and neither can be answered about
        a stimulus that is not a current, so each raises a
        :py:class:`~pulse2percept.units.DimensionMismatchError` on one. In the
        ordinary flow this cannot happen: assigning to
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim` has already
        checked the stimulus against
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stimulus_unit`.
        ``check_stim`` is public, though, and may be handed anything.

        The user can define their own checks in implants that inherit from
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).

        Raises
        ------
        DimensionMismatchError
            If an electrical check was requested and ``stim`` is not measured
            in units of current.

        .. versionchanged:: 0.10.0
            The electrical checks verify that the stimulus really is
            electrical, instead of reading whatever numbers it holds as
            microamps.

        """
        if self.safe_mode:
            self._require_charge_balanced(stim)
        if self.max_current is not None:
            self._require_within_current_limit(stim)

    def preprocess_stim(self, stim):
        """Preprocess the stimulus

        This methods is executed every time a new value is assigned to ``stim``.

        No preprocessing is performed by default, but the user can define their
        own method in implants that inherit from
        return stim
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        A custom method must return a
        :py:class:`~pulse2percept.stimuli.Stimulus` object with the correct
        number of electrodes for the implant.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).

        Returns
        ----------
        stim_out : :py:class:`~pulse2percept.stimuli.Stimulus` object
        """
        return stim

    def reshape_stim(self, stim):
        if isinstance(stim, (ImageStimulus, VideoStimulus)):
            # Convert to grayscale:
            img = stim.rgb2gray()

            # Extract electrode coordinates, in the same units the image grid
            # below is laid out in:
            x, y = self.earray.coordinates(um)[:, :2].T

            # Define image coordinate space
            if isinstance(stim, ImageStimulus):
                img_h, img_w = img.img_shape
                data = img.data.reshape(img_h, img_w)  # Ensure 2D format
            elif isinstance(stim, VideoStimulus):
                img_h, img_w, n_frames = img.vid_shape
                data = img.data.reshape(img_h, img_w, n_frames)  # 3D format


            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

            # Create grid along original image axes
            img_x = np.linspace(x_min, x_max, img_w)
            img_y = np.linspace(y_min, y_max, img_h)

            # One interpolator covers every frame: the grid is the leading two
            # axes of `data`, and anything past them -- the frame axis of a
            # video -- is carried along, so a video comes back as
            # (n_electrodes, n_frames) from a single call. Building one per
            # frame instead meant re-deriving the same grid for each of them.
            interpolator = RegularGridInterpolator(
                (img_y, img_x), data, method='linear',
                bounds_error=False, fill_value=0
            )
            pixel_values = interpolator(np.vstack((y, x)).T)

            return Stimulus(
                pixel_values, electrodes=self.electrode_names,
                time=stim.time, metadata=stim.metadata)._inherit_units(stim)
        
        else:
            raise ValueError(
                f"Number of electrodes in the stimulus ({len(stim.electrodes)}) "
                f"does not match the number of electrodes in the implant ({self.n_electrodes})."
            )

    def plot(self, annotate=False, autoscale=True, ax=None, stim_cmap=False):
        """Plot

        Parameters
        ----------
        annotate : bool, optional
            Whether to scale the axes view to the data
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.
        stim_cmap : bool, str, or matplotlib colormap, optional
            If not false, the fill color of the plotted electrodes will vary based 
            on maximum stimulus amplitude on each electrode. The chosen colormap
            will be used if provided

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        stim = None
        if stim_cmap:
            if self.stim is None:
                raise ValueError("Must assign a stimulus in order to enable stimulus coloring")
            stim = self.stim
            if stim_cmap == True:
                stim_cmap = 'YlOrRd'
        return self.earray.plot(annotate=annotate, autoscale=autoscale, ax=ax, color_stim=stim, cmap=stim_cmap)

    def activate(self, electrodes):
        self.earray.activate(electrodes)

    def deactivate(self, electrodes):
        self.earray.deactivate(electrodes)
        # Switching an electrode off rewrites the stimulus, so it is replaced
        # rather than modified in place: it may be an object the caller still
        # holds, and one defined by more than its samples cannot lose an
        # electrode and remain one unless it says how (see
        # `Stimulus._without_electrodes`).
        if self.stim is not None:
            self._stim = self.stim._without_electrodes(electrodes)

    @property
    def earray(self):
        """Electrode array

        """
        return self._earray

    @earray.setter
    def earray(self, earray):
        """Electrode array setter (called upon ``self.earray = earray``)"""
        # Assign the electrode array:
        if isinstance(earray, Electrode):
            # For convenience, build an array from a single electrode:
            earray = ElectrodeArray(earray)
        if not isinstance(earray, ElectrodeArray):
            raise TypeError(f"'earray' must be an ElectrodeArray object, not "
                            f"{type(earray)}.")
        self._earray = earray

    @property
    def stim(self):
        """Stimulus

        A stimulus can be created from many source types, such as scalars,
        NumPy arrays, and dictionaries (see
        :py:class:`~pulse2percept.stimuli.Stimulus` for a complete list).

        A stimulus can be assigned either in the
        :py:class:`~pulse2percept.implants.ProsthesisSystem` constructor
        or later by assigning a value to `stim`.

        .. note::
           Unless when using dictionary notation, the number of stimuli must
           equal the number of electrodes in ``earray``.

        What is stored is always something the implant can deliver; for an
        electrical prosthesis that means a current (see
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stimulus_unit`).
        An image or a video is not that, so it is run through the implant's
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.encoder` on the way
        in. Without an encoder there is no principled default mapping from a
        gray level to an amplitude or a frequency, so assigning one raises a
        :py:class:`~pulse2percept.units.DimensionMismatchError`: say which you
        want with an
        :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` or a
        :py:class:`~pulse2percept.stimuli.FrequencyEncoder`.

        .. versionchanged:: 0.10.0
            A non-electrical stimulus is encoded on assignment if the implant
            has an encoder, and refused otherwise.

        Examples
        --------
        Send a biphasic pulse (30uA, 0.45ms phase duration) to an implant made
        from a single :py:class:`~pulse2percept.implants.DiskElectrode`:

        >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
        >>> from pulse2percept.stimuli import BiphasicPulse
        >>> implant = ProsthesisSystem(DiskElectrode(0, 0, 0, 100))
        >>> implant.stim = BiphasicPulse(30, 0.45)

        Stimulate Electrode B7 in Argus II with 13 uA:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(stim={'B7': 13})

        Argus II comes with an encoder, so an image can be assigned directly:

        >>> from pulse2percept.stimuli import LogoBVL
        >>> implant = ArgusII(stim=LogoBVL())
        >>> implant.stim.unit
        uA

        """
        return self._stim

    @stim.setter
    def stim(self, data):
        """Stimulus setter (called upon ``self.stim = data``)"""
        # if stim is empty or None
        if data is None:
            self._stim = None
        elif isinstance(data, (list, tuple, dict)) and not data:
            self._stim = None
        elif isinstance(data, np.ndarray) and data.size == 0:
            self._stim = None
        else:
            # Preprocess can be a function (callable) or True/False:
            if callable(self.preprocess):
                data = self.preprocess(data)
            elif self.preprocess:
                data = self.preprocess_stim(data)
            # Convert to stimulus object:
            if isinstance(data, Stimulus):
                # Already a stimulus object:
                stim = data
            elif isinstance(data, dict):
                # Electrode names already provided by keys:
                stim = Stimulus(data)
            else:
                # Use electrode names as stimulus coordinates:
                stim = Stimulus(data, electrodes=self.electrode_names)

            # A picture is not something an implant can deliver, so this is
            # where it becomes stimulation. Preprocessing goes first and may
            # have done the job already (a `preprocess` that encodes is exactly
            # as valid as an `encoder`), in which case there is nothing
            # dimensionless left to encode. What comes back knows both what
            # the device delivers and what it was asked for, so there is one
            # stimulus here and not two (see `Stimulus._spatial_view`).
            if (self.encoder is not None and
                    stim.unit.dimension.is_dimensionless and
                    stim.unit.dimension != self.stimulus_unit.dimension):
                stim = self.encoder.encode(stim, implant=self)

            # If the stim is larger than the number of electrodes, most commonly
            # we're dealing with an image or video stim. In this case, we might
            # want to try and reshape the stimulus to fit the array:
            if len(stim.electrodes) > self.n_electrodes:
                stim = self.reshape_stim(stim)

            # Whatever came in is now a Stimulus laid out on this implant's
            # electrodes. Whether it is a stimulus the implant can *deliver*
            # is the next question, and it is asked before anything else looks
            # at the numbers:
            self._require_deliverable_stim(stim)

            # Make sure all electrode names are valid:
            for electrode in stim.electrodes:
                # Invalid index will return None:
                if not self.earray[electrode]:
                    raise ValueError(f'Electrode "{electrode}" not found in '
                                     f'implant.')
            # Remove deactivated electrodes from the stimulus. Removal
            # rewrites the stimulus, so it happens on a copy: the caller's
            # object is theirs, and a stimulus defined by more than its
            # samples keeps that description only if it says how to drop an
            # electrode (see `Stimulus._without_electrodes`).
            off = [name for (name, e) in self.electrodes.items()
                   if not e.activated and name in stim.electrodes]
            if off:
                stim = stim._without_electrodes(off)
            # Copied before calibration rewrites it: the caller's object is
            # theirs.
            stim = self._calibrated(deepcopy(stim))
            # Perform safety checks, etc. These are all questions about what
            # gets delivered, so they are asked of the calibrated pulse train:
            self.check_stim(stim)
            # Store stimulus:
            self._stim = stim

    @property
    def eye(self):
        """Implanted eye

        A :py:class:`~pulse2percept.implants.ProsthesisSystem` can be implanted
        either in a left eye ('LE') or right eye ('RE'). Models such as
        :py:class:`~pulse2percept.models.AxonMapModel` will treat left and
        right eyes differently (for example, adjusting the location of the
        optic disc).

        Examples
        --------
        Implant Argus II in a left eye:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(eye='LE')
        """
        return self._eye

    @eye.setter
    def eye(self, eye):
        """Eye setter (called upon `self.eye = eye`)"""
        if not isinstance(eye, str):
            raise TypeError(f"'eye' must be a string, not {type(eye)}.")
        eye = eye.upper()
        if eye != 'LE' and eye != 'RE':
            raise ValueError(f"'eye' must be either 'LE' or 'RE', not "
                             f"{eye}.")
        self._eye = eye

    @property
    def n_electrodes(self):
        """Number of electrodes in the array

        This is equivalent to calling ``earray.n_electrodes``.
        """
        return self.earray.n_electrodes

    def __getitem__(self, item):
        return self.earray[item]

    def __iter__(self):
        return iter(self.earray)

    @property
    def electrodes(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in an ordered dictionary.
        You can iterate over different electrodes in the array as follows:

        .. code::

            for name, electrode in implant.electrodes.items():
                print(name, electrode)

        You can access an individual electrode by indexing directly into the
        prosthesis system object, e.g. ``implant['A1']`` or ``implant[0]``.

        """
        return self.earray.electrodes

    @property
    def electrode_names(self):
        """Return a list of all electrode names in the electrode array"""
        return self.earray.electrode_names

    @property
    def electrode_objects(self):
        """Return a list of all electrode objects in the array"""
        return self.earray.electrode_objects



class RectangleImplant(ProsthesisSystem):
    """ A generic rectangular implant

    Parameters
    ----------
    x, y, z : float, optional
        The x, y, z coordinates (um) of the center of the implant
    rot : float, optional
        The rotation of the implant in degrees
    shape : tuple, optional
        The number of rows and columns in the implant
    r : float, optional
        The electrode radius (um)
    spacing : float, optional
        The distance (um) between electrodes in the implant
    eye : str, optional
        The eye in which the implant is implanted
    stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
        A valid source type for a stimulus
    preprocess : bool, optional
        Whether to preprocess the stimulus
    safe_mode : bool, optional
        Whether to enforce charge balance

    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful
       quantities (e.g. ``spacing=0.4 * mm``). See
       :py:mod:`pulse2percept.units`.

    """
    def __init__(self, x=0, y=0, z=0, rot=0, shape=(15, 15), r=150./2, spacing=400., eye='RE', stim=None,
                 preprocess=True, safe_mode=False):
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.shape = shape
        names = ('A', '1')
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, r=r,
                                    rot=rot, names=names, etype=DiskElectrode)
        self.stim = stim
        
        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # TODO: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes
    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
