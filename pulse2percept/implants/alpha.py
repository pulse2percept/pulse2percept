""":py:class:`~pulse2percept.implants.AlphaIMS`, 
   :py:class:`~pulse2percept.implants.AlphaAMS`"""
import numpy as np
from collections import OrderedDict

from .base import Implant
from .electrodes import SquareElectrode, DiskElectrode
from .electrode_arrays import ElectrodeGrid
from ..units import as_value, um


class AlphaIMS(Implant):
    """Alpha-IMS

    This class creates an Alpha-IMS array with 1500 photovoltaic pixels (each
    50um in diameter) as described in [Stingl2013]_. Electrode coordinates
    are given in the array's own frame, centered on ``(0, 0)``. Where the
    array is implanted in the subretinal space is set by the model's
    ``implant_position``, ``implant_rotation`` and ``implant_depth``.

    The device consists of 1500 50um-wide square pixels, arranged on a 39x39
    rectangular grid with 72um pixel pitch.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field).

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    z : float, list, or Quantity, optional
        Electrode height (um) above the array's own plane: a scalar
        applies to every electrode, a list of 1500 entries gives each its own.
        Electrode-retina distance is the model's ``implant_depth``.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Examples
    --------
    Create an Alpha-IMS array:

    >>> from pulse2percept.implants import AlphaIMS
    >>> AlphaIMS()  # doctest: +NORMALIZE_WHITESPACE
    AlphaIMS(electrode_array=ElectrodeGrid, eye='RE', preprocess=True,
             safe_mode=False, shape=(39, 39))

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaIMS()
    >>> alpha_ims['A3']  # doctest: +NORMALIZE_WHITESPACE
    SquareElectrode(activated=True, name='A3', side_length=50.0,
                    x=-1224.0, y=-1368.0, z=0.0)
    >>> alpha_ims[0, 2]  # doctest: +NORMALIZE_WHITESPACE
    SquareElectrode(activated=True, name='A3', side_length=50.0,
                    x=-1224.0, y=-1368.0, z=0.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    placement = 'subretinal'

    def __init__(self, z=0, eye='RE', preprocess=True,
                 safe_mode=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (39, 39)
        elec_width = 50.0  # um
        e_spacing = 72.0  # um

        # Normalized here rather than in ElectrodeGrid, because a per-electrode
        # list of heights never reaches the grid at all -- it is written onto
        # the electrodes further down:
        z = as_value(z, um, 'z')

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = 0.0 if overwrite_z else z
        self.electrode_array = ElectrodeGrid(
            self.shape, e_spacing, z=zarr,
            electrode_type=SquareElectrode, side_length=elec_width)

        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.electrode_array.electrode_names
            objects = self.electrode_array.electrode_objects
            names = np.array(names).reshape(self.electrode_array.shape)
            # Reverse column names:
            for row in range(self.electrode_array.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to electrode_array:
            self.electrode_array._electrodes = electrodes

        # Remove electrodes:
        extra_elecs = ['AM39', 'AL39', 'AK39', 'AJ39', 'AI39', 'AH39', 'AG39',
                       'AF39', 'AE39', 'AD39', 'AC39',
                       'AM38', 'AL38', 'AK38', 'AJ38', 'AI38', 'AH38', 'AG38',
                       'AF38', 'AE38', 'AD38']
        for elec in extra_elecs:
            self.electrode_array.remove_electrode(elec)

        # Now that the superfluous electrodes have been deleted, adjust the
        # z values:
        if overwrite_z:
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have {self.n_electrodes} entries, "
                                 f"not {z_arr.size}.")
            for elec, z_elec in zip(self.electrode_array.electrode_objects, z):
                elec.z = z_elec

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape})
        return params


class AlphaAMS(Implant):
    """Alpha-AMS

    This class creates an Alpha-AMS array with 1600 photovoltaic pixels (each
    30um in diameter) as described in [Stingl2017]_. Electrode coordinates
    are given in the array's own frame, centered on ``(0, 0)``. Where the
    array is implanted in the subretinal space is set by the model's
    ``implant_position``, ``implant_rotation`` and ``implant_depth``.

    The device consists of 1600 30um-wide round pixels, arranged on a 40x40
    rectangular grid with 70um pixel pitch.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field), as shown below:

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    z : float, list, or Quantity, optional
        Electrode height (um) above the array's own plane: a scalar
        applies to every electrode, a list of 1600 entries gives each its own.
        Electrode-retina distance is the model's ``implant_depth``.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Examples
    --------
    Create an AlphaAMS array:

    >>> from pulse2percept.implants import AlphaAMS
    >>> AlphaAMS()  # doctest: +NORMALIZE_WHITESPACE
    AlphaAMS(electrode_array=ElectrodeGrid, eye='RE', preprocess=True,
             safe_mode=False, shape=(40, 40))

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaAMS()
    >>> alpha_ims['A3']  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='A3', radius=15.0,
                  x=-1225.0, y=-1365.0, z=0.0)
    >>> alpha_ims[0, 2]  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='A3', radius=15.0,
                  x=-1225.0, y=-1365.0, z=0.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    placement = 'subretinal'

    def __init__(self, z=0, eye='RE', preprocess=True,
                 safe_mode=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (40, 40)
        elec_radius = 15.0
        e_spacing = 70.0  # um

        self.electrode_array = ElectrodeGrid(
            self.shape, e_spacing, z=z,
            electrode_type=DiskElectrode, radius=elec_radius)

        # Set left/right eye:
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.electrode_array.electrode_names
            objects = self.electrode_array.electrode_objects
            names = np.array(names).reshape(self.electrode_array.shape)
            # Reverse column names:
            for row in range(self.electrode_array.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to electrode_array:
            self.electrode_array._electrodes = electrodes

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape})
        return params
