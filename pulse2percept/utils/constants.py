""":py:class:`~pulse2percept.utils.DT`,
   :py:class:`~pulse2percept.utils.MIN_AMP`,
   :py:class:`~pulse2percept.utils.MS_PER_S`,
   :py:class:`~pulse2percept.utils.UM_PER_MM`,
   :py:class:`~pulse2percept.utils.VIDEO_BLOCK_SIZE`,
   :py:class:`~pulse2percept.utils.ZORDER`"""
from ..units import Quantity, mm, ms, s, um

#: Pulses with net currents smaller than 10 picoamps are considered
#: charge-balanced (here expressed in microamps).
MIN_AMP = 1e-5

#: Sampling time step (ms); defines the duration of the signal edge
#: transitions.
DT = 1e-3

#: Milliseconds in a second (1000).
#:
#: p2p counts durations in milliseconds and frequencies in hertz, so anything
#: that turns one into the other needs this factor: a pulse train's window is
#: ``MS_PER_S / freq`` ms, and a duration in ms is ``dur / MS_PER_S`` seconds.
#: Derived from the unit system once, at import time, rather than written down
#: as a bare 1000 at each site -- those are the conversions that go wrong
#: silently. Numerical code divides by it and stays plain floats; nothing here
#: puts a :py:class:`~pulse2percept.units.Quantity` inside a loop.
#:
#: .. versionadded:: 0.10.0
MS_PER_S = Quantity(1, s).to_value(ms)

#: Microns in a millimeter (1000).
#:
#: p2p stores tissue coordinates in microns, while published cortical and
#: retinal fits are written in millimeters and plots are labelled in them.
#: Same idea as :py:data:`MS_PER_S`: derive the factor from the unit system
#: once, then do plain arithmetic with it.
#:
#: .. versionadded:: 0.10.0
UM_PER_MM = Quantity(1, mm).to_value(um)

# Block size for saving videos: width/height must be divisible by 16 for most
# codecs to work
VIDEO_BLOCK_SIZE = 16

#: An enum specifying the zorder values to use in Matplotlib plots, ensuring
#: that foreground items (like implants) always appear on top of background
#: items (like axon maps).
ZORDER = {
    'front': 9999,
    'back': 0,
    'background': 1,
    'foreground': 50,
    'annotate': 2000
}
