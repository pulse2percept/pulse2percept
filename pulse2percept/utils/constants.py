"""`DT`, `MIN_AMP`, `ZORDER`"""

#: Pulses with net currents smaller than 10 picoamps are considered
#: charge-balanced (here expressed in microamps).
MIN_AMP = 1e-5

#: Sampling time step (ms); defines the duration of the signal edge
#: transitions.
DT = 1e-3


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
