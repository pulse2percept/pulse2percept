.. _topics-implants:

=================
Visual Prostheses
=================

The :py:class:`~pulse2percept.implants.ProsthesisSystem` object defines a
common interface for all retinal prostheses ('bionic eyes'), consisting
of an :py:class:`~pulse2percept.implants.ElectrodeArray` and a
:py:class:`~pulse2percept.stimuli.Stimulus`.
Stimuli can be assigned to the various electrodes in the electrode array,
who will deliver them to the retina.
A mathematical model is then used to compute the neural stimulus response
and predict the resulting visual percept.

pulse2percept currently provides the following devices:

-  epiretinal (placed on top of the retinal surface):
   :py:class:`~pulse2percept.implants.ArgusI`,
   :py:class:`~pulse2percept.implants.ArgusII`
-  subretinal (placed next to the bipolar cells):
   :py:class:`~pulse2percept.implants.AlphaIMS`,
   :py:class:`~pulse2percept.implants.AlphaAMS`

Planned additions: PRIMA (Pixium Vision), BTV Bionic Eye (Bionic Vision
Technologies).

.. note::

    Users are free to create their own
    :py:class:`~pulse2percept.implants.Electrode`,
    :py:class:`~pulse2percept.implants.ElectrodeArray`, or
    :py:class:`~pulse2percept.implants.ProsthesisSystem`
    (see :ref:`Extending pulse2percept <dev-extending>`).

    




