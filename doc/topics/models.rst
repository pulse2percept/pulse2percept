.. _topics-models:

====================
Computational Models
====================

pulse2percept provides the following computational models:

From [Nanduri2012]_:

*  :py:class:`~pulse2percept.models.Nanduri2012Model`, consisting of two parts:

   *   :py:class:`~pulse2percept.models.Nanduri2012SpatialMixin`:
       spatial attenuation function of current spread as described in
       [Ahuja2008]_
   *   :py:class:`~pulse2percept.models.Nanduri2012TemporalMixin`:
       linear-nonlinear cascade model of temporal sensitivity

From [Beyeler2019]_:

*  :py:class:`~pulse2percept.models.ScoreboardModel`:
   spatial phosphene model assuming all phosphenes are focal dots of light
*  :py:class:`~pulse2percept.models.AxonMapModel`:
   spatial phosphene model assuming phosphene shape is determined by the
   spatial arrangement of nerve fiber bundles in the optic fiber layer
