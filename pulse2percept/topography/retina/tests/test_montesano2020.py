import copy
import subprocess
import sys
from importlib import resources

import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.topography import Grid2D
from pulse2percept.topography.retina import (Curcio1990Map, Montesano2020Map,
                                             Watson2014Map)

# ---------------------------------------------------------------------------
# Independent reference values.
#
# Taken from the directly solved reconstruction (7200 meridians on a 1 um
# radial grid) that the packaged field was resampled from, not from the
# packaged field itself, so these check the runtime interpolation rather than
# restating it. Anatomical angle: 0 nasal, 90 superior, 180 temporal,
# 270 inferior.
# ---------------------------------------------------------------------------

#: r_rf (dva) the forward reference is tabulated at: inside the displacement
#: zone, at the global peak, in its outer half, and past the nasal/inferior
#: zone extents where the map is the identity.
_FORWARD_RADII = (0.5, 1.0, 2.44, 5.0, 8.0, 12.0)

#: r_soma = F(theta, r_rf) in dva
_FORWARD_REF = {
    0.0: (1.85455, 2.66891, 4.26097, 6.18923, 8.30252, 12.00000),
    45.0: (1.82899, 2.57250, 4.04538, 6.12473, 8.69370, 12.33515),
    90.0: (1.88677, 2.61532, 4.16064, 6.38372, 9.05385, 12.42990),
    135.0: (1.89807, 2.65607, 4.18955, 6.34609, 8.94263, 12.32390),
    180.0: (1.94801, 2.80622, 4.52330, 6.74709, 9.13521, 12.33158),
    225.0: (1.78347, 2.55407, 4.08092, 6.23245, 8.77965, 12.22550),
    270.0: (1.71475, 2.44061, 3.89550, 5.84349, 8.19664, 12.00000),
    315.0: (1.75812, 2.52205, 4.02135, 5.97731, 8.22723, 12.00000),
}

#: Retinal radii (um) that [Watson2014]_ Eq. A6 maps to exactly 1, 2, 4, 6, 10
#: and 13 dva. Stated in microns so that the inverse reference below tests the
#: displacement inverse alone: feeding these to ``ret_to_dva`` reaches soma
#: space exactly, without Watson's Eq. A5/A6 mismatch entering.
_INVERSE_UM = (279.939003, 557.541230, 1106.889150, 1650.067588,
               2724.219733, 3523.929907)
_INVERSE_SOMA = (1.0, 2.0, 4.0, 6.0, 10.0, 13.0)

#: r_rf = G(theta, r_soma) in dva
_INVERSE_REF = {
    0.0: (0.13016, 0.57936, 2.15885, 4.72108, 10.00000, 13.00000),
    45.0: (0.13213, 0.60150, 2.38824, 4.84332, 9.39810, 12.77373),
    90.0: (0.08858, 0.56926, 2.26927, 4.55072, 9.06758, 12.71335),
    135.0: (0.10906, 0.55884, 2.23536, 4.57688, 9.20600, 12.81633),
    180.0: (0.12044, 0.52627, 1.94121, 4.08138, 9.09206, 12.80795),
    225.0: (0.15087, 0.62552, 2.35113, 4.71666, 9.40453, 12.88356),
    270.0: (0.15652, 0.68000, 2.56426, 5.21508, 9.96755, 13.00000),
    315.0: (0.15467, 0.64280, 2.41545, 5.03191, 10.00000, 13.00000),
}

#: Displacement-zone extent (dva) per cardinal meridian: the largest r_rf that
#: is still displaced.
_ZONE_EXTENT = {0.0: 9.5426, 90.0: 14.0971, 180.0: 14.0971, 270.0: 10.5238}

#: Published global maximum displacement (dva), its meridian and its r_rf.
_MAX_DISPLACEMENT = (2.0843, 182.75, 2.439)


def _visual_deg(theta_ret_deg, eye):
    """Visual-field polar angle (deg) that falls on an anatomical meridian"""
    return theta_ret_deg - 180.0 if eye == 'left' else -theta_ret_deg


def _dva_point(theta_ret_deg, radius_dva, eye):
    """Visual-field x,y (dva) at a radius along an anatomical meridian"""
    theta = np.deg2rad(_visual_deg(theta_ret_deg, eye))
    return radius_dva * np.cos(theta), radius_dva * np.sin(theta)


def _ret_point(theta_ret_deg, radius_um, eye):
    """Retinal x,y (um) whose visual direction is that anatomical meridian

    Eq. A6 flips the y axis, so the retinal polar angle is the negated visual
    one.
    """
    theta = np.deg2rad(-_visual_deg(theta_ret_deg, eye))
    return radius_um * np.cos(theta), radius_um * np.sin(theta)


#: The field is stored as float32, so "undisplaced" holds to about 1e-6 dva,
#: which is 3e-4 um. Displaced points are tens of microns out.
_UNDISPLACED_UM = 1e-2


def _displacement_um(vfmap, theta_ret_deg, radius_dva):
    """How much farther out than Watson2014Map the map places a point (um)"""
    xdva, ydva = _dva_point(theta_ret_deg, radius_dva, vfmap.eye)
    displaced = np.hypot(*vfmap.dva_to_ret(xdva, ydva))
    plain = np.hypot(*Watson2014Map().dva_to_ret(xdva, ydva))
    return displaced - plain


def _radius_um(vfmap, theta_ret_deg, radius_dva):
    """Retinal radius (um) a point on an anatomical meridian maps to"""
    return np.hypot(*vfmap.dva_to_ret(
        *_dva_point(theta_ret_deg, radius_dva, vfmap.eye)))


def test_Montesano2020Map_eye():
    npt.assert_equal(Montesano2020Map().eye, 'right')
    npt.assert_equal(Montesano2020Map(eye='LEFT').eye, 'left')
    npt.assert_equal(Montesano2020Map(eye='Right').eye, 'right')
    with pytest.raises(TypeError):
        Montesano2020Map(eye=0)
    with pytest.raises(ValueError):
        Montesano2020Map(eye='both')
    # `eye` is a regular parameter:
    npt.assert_equal('right' in repr(Montesano2020Map()), True)
    npt.assert_equal(Montesano2020Map() == Montesano2020Map(), True)
    npt.assert_equal(Montesano2020Map() == Montesano2020Map(eye='left'),
                     False)
    npt.assert_equal(Montesano2020Map() == copy.deepcopy(Montesano2020Map()),
                     True)
    # Equality is exact-class, so a plain Watson map is not a Montesano one:
    npt.assert_equal(Montesano2020Map() == Watson2014Map(), False)
    npt.assert_equal(Montesano2020Map() == Curcio1990Map(), False)


def test_Montesano2020Map_origin():
    trafo = Montesano2020Map()
    npt.assert_array_equal(trafo.dva_to_ret(0, 0), (0, 0))
    npt.assert_array_equal(trafo.ret_to_dva(0, 0), (0, 0))


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_forward_reference(eye):
    """dva_to_ret reproduces the directly solved forward field"""
    trafo = Montesano2020Map(eye=eye)
    watson = Watson2014Map()
    for theta_ret, expected in _FORWARD_REF.items():
        for r_rf, r_soma in zip(_FORWARD_RADII, expected):
            got = trafo.dva_to_ret(*_dva_point(theta_ret, r_rf, eye))
            # Both sides take the identical Watson2014Map step, so this
            # compares the displacement alone. Agreement is 0.003 um; the
            # packaged field's linear interpolation is worth up to 0.003 dva
            # (~0.8 um) elsewhere in the field.
            npt.assert_allclose(got,
                                watson.dva_to_ret(*_dva_point(theta_ret,
                                                              r_soma, eye)),
                                atol=0.02,
                                err_msg=f'{theta_ret} deg, r_rf {r_rf} dva')


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_inverse_reference(eye):
    """ret_to_dva reproduces the inverse of the directly solved field"""
    trafo = Montesano2020Map(eye=eye)
    for theta_ret, expected in _INVERSE_REF.items():
        for r_um, r_soma, r_rf in zip(_INVERSE_UM, _INVERSE_SOMA, expected):
            xret, yret = _ret_point(theta_ret, r_um, eye)
            # The hard-coded microns land on a whole number of soma dva:
            npt.assert_allclose(np.hypot(*Watson2014Map().ret_to_dva(xret,
                                                                     yret)),
                                r_soma, atol=1e-6)
            got = np.hypot(*trafo.ret_to_dva(xret, yret))
            # Measured agreement 2.6e-5 dva across every case here.
            npt.assert_allclose(got, r_rf, atol=1e-4,
                                err_msg=f'{theta_ret} deg, {r_soma} dva')


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_horizontal_orientation(eye):
    """Which side of the visual field is nasal retina depends on the eye

    The nasal displacement zone ends at 9.54 dva and the temporal one at
    14.10, so a point 12 dva out is displaced on one horizontal side and not
    on the other. Which side that is flips between the eyes.
    """
    trafo = Montesano2020Map(eye=eye)
    nasal_x = -1.0 if eye == 'left' else 1.0
    # Anatomy first: nasal is undisplaced at 12 dva, temporal is not.
    npt.assert_allclose(_displacement_um(trafo, 0.0, 12.0), 0.0,
                        atol=_UNDISPLACED_UM)
    npt.assert_equal(_displacement_um(trafo, 180.0, 12.0) > 50.0, True)
    # And that anatomy sits on the expected side of the visual field:
    for x_sign, theta_ret in [(nasal_x, 0.0), (-nasal_x, 180.0)]:
        ref = Watson2014Map().dva_to_ret(
            *_dva_point(theta_ret, _FORWARD_REF[theta_ret][2], eye))
        npt.assert_allclose(trafo.dva_to_ret(x_sign * 2.44, 0.0), ref,
                            atol=0.02)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_vertical_orientation(eye):
    """Superior/inferior anatomy is the same in both eyes

    Positive y in the visual field is inferior retina, whose displacement zone
    ends at 10.52 dva; the superior zone reaches 14.10.
    """
    trafo = Montesano2020Map(eye=eye)
    npt.assert_allclose(_displacement_um(trafo, 270.0, 12.0), 0.0,
                        atol=_UNDISPLACED_UM)
    npt.assert_equal(_displacement_um(trafo, 90.0, 12.0) > 50.0, True)
    for y_sign, theta_ret in [(1.0, 270.0), (-1.0, 90.0)]:
        ref = Watson2014Map().dva_to_ret(
            0.0, y_sign * _FORWARD_REF[theta_ret][2])
        npt.assert_allclose(trafo.dva_to_ret(0.0, y_sign * 2.44), ref,
                            atol=0.02)
    # Both eyes agree on the vertical meridian:
    other = Montesano2020Map(eye='right' if eye == 'left' else 'left')
    y = np.array([-12.0, -6.0, -1.5, 0.0, 1.5, 6.0, 12.0])
    npt.assert_allclose(trafo.dva_to_ret(np.zeros_like(y), y),
                        other.dva_to_ret(np.zeros_like(y), y), rtol=1e-12)


def test_Montesano2020Map_mirror():
    """A left eye is the horizontal mirror of a right eye"""
    right = Montesano2020Map(eye='right')
    left = Montesano2020Map(eye='left')
    x = np.array([-13.0, -8.0, -2.5, -0.5, 0.5, 2.5, 8.0, 13.0])
    y = np.array([-11.0, -6.0, 3.0, -1.5, 0.0, 4.5, -2.0, 9.0])
    x_right, y_right = right.dva_to_ret(x, y)
    x_left, y_left = left.dva_to_ret(-x, y)
    npt.assert_allclose(x_left, -x_right, rtol=1e-12, atol=1e-9)
    npt.assert_allclose(y_left, y_right, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_displacement_is_radial(eye):
    """Only the radius changes; the visual-field direction does not"""
    trafo = Montesano2020Map(eye=eye)
    theta = np.deg2rad(np.arange(0.0, 360.0, 7.0))
    for radius in (0.3, 2.44, 7.0, 12.0, 20.0):
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        xret, yret = trafo.dva_to_ret(x, y)
        # Watson2014Map is itself radial, so the retinal direction of the
        # displaced point must match the undisplaced one:
        xref, yref = Watson2014Map().dva_to_ret(x, y)
        npt.assert_allclose(np.arctan2(yret, xret), np.arctan2(yref, xref),
                            atol=1e-12)
        # Outward only, and monotonically increasing in radius:
        npt.assert_array_less(-1e-9, np.hypot(xret, yret) -
                              np.hypot(xref, yref))


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_identity_beyond_the_support(eye):
    """At and beyond 15 dva the map is exactly Watson2014Map"""
    trafo = Montesano2020Map(eye=eye)
    theta = np.deg2rad(np.arange(0.0, 360.0, 3.0))
    for radius in (15.0, 15.5, 20.0, 45.0):
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        npt.assert_allclose(trafo.dva_to_ret(x, y),
                            Watson2014Map().dva_to_ret(x, y), rtol=1e-12)
        xret, yret = Watson2014Map().dva_to_ret(x, y)
        npt.assert_allclose(trafo.ret_to_dva(xret, yret),
                            Watson2014Map().ret_to_dva(xret, yret),
                            rtol=1e-12)


def test_Montesano2020Map_zone_boundary():
    """Displacement dies out at the per-meridian zone extent

    The extent varies from 9.54 dva (nasal) to 14.10 (superior/temporal), and
    the 751-node radial grid resolves the boundary to about 0.05 dva.
    """
    trafo = Montesano2020Map()
    for theta_ret, extent in _ZONE_EXTENT.items():
        npt.assert_equal(_displacement_um(trafo, theta_ret, extent - 0.5) > 0,
                         True, err_msg=f'{theta_ret} deg')
        npt.assert_allclose(_displacement_um(trafo, theta_ret, extent + 0.1),
                            0.0, atol=_UNDISPLACED_UM,
                            err_msg=f'{theta_ret} deg')
    # The extents genuinely differ, so a single meridian would not do:
    npt.assert_allclose(_displacement_um(trafo, 0.0, 12.0), 0.0,
                        atol=_UNDISPLACED_UM)
    npt.assert_equal(_displacement_um(trafo, 90.0, 12.0) > 50.0, True)


def test_Montesano2020Map_periodic_seam():
    """0 and 360 deg are the same meridian, with no seam between them"""
    trafo = Montesano2020Map()
    # 0 and 360 deg name one meridian, and 359.9 deg is 0.1 deg short of it:
    npt.assert_allclose(trafo.dva_to_ret(*_dva_point(360.0, 2.44, 'right')),
                        trafo.dva_to_ret(*_dva_point(0.0, 2.44, 'right')),
                        rtol=1e-12, atol=1e-9)
    npt.assert_allclose(trafo.dva_to_ret(*_dva_point(359.9, 2.44, 'right')),
                        trafo.dva_to_ret(*_dva_point(-0.1, 2.44, 'right')),
                        rtol=1e-12, atol=1e-9)
    for radius in (0.5, 2.44, 8.0):
        # A query between the last stored meridian (359.75 deg) and the first
        # (0 deg) lands between the two, rather than clamping to either:
        lo = _radius_um(trafo, 359.75, radius)
        hi = _radius_um(trafo, 0.0, radius)
        mid = _radius_um(trafo, 359.9, radius)
        npt.assert_allclose((mid - lo) / (hi - lo), 0.15 / 0.25, atol=1e-3)
        # And the field is as smooth across the seam as it is anywhere:
        sweep = np.array([_radius_um(trafo, angle % 360.0, radius)
                          for angle in np.arange(-1.0, 1.0001, 0.05)])
        steps = np.abs(np.diff(sweep))
        npt.assert_array_less(steps.max(), 2.0 * np.median(steps))


def test_Montesano2020Map_shapes():
    """Scalars stay scalar, arrays keep their shape, and inputs broadcast"""
    trafo = Montesano2020Map()
    for x, y in trafo.dva_to_ret(2.0, 2.0), trafo.ret_to_dva(560.0, -560.0):
        npt.assert_equal(np.ndim(x), 0)
        npt.assert_equal(np.ndim(y), 0)
        npt.assert_equal(np.isfinite(x) and np.isfinite(y), True)

    vec = np.array([-8.0, -2.0, 0.0, 2.0, 8.0, 18.0])
    xret, yret = trafo.dva_to_ret(vec, vec[::-1])
    npt.assert_equal(xret.shape, vec.shape)
    npt.assert_equal(np.all(np.isfinite(xret)) and np.all(np.isfinite(yret)),
                     True)
    npt.assert_allclose(trafo.ret_to_dva(xret, yret)[0].shape, vec.shape)

    grid_x, grid_y = np.meshgrid(vec, vec)
    xret, yret = trafo.dva_to_ret(grid_x, grid_y)
    npt.assert_equal(xret.shape, grid_x.shape)
    npt.assert_equal(np.all(np.isfinite(xret)), True)
    # A 2-D pass agrees elementwise with the 1-D one:
    npt.assert_allclose(xret[0], trafo.dva_to_ret(vec, grid_y[0])[0],
                        rtol=1e-12)

    # Broadcasting a column against a row:
    xb, yb = trafo.dva_to_ret(vec[:, np.newaxis], vec[np.newaxis, :])
    npt.assert_equal(xb.shape, (vec.size, vec.size))
    npt.assert_allclose(xb, xret.T, rtol=1e-12)
    npt.assert_allclose(yb, yret.T, rtol=1e-12)
    # Lists are arrays too:
    npt.assert_allclose(trafo.dva_to_ret([1.0, 2.0], [3.0, 4.0]),
                        trafo.dva_to_ret(np.array([1.0, 2.0]),
                                         np.array([3.0, 4.0])), rtol=1e-12)


def test_Montesano2020Map_preserves_precision():
    """A model grid is float32, and the spatial kernels require it back"""
    trafo = Montesano2020Map()
    for dtype in (np.float32, np.float64):
        x = np.array([0.5, 2.44, 12.0, 20.0], dtype=dtype)
        xret, yret = trafo.dva_to_ret(x, x)
        npt.assert_equal(xret.dtype, dtype)
        npt.assert_equal(yret.dtype, dtype)
        xdva, ydva = trafo.ret_to_dva(xret, yret)
        npt.assert_equal(xdva.dtype, dtype)
        npt.assert_equal(ydva.dtype, dtype)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_Grid2D_build(eye):
    grid = Grid2D((-14, 14), (-14, 14), step=1)
    grid.build(Montesano2020Map(eye=eye))
    npt.assert_equal(grid.ret.x.shape, grid.x.shape)
    npt.assert_equal(np.all(np.isfinite(grid.ret.x)), True)
    npt.assert_equal(np.all(np.isfinite(grid.ret.y)), True)
    # The grid is displaced outward relative to a plain Watson grid:
    plain = Grid2D((-14, 14), (-14, 14), step=1)
    plain.build(Watson2014Map())
    outward = np.hypot(plain.ret.x, plain.ret.y) - _UNDISPLACED_UM
    npt.assert_array_less(outward, np.hypot(grid.ret.x, grid.ret.y))


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_round_trip_dva(eye):
    """dva -> ret -> dva, whose residual is Watson2014Map's own

    Watson's Eqs. A5 and A6 are fitted separately and only agree to a few
    percent, so a Watson round trip does not close either. The displacement
    inverse adds at most 0.057 dva on top of that (measured; its own
    interpolation error is below 0.0005 dva, and the Jacobian of the
    displacement amplifies Watson's residual inside the fovea).
    """
    trafo = Montesano2020Map(eye=eye)
    watson = Watson2014Map()
    theta = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    radius = np.array([0.05, 0.5, 2.44, 5.0, 9.0, 12.0, 14.0, 15.0, 25.0,
                       40.0])
    x = np.outer(radius, np.cos(theta))
    y = np.outer(radius, np.sin(theta))
    err = np.hypot(*[back - fwd for back, fwd
                     in zip(trafo.ret_to_dva(*trafo.dva_to_ret(x, y)),
                            (x, y))])
    watson_err = np.hypot(*[back - fwd for back, fwd
                            in zip(watson.ret_to_dva(*watson.dva_to_ret(x, y)),
                                   (x, y))])
    npt.assert_array_less(err, watson_err + 0.06)
    npt.assert_array_less(err, 0.24)
    # Beyond the displacement zone the two round trips are the same thing:
    outside = radius >= 15.0
    npt.assert_allclose(err[outside], watson_err[outside], rtol=1e-9)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Montesano2020Map_round_trip_ret(eye):
    """ret -> dva -> ret closes as tightly as Watson2014Map alone does"""
    trafo = Montesano2020Map(eye=eye)
    watson = Watson2014Map()
    theta = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    radius = np.array([50.0, 280.0, 560.0, 1100.0, 2700.0, 3520.0, 4100.0,
                       6000.0])
    x = np.outer(radius, np.cos(theta))
    y = np.outer(radius, np.sin(theta))
    err = np.hypot(*[back - fwd for back, fwd
                     in zip(trafo.dva_to_ret(*trafo.ret_to_dva(x, y)),
                            (x, y))])
    watson_err = np.hypot(*[back - fwd for back, fwd
                            in zip(watson.dva_to_ret(*watson.ret_to_dva(x, y)),
                                   (x, y))])
    # Measured excess over Watson's own residual: 0.07 um.
    npt.assert_array_less(err, watson_err + 0.5)


def test_montesano2020_field_invariants():
    """Scientific invariants of the packaged field

    The full audit (E2v fits, zone extents, figure comparisons) lives in
    ``tools/generate_montesano2020_map.py``; these are the properties the
    runtime interpolators rely on.
    """
    path = resources.files('pulse2percept.topography.retina').joinpath(
        'data', 'montesano2020.npz')
    with path.open('rb') as f:
        with np.load(f, allow_pickle=False) as npz:
            angle = np.asarray(npz['retinal_angle_deg'], dtype=np.float64)
            r_rf = np.asarray(npz['rf_eccentricity_dva'], dtype=np.float64)
            table = np.asarray(npz['soma_eccentricity_dva'],
                               dtype=np.float64)
            npt.assert_equal(str(npz['reference_eye']), 'right')

    npt.assert_equal(table.shape, (angle.size, r_rf.size))
    npt.assert_equal(np.all(np.isfinite(table)), True)
    npt.assert_allclose(angle, np.arange(0.0, 360.0, 0.25), atol=1e-4)
    npt.assert_allclose(r_rf, np.linspace(0.0, 15.0 ** (1 / 3), 751) ** 3,
                        atol=1e-12)
    # F(theta, 0) = 0 and F(theta, 15) = 15 on every meridian:
    npt.assert_array_equal(table[:, 0], np.zeros(angle.size))
    npt.assert_allclose(table[:, -1], r_rf[-1], atol=1e-5)
    # Strictly increasing in r_rf, so each meridian is invertible:
    npt.assert_equal(np.diff(table, axis=1).min() > 0.0, True)
    # Displacement is centrifugal everywhere (float32 storage puts the
    # identity part of the field within ~1e-6 dva of zero, not exactly on it):
    displacement = table - r_rf[np.newaxis, :]
    npt.assert_array_less(-2e-6, displacement)
    # The published global maximum:
    i, j = np.unravel_index(displacement.argmax(), displacement.shape)
    peak, peak_angle, peak_r = _MAX_DISPLACEMENT
    npt.assert_allclose(displacement[i, j], peak, atol=5e-3)
    npt.assert_allclose(angle[i], peak_angle, atol=1.0)
    npt.assert_allclose(r_rf[j], peak_r, atol=0.05)


def test_montesano2020_field_loads_lazily():
    """The 3.7 MB field is read on first use, not at import

    Run in a subprocess so the check does not depend on what earlier tests
    have already loaded.
    """
    script = """
import numpy as np
real_load, seen = np.load, []
def spy(file, *args, **kwargs):
    seen.append(str(file))
    return real_load(file, *args, **kwargs)
np.load = spy

def read_yet():
    return [f for f in seen if 'montesano' in f]

import pulse2percept.topography.retina as retina
assert not read_yet(), 'read at import: %s' % read_yet()
vfmap = retina.Montesano2020Map()
assert not read_yet(), 'read on construction: %s' % read_yet()
vfmap.dva_to_ret(2.0, 2.0)
assert len(read_yet()) == 1, 'read %d times' % len(read_yet())
vfmap.ret_to_dva(560.0, 560.0)
retina.Montesano2020Map(eye='left').dva_to_ret(2.0, 2.0)
assert len(read_yet()) == 1, 'cached field re-read %d times' % len(read_yet())
"""
    done = subprocess.run([sys.executable, '-c', script],
                          capture_output=True, text=True)
    npt.assert_equal(done.returncode, 0, err_msg=done.stderr)
