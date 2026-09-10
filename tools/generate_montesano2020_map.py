#!/usr/bin/env python3
"""Generate the packaged Montesano et al. (2020) RGC displacement field.

Reconstructs the retinal ganglion cell (RGC) displacement model of

    Montesano G, Ometto G, Hogg RE, Rossetti LM, Garway-Heath DF, Crabb DP
    (2020). "Revisiting the Drasdo Model: Implications for Structure-Function
    Analysis of the Macular Region." Transl Vis Sci Technol 9(10):15.
    doi:10.1167/tvst.9.10.15

from its published equations plus the Curcio & Allen (1990) ganglion cell
topography map, and writes the resulting field to a compressed .npz for
``pulse2percept.topography.retina``.

Usage::

    python tools/generate_montesano2020_map.py path/to/SORTED.txt

Source histology data (not vendored)
------------------------------------
``SORTED.txt`` from https://christineacurcio.com/GanglionCellTopography/,
SHA256 ``b29a3eb50fb1bd56742b86e87a9f585488baf94ec8646993b762a91149734c2e``
(15460 bytes). It holds 171 scattered (co-latitude, longitude, density)
samples of RGC soma density on a left-eye retinal sphere of radius 11.459 mm,
which is the map Montesano's Appendix A describes. Curcio's site states no
data-use license, so the file is not redistributed here; only the derived
displacement field is packaged. The hash is enforced below: any other input
would change the artifact without re-auditing the reconstruction.

The authors' own implementation (a Shiny app at relayer.online/drasdo) is no
longer online and was not archived, so this is an independent reconstruction
validated against the published figures, not a port of author code.

Coordinate conventions
----------------------
``SORTED.txt`` is a left eye with longitude 0 deg temporal, 90 deg superior,
180 deg nasal, 270 deg inferior. The artifact instead stores right-eye
*retinal anatomical* meridians::

    0 deg    nasal
    90 deg   superior
    180 deg  temporal
    270 deg  inferior

Montesano's right eye is a horizontal mirror of the left-eye source map, so the
anatomical-meridian profile is the same in both eyes; laterality enters only in
the visual-field-to-anatomical-angle mapping, which is a runtime concern and is
deliberately not baked into the artifact.

Eccentricities are visual degrees (dva) throughout, converted from retinal mm
with the schematic eye below.

Resolved ambiguities in the published description
-------------------------------------------------
1. Table A1 gives "eccentricity of corneal ellipse = 0.500" without saying how
   it enters the surface equation. Reading it as the conic constant (K = -0.5)
   reproduces the published first nodal point to 0.5 um (6.9295 mm vs
   6.930 mm); reading it as an ellipse eccentricity (K = -e**2 = -0.25) gives
   6.8501 mm. K = -0.5 is used. The choice is immaterial downstream: retinal
   distances differ by <= 8 um out to 15 deg.
2. The piecewise-cubic reconstruction of the histology undershoots to about
   -61 cells/mm2 near 0.1 mm on the superior meridian. Density is clipped at
   zero as a physical safeguard. The effect on displacement is below 1 um for
   r_rf >= 5 um, and the clipped cumulative count is invertible where the raw
   one is not.
3. Curcio's Figure 6 "Norm. mm" column implies an 11.556 mm sphere, while
   Appendix A prescribes 11.459 mm (a 0.84% discrepancy internal to the
   sources). Appendix A is followed.

Preserved caveats -- do not "fix" these
---------------------------------------
* On the nasal meridian the displacement-zone radius rDZ = 4.034 mm falls about
  34 um beyond the optic nerve head centre (20 deg co-latitude = 4.000 mm).
  ``SORTED.txt`` has no samples inside the disc, so the nasal cumulative count
  relies on the interpolant bridging it. Masking the disc instead moves nasal
  E2v to 1.91-2.00 and displacement by up to 218 um, and no longer matches the
  paper (published Fig. 4A nasal E2v 2.1804, this reconstruction 2.1855). The
  published behaviour is reproduced, not corrected.
* The field has two genuine angular discontinuities, near 11.05 deg and
  354.4 deg, where a local minimum of ``C_gcrf - C_gcb`` becomes the first
  cumulative-curve crossing and truncates the displacement zone. They jump by
  0.013-0.025 deg (4-7 um), persist under radial and angular refinement, and
  are visible in the paper's Figure 4B. They are not smoothed.
* Count conservation (Montesano's "Method 2") holds only inside the
  displacement zone. Beyond the first crossing the model sets r_soma = r_rf,
  making the Jacobian 1, which would additionally require the soma and RF
  densities to be pointwise equal. That is a property of the model, not a
  defect.
"""
import argparse
import hashlib
import io
import os
import sys
import zipfile

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import brentq

PAPER_DOI = '10.1167/tvst.9.10.15'
CURCIO_FILENAME = 'SORTED.txt'
CURCIO_SHA256 = ('b29a3eb50fb1bd56742b86e87a9f585488baf'
                 '94ec8646993b762a91149734c2e')

# --- histology map (Appendix A) ---------------------------------------------
R0_MM = 11.459                                    # retinal sphere radius
MM_PER_DEG_COLAT = 2 * np.pi * R0_MM / 360.0      # co-latitude deg -> mm

# --- schematic eye, Table A1 "Current Study" column -------------------------
R_RETINA = 11.459                    # retinal sphere radius (mm)
C_RETINA = 12.381                    # retinal sphere centre (mm from cornea)
R_CORNEA = 7.800                     # corneal apical radius of curvature (mm)
K_CORNEA = -0.500                    # corneal conic constant; see ambiguity 1
Z_LENS_ANT, R_LENS_ANT = 3.600, 10.000
Z_LENS_POST, R_LENS_POST = 7.375, -6.000
N_AIR, N_AQUEOUS, N_LENS = 1.0, 1.336, 1.430
NODAL_PUBLISHED = 6.930              # published first nodal point (mm)

# --- displacement model (Eq. 3, Appendix C) ---------------------------------
RV = 0.011785                        # Eq. 3
RO = 0.008333                        # Eq. 3
R_DZ_MM = 4.034                      # radius of the maximum displacement zone
E2V_BRACKET = (0.3, 12.0)            # audited: exactly one root per meridian

# --- grids ------------------------------------------------------------------
R_WORK_MM = np.arange(0.0, 6.0005, 0.001)    # 1 um radial working grid
ANGLE_STEP_DEG = 0.25                        # 1440 stored meridians
SUPPORT_MAX_DVA = 15.0                       # max zone extent is 14.10 deg
N_RADIAL_NODES = 751


# ===========================================================================
# Curcio & Allen (1990) histology
# ===========================================================================
def load_sorted(path):
    """Parse SORTED.txt: eye label, sphere radius, and the scattered samples.

    Returns co-latitude (deg), longitude (deg) and RGC soma density
    (cells/mm2) in file order, which is the order the Delaunay triangulation
    behind the interpolant is built from.
    """
    with open(path) as f:
        lines = f.read().split('\n')
    r0 = float(lines[1])
    n = int(lines[4])
    rows = [line.split() for line in lines[5:5 + n]]
    return dict(eye=lines[0].strip(), r0=r0,
                colat=np.array([float(r[2]) for r in rows]),
                lon=np.array([float(r[3]) for r in rows]),
                dens=np.array([float(r[4]) for r in rows]))


def build_density_interpolator(data):
    """Piecewise-cubic RGC soma density over the left-eye retinal plane.

    Co-latitude maps to retinal mm as ``r = R0 * colat``; samples are then laid
    out planar as ``(r cos lon, r sin lon)`` and interpolated with
    ``CloughTocher2DInterpolator``, the triangulation-based cubic scheme
    matching MATLAB's ``griddata(..., 'cubic')``. Validated against Montesano
    Figure A1 and Curcio's cardinal averages (RMSE 371-635 cells/mm2, every
    residual within 0.62 SD); not bit-for-bit equivalent to author code.
    """
    r_mm = data['colat'] * MM_PER_DEG_COLAT
    th = np.deg2rad(data['lon'])
    xy = np.column_stack([r_mm * np.cos(th), r_mm * np.sin(th)])
    return CloughTocher2DInterpolator(xy, data['dens'])


def sample_meridian(interp, lon_deg, r_mm):
    """Density along one left-eye source longitude, clipped at zero."""
    th = np.deg2rad(lon_deg)
    dens = interp(r_mm * np.cos(th), r_mm * np.sin(th))
    dens = np.maximum(dens, 0.0)     # cubic overshoot; see ambiguity 2
    dens[0] = 0.0                    # no somas at the foveal centre
    return dens


# ===========================================================================
# Schematic eye: Drasdo & Fowler (1974) as re-parameterised in Table A1
# ===========================================================================
# Meridional (2-D) numerical ray tracing. `z` runs along the optic axis from the
# corneal vertex into the eye, `y` is transverse. Rotational symmetry about the
# optic axis is assumed (Appendix B) and the fovea sits at the posterior pole.
# Visual angle is referred to the nodal point: for each angle the traced ray is
# the one whose emergent segment is parallel to the incident one.
def _refract(d, normal, n1, n2):
    """Snell's law in vector form; `normal` may point either way."""
    d = d / np.linalg.norm(d)
    normal = normal / np.linalg.norm(normal)
    if np.dot(d, normal) > 0:
        normal = -normal
    mu = n1 / n2
    cosi = -np.dot(d, normal)
    k = 1.0 - mu ** 2 * (1.0 - cosi ** 2)
    if k < 0:
        return None                  # total internal reflection
    return mu * d + (mu * cosi - np.sqrt(k)) * normal


def _hit_conic(p, d):
    """First intersection with the cornea ``(1+K)z^2 - 2Rz + y^2 = 0``."""
    y0, z0 = p
    dy, dz = d
    a = (1 + K_CORNEA) * dz ** 2 + dy ** 2
    b = (2 * (1 + K_CORNEA) * z0 * dz - 2 * R_CORNEA * dz + 2 * y0 * dy)
    c = (1 + K_CORNEA) * z0 ** 2 - 2 * R_CORNEA * z0 + y0 ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    ts = sorted(t for t in ((-b - np.sqrt(disc)) / (2 * a),
                            (-b + np.sqrt(disc)) / (2 * a)) if t > 1e-9)
    if not ts:
        return None
    q = p + ts[0] * d
    normal = np.array([2 * q[0],
                       2 * (1 + K_CORNEA) * q[1] - 2 * R_CORNEA])
    return q, normal


def _hit_sphere(p, d, centre_z, radius, far):
    """Intersection with an axial sphere; `far` selects the +z root."""
    oc = p - np.array([0.0, centre_z])
    b = 2 * np.dot(oc, d)
    c = np.dot(oc, oc) - radius ** 2
    disc = b ** 2 - 4 * c
    if disc < 0:
        return None
    t1, t2 = (-b - np.sqrt(disc)) / 2, (-b + np.sqrt(disc)) / 2
    ts = [t for t in ((t2, t1) if far else (t1, t2)) if t > 1e-9]
    if not ts:
        return None
    q = p + ts[0] * d
    return q, q - np.array([0.0, centre_z])


def trace(h, theta_deg):
    """Trace one ray crossing the corneal-vertex plane at height `h`.

    Returns the emergent angle, both axis crossings (the first and second nodal
    points for a nodal ray) and the retinal arc distance from the fovea.
    """
    th = np.deg2rad(theta_deg)
    d = np.array([-np.sin(th), np.cos(th)])      # travels +z, drifts to -y
    p = np.array([h, 0.0])

    hit = _hit_conic(p, d)
    if hit is None:
        return None
    q, normal = hit
    d = _refract(d, normal, N_AIR, N_AQUEOUS)
    if d is None:
        return None

    surfaces = ((Z_LENS_ANT + R_LENS_ANT, abs(R_LENS_ANT),
                 N_AQUEOUS, N_LENS, False),
                (Z_LENS_POST + R_LENS_POST, abs(R_LENS_POST),
                 N_LENS, N_AQUEOUS, True))
    for centre_z, radius, n1, n2, far in surfaces:
        hit = _hit_sphere(q, d, centre_z, radius, far)
        if hit is None:
            return None
        q, normal = hit
        d = _refract(d, normal, n1, n2)
        if d is None:
            return None

    hit = _hit_sphere(q, d, C_RETINA, R_RETINA, True)
    if hit is None:
        return None
    q_ret = hit[0]

    z_n1 = h / np.tan(th) if abs(th) > 1e-12 else np.nan
    z_n2 = q[1] - q[0] * d[1] / d[0] if abs(d[0]) > 1e-15 else np.nan
    phi = np.arctan2(abs(q_ret[0]), q_ret[1] - C_RETINA)
    return dict(emergent_deg=np.rad2deg(np.arctan2(-d[0], d[1])),
                nodal1=z_n1, nodal2=z_n2, phi_deg=np.rad2deg(phi),
                arc_mm=R_RETINA * phi)


def nodal_ray(theta_deg):
    """Ray at `theta_deg` whose emergent segment is parallel to it.

    The entrance height is bracketed around ``NODAL_PUBLISHED * tan(theta)``;
    the emergent angle increases monotonically with it.
    """
    def resid(h):
        r = trace(h, theta_deg)
        return np.nan if r is None else r['emergent_deg'] - theta_deg

    guess = NODAL_PUBLISHED * np.tan(np.deg2rad(theta_deg))
    hs = np.linspace(0.3 * guess, 2.0 * guess, 60)
    vals = np.array([resid(h) for h in hs])
    ok = np.isfinite(vals)
    hs, vals = hs[ok], vals[ok]
    idx = np.nonzero(np.diff(np.sign(vals)) != 0)[0]
    if len(idx) == 0:
        raise RuntimeError('no nodal-ray bracket at %g deg' % theta_deg)
    i = idx[0]
    h_star = brentq(resid, hs[i], hs[i + 1], xtol=1e-13, rtol=8.9e-16)
    return trace(h_star, theta_deg)


class SchematicEye:
    """Tabulated visual deg <-> retinal mm and solid deg -> mm2 conversions."""

    def __init__(self, theta_max=70.0, n=1401):
        theta = np.linspace(0.0, theta_max, n)
        phi = np.zeros(n)
        nodal1 = np.full(n, np.nan)
        for i, t in enumerate(theta[1:], 1):
            ray = nodal_ray(t)
            phi[i] = np.deg2rad(ray['phi_deg'])
            nodal1[i] = ray['nodal1']
        arc = R_RETINA * phi
        # Solid-angle Jacobian: retinal area per square visual degree.
        dphi = np.gradient(phi, np.deg2rad(theta))
        with np.errstate(invalid='ignore', divide='ignore'):
            area = (R_RETINA ** 2 * np.sin(phi) * dphi /
                    np.sin(np.deg2rad(theta)) * (np.pi / 180) ** 2)
        area[0] = np.gradient(arc, theta)[0] ** 2      # 0/0 limit at the fovea
        self.theta, self.arc, self.area = theta, arc, area
        # Appendix B averages the nodal point over rays from 0.1 to 67 deg.
        band = (theta >= 0.1) & (theta <= 67.0)
        self.nodal1_mean = float(np.nanmean(nodal1[band]))

    def deg_to_mm(self, e):
        return np.interp(e, self.theta, self.arc)

    def mm_to_deg(self, r):
        return np.interp(r, self.arc, self.theta)

    def mm2_per_deg2(self, e):
        return np.interp(e, self.theta, self.area)


# ===========================================================================
# Displacement model
# ===========================================================================
def k_factor(e):
    """Eq. C.1: ON/OFF midget RGC-RF correction, `e` in visual deg."""
    poly = 1.004 - 0.007209 * e + 0.001694 * e ** 2 - 0.00003765 * e ** 3
    return 1.0 + poly ** -2.0


def d_gcrf_per_deg2(e, e2v):
    """Eq. 3: RGC-RF density per solid visual degree at eccentricity `e` (deg).

    Transcribed literally, with no square root in the denominator: at e = 0 and
    E2v = 1.8 this gives 27815 RF/deg2 against Watson (2014)'s ~29600 peak
    midget RGC-RF density, whereas a square root would give ~250 /deg2.
    """
    num = k_factor(e) * (1.12 + 0.0273 * e)
    den = 1.155 * ((RV * (1 + e / e2v)) ** 2 - (RO * (1 + e / 20.0)) ** 2)
    return num / den


def cumulative(r_mm, dens_per_mm2):
    """Eq. C.2: ``C(r) = int_0^r 2 pi r D(r) dr`` by cumulative trapezoid."""
    integrand = 2 * np.pi * r_mm * dens_per_mm2
    out = np.zeros_like(r_mm)
    out[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_mm))
    return out


class MeridianFit:
    """Fits E2v on one retinal meridian and builds the RF -> soma map."""

    def __init__(self, eye, dens_gcb, r_mm):
        self.eye, self.r = eye, r_mm
        self.d_gcb = dens_gcb                       # RGC somas, per mm2
        self.e_deg = eye.mm_to_deg(r_mm)            # visual deg at each mm
        self.area = eye.mm2_per_deg2(self.e_deg)    # mm2 per solid deg
        self.c_gcb = cumulative(r_mm, self.d_gcb)
        self._i_dz = int(np.argmin(np.abs(r_mm - R_DZ_MM)))

    def c_gcrf(self, e2v):
        dens = d_gcrf_per_deg2(self.e_deg, e2v) / self.area     # per mm2
        return cumulative(self.r, dens)

    def residual(self, e2v):
        """``C_gcrf(rDZ) - C_gcb(rDZ)``; the fit drives this to zero."""
        return self.c_gcrf(e2v)[self._i_dz] - self.c_gcb[self._i_dz]

    def fit(self, bracket=E2V_BRACKET):
        # Eq. 3 has a pole where the two squared terms cancel. With Rv > Ro and
        # E2v below ~48 the denominator stays positive out to 20 deg, well past
        # the displacement zone; guard rather than assume.
        e_max = self.e_deg.max()
        for e2v in bracket:
            den = ((RV * (1 + e_max / e2v)) ** 2 -
                   (RO * (1 + e_max / 20.0)) ** 2)
            if den <= 0:
                raise ValueError('Eq. 3 pole inside bracket at E2v=%g' % e2v)
        self.e2v = brentq(self.residual, *bracket, xtol=1e-12, rtol=8.9e-16)
        self.c_rf = self.c_gcrf(self.e2v)
        return self.e2v

    def displacement(self):
        """Forward map with ``C_gcb(r_soma) = C_gcrf(r_rf)``.

        Displacement is zero beyond the first crossing of the two cumulative
        curves (Appendix C), so the map is the identity from there outward.
        Monotonicity and sign are not imposed; both are checked in `validate`.
        """
        diff = self.c_rf - self.c_gcb
        cross = np.nonzero((diff[:-1] > 0) & (diff[1:] <= 0))[0]
        r_cross = self.r[-1] if len(cross) == 0 else self.r[cross[0] + 1]
        # C_gcb is flat wherever the clipped density is zero, so invert only on
        # its strictly increasing part. Keeping the smallest radius per distinct
        # cumulative value is the monotone generalized inverse and pins
        # C_gcb = 0 -> r_soma = 0.
        keep = np.concatenate(([0], 1 + np.nonzero(np.diff(self.c_gcb) > 0)[0]))
        r_soma = np.interp(self.c_rf, self.c_gcb[keep], self.r[keep])
        beyond = self.r >= r_cross
        r_soma[beyond] = self.r[beyond]
        self.r_soma, self.r_cross = r_soma, r_cross
        return r_soma


def build_field(interp, eye, angles_deg, r_mm=R_WORK_MM, verbose=True):
    """Solve every anatomical meridian on the 1 um working grid.

    Returns soma radius (mm), fitted E2v, first-crossing radius (mm), the
    cumulative-count residual at rDZ and the soma count there, per meridian.
    """
    n = len(angles_deg)
    r_soma = np.empty((n, len(r_mm)))
    e2v = np.empty(n)
    r_cross = np.empty(n)
    resid = np.empty(n)
    c_dz = np.empty(n)
    for i, ang in enumerate(angles_deg):
        # right-eye anatomical angle -> left-eye source longitude
        lon = (180.0 - ang) % 360.0
        fit = MeridianFit(eye, sample_meridian(interp, lon, r_mm), r_mm)
        fit.fit()
        r_soma[i] = fit.displacement()
        e2v[i], r_cross[i] = fit.e2v, fit.r_cross
        resid[i] = fit.residual(fit.e2v)
        c_dz[i] = fit.c_gcb[fit._i_dz]
        if verbose and (i + 1) % 120 == 0:
            print('    %4d / %d meridians' % (i + 1, n), flush=True)
    return dict(r_soma=r_soma, e2v=e2v, r_cross=r_cross, resid=resid,
                c_dz=c_dz)


def resample(eye, r_mm, r_soma_mm):
    """Resample the field onto the canonical cube-root-spaced dva grid.

    Uniform radial grids cannot represent the near-foveal rise (0.12 deg error
    even at 0.005 deg spacing); cube-root spacing concentrates nodes there and
    reaches 0.0030 deg with 751 of them.
    """
    rf_deg = eye.mm_to_deg(r_mm)
    soma_deg = eye.mm_to_deg(r_soma_mm)
    nodes = np.linspace(0.0, SUPPORT_MAX_DVA ** (1 / 3), N_RADIAL_NODES) ** 3
    if nodes[-1] > rf_deg[-1]:
        raise ValueError('radial support exceeds the working grid')
    table = np.array([np.interp(nodes, rf_deg, row) for row in soma_deg])
    return nodes, table


# ===========================================================================
# Validation
# ===========================================================================
# Anchors from the audit of the fine (0.05 deg x 1 um) reconstruction. These
# guard against orientation, unit and model regressions; the full figure and
# count-conservation audit is not repeated here.
CARDINALS = (('nasal', 0.0), ('superior', 90.0), ('temporal', 180.0),
             ('inferior', 270.0))
E2V_EXPECTED = {'nasal': 2.18552, 'superior': 1.84865,
                'temporal': 2.15379, 'inferior': 1.68159}
ZONE_EXPECTED = {'nasal': 9.54, 'superior': 14.10,
                 'temporal': 14.10, 'inferior': 10.52}
# The two first-crossing discontinuities (visible in Fig. 4B); angular jumps
# here are model content, not regressions.
DISCONTINUITY_DEG = (11.05, 354.4)


def _fail(msg):
    raise SystemExit('FAILED: ' + msg)


def validate(angles, nodes, table, field, eye):
    """Check the artifact invariants and the published scientific anchors."""
    print('  invariants')
    if table.shape != (len(angles), N_RADIAL_NODES):
        _fail('shape %s, expected (%d, %d)'
              % (table.shape, len(angles), N_RADIAL_NODES))
    if not np.all(np.isfinite(table)):
        _fail('non-finite soma eccentricity')
    if table.min() < 0.0:
        _fail('negative soma eccentricity: %g' % table.min())
    origin_err = np.abs(table[:, 0]).max()
    if origin_err > 0.0:
        _fail('F(theta, 0) != 0: max %g' % origin_err)
    end_err = np.abs(table[:, -1] - nodes[-1]).max()
    if end_err > 1e-5:
        _fail('F(theta, %g) is not the identity: max error %g'
              % (nodes[-1], end_err))
    # Strictly increasing: the map is a cumulative-count correspondence, so a
    # flat or decreasing step would make the runtime inverse ill-posed.
    steps = np.diff(table, axis=1)
    if steps.min() <= 0.0:
        _fail('r_soma not strictly increasing with r_rf (min step %g)'
              % steps.min())
    disp = table - nodes[None, :]
    if disp.min() < 0.0:
        _fail('negative displacement: %g' % disp.min())
    print('    F(theta,0) = 0 exactly; |F(theta,15) - 15| <= %.1e deg'
          % end_err)
    print('    min radial step %.3e deg; displacement in [%.4f, %.4f] deg'
          % (steps.min(), disp.min(), disp.max()))

    print('  angular continuity')
    # A first difference between adjacent meridians measures the field's own
    # angular gradient, which legitimately reaches 0.014 deg per 0.25 deg step
    # in the outer displacement zone (e.g. the smooth 25-35 deg sector, where
    # E2v and the crossing radius are constant). A second difference stays
    # small under a steep but smooth gradient and spikes at a true jump, so
    # that is what gates here.
    n = len(angles)
    prev, nxt = np.roll(table, 1, axis=0), np.roll(table, -1, axis=0)
    curv = np.abs(prev - 2.0 * table + nxt).max(axis=1)
    jump = np.abs(np.diff(table, axis=0, append=table[:1])).max(axis=1)
    known = np.zeros(n, bool)
    for ang in DISCONTINUITY_DEG:
        known |= np.abs((angles - ang + 180.0) % 360.0 - 180.0) <= 0.75
    if curv[~known].max() > 0.01:
        i = int(np.argmax(np.where(known, -1.0, curv)))
        _fail('unexplained angular discontinuity, second difference %.4f deg '
              'at %.2f deg' % (curv[i], angles[i]))
    if jump.max() > 0.05:
        i = int(jump.argmax())
        _fail('angular gradient %.4f deg per %.2f deg at %.2f deg is far '
              'outside the audited field'
              % (jump[i], ANGLE_STEP_DEG, angles[i]))
    print('    second difference %.4f deg across the two known '
          'discontinuities' % curv[known].max())
    print('    second difference %.4f deg elsewhere; max smooth gradient '
          '%.4f deg per %.2f deg' % (curv[~known].max(), jump[~known].max(),
                                     ANGLE_STEP_DEG))

    print('  E2v fit residual at rDZ = %.3f mm' % R_DZ_MM)
    resid = np.abs(field['resid'])
    if resid.max() > 1e-5:
        _fail('E2v fit residual %g cells' % resid.max())
    print('    max %.2e cells (%.2e relative to %.0f-%.0f somas)'
          % (resid.max(), (resid / field['c_dz']).max(),
             field['c_dz'].min(), field['c_dz'].max()))

    print('  E2v by cardinal meridian')
    for name, ang in CARDINALS:
        got = field['e2v'][int(round(ang / ANGLE_STEP_DEG))]
        want = E2V_EXPECTED[name]
        if abs(got - want) > 2e-4:
            _fail('E2v %s = %.5f, expected %.5f' % (name, got, want))
        print('    %-9s %.5f (audit %.5f)' % (name, got, want))
    if field['e2v'].min() < 1.67 or field['e2v'].max() > 2.20:
        _fail('E2v outside the audited range: %.5f .. %.5f'
              % (field['e2v'].min(), field['e2v'].max()))
    print('    range %.5f .. %.5f (audit 1.67246 .. 2.19087)'
          % (field['e2v'].min(), field['e2v'].max()))

    print('  displacement-zone extent')
    # The extent is the last radius still displaced, i.e. one working-grid step
    # inside the first crossing.
    dr = float(np.diff(R_WORK_MM)[0])
    extent = eye.mm_to_deg(field['r_cross'] - dr)
    for name, ang in CARDINALS:
        got = extent[int(round(ang / ANGLE_STEP_DEG))]
        want = ZONE_EXPECTED[name]
        if abs(got - want) > 0.01:
            _fail('zone extent %s = %.2f deg, expected %.2f'
                  % (name, got, want))
        print('    %-9s %5.2f deg (audit %5.2f)' % (name, got, want))
    if extent.max() > SUPPORT_MAX_DVA:
        _fail('zone extent %.2f deg exceeds the %.1f deg radial support'
              % (extent.max(), SUPPORT_MAX_DVA))

    print('  global maximum displacement')
    i, j = np.unravel_index(disp.argmax(), disp.shape)
    if abs(disp[i, j] - 2.0843) > 0.005:
        _fail('max displacement %.4f deg, expected 2.0843' % disp[i, j])
    if abs((angles[i] - 182.75 + 180.0) % 360.0 - 180.0) > 1.0:
        _fail('max displacement at %.2f deg, expected near 182.75 (temporal)'
              % angles[i])
    if abs(nodes[j] - 2.439) > 0.05:
        _fail('max displacement at r_rf = %.3f deg, expected near 2.439'
              % nodes[j])
    print('    %.4f deg at theta = %.2f deg (temporal), r_rf = %.3f deg'
          '  (audit 2.0843, 182.75, 2.439)'
          % (disp[i, j], angles[i], nodes[j]))


# ===========================================================================
# Output
# ===========================================================================
def save_npz(path, **arrays):
    """Write a savez_compressed-compatible archive with fixed timestamps.

    ``np.savez_compressed`` stamps each member with the current time, so two
    runs produce different bytes. Normalising the timestamps makes the artifact
    byte-reproducible and its SHA256 a usable provenance record. Only numeric
    and string arrays are stored, so the result loads with allow_pickle=False.
    """
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            info = zipfile.ZipInfo(name + '.npy',
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arr),
                                      allow_pickle=False)
            zf.writestr(info, buf.getvalue())


def main(argv=None):
    default_out = os.path.join('pulse2percept', 'topography', 'retina',
                               'data', 'montesano2020.npz')
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('sorted_txt',
                   help='path to Curcio & Allen (1990) SORTED.txt')
    p.add_argument('--output', default=default_out,
                   help='output .npz path (default: %s)' % default_out)
    args = p.parse_args(argv)

    print('Curcio source: %s' % args.sorted_txt)
    with open(args.sorted_txt, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != CURCIO_SHA256:
        _fail('SHA256 mismatch for %s\n  got      %s\n  expected %s\n'
              'The reconstruction was audited against exactly this file; a\n'
              'different one must be re-audited before regenerating.'
              % (args.sorted_txt, digest, CURCIO_SHA256))
    print('  SHA256 %s OK' % digest)

    data = load_sorted(args.sorted_txt)
    print('  %s eye, sphere radius %.3f mm, %d samples'
          % (data['eye'], data['r0'], len(data['dens'])))
    interp = build_density_interpolator(data)

    print('Schematic eye (Table A1, K = %.3f)' % K_CORNEA)
    eye = SchematicEye()
    print('  first nodal point %.4f mm (published %.3f); foveal scale '
          '%.5f mm/deg' % (eye.nodal1_mean, NODAL_PUBLISHED,
                           np.gradient(eye.arc, eye.theta)[0]))

    angles = np.arange(0.0, 360.0, ANGLE_STEP_DEG)
    print('Solving %d meridians on a %.0f um radial grid out to %.1f mm'
          % (len(angles), 1e3 * np.diff(R_WORK_MM)[0], R_WORK_MM[-1]))
    field = build_field(interp, eye, angles)

    print('Resampling onto %d cube-root-spaced nodes out to %.1f deg'
          % (N_RADIAL_NODES, SUPPORT_MAX_DVA))
    nodes, table = resample(eye, R_WORK_MM, field['r_soma'])

    print('Validating')
    validate(angles, nodes, table, field, eye)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_npz(
        args.output,
        retinal_angle_deg=angles.astype(np.float32),
        rf_eccentricity_dva=nodes.astype(np.float64),
        soma_eccentricity_dva=table.astype(np.float32),
        paper_doi=np.str_(PAPER_DOI),
        curcio_source_filename=np.str_(CURCIO_FILENAME),
        curcio_source_sha256=np.str_(CURCIO_SHA256),
        reference_eye=np.str_('right'),
        angle_convention=np.str_('0=nasal, 90=superior, 180=temporal, '
                                 '270=inferior'),
        angular_resolution_deg=np.float64(ANGLE_STEP_DEG),
        radial_support_max_dva=np.float64(SUPPORT_MAX_DVA),
        retinal_sphere_radius_mm=np.float64(R0_MM),
    )
    with open(args.output, 'rb') as f:
        out_digest = hashlib.sha256(f.read()).hexdigest()
    print('Wrote %s (%.2f MB)'
          % (args.output, os.path.getsize(args.output) / 1e6))
    print('  SHA256 %s' % out_digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
