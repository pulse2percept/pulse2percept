import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

from pulse2percept.implants import (Electrode, DiskElectrode, PointSource,
                                    SquareElectrode, HexElectrode)
from pulse2percept.units import (DimensionMismatchError, Quantity, dva, mm,
                                 ms, uA, um)


class ValidElectrode(Electrode):
    __slots__ = ()

    def electric_potential(self, x, y, z):
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        return r


def test_Electrode():
    electrode = ValidElectrode(0, 1, 2, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_equal(electrode.name, 'A001')
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2), 0)
    with pytest.raises(TypeError):
        ValidElectrode([0], 1, 2)
    with pytest.raises(TypeError):
        ValidElectrode(0, np.array([1, 2]), 2)
    with pytest.raises(TypeError):
        ValidElectrode(0, 1, [2, 3])
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)


def test_PointSource():
    electrode = PointSource(0, 1, 2, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_equal(electrode.name, 'A001')
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2, 1, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, 0, 0, 1, 1), 0.035,
                            decimal=3)
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)
    # Plots:
    ax = electrode.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.patches), 1)
    npt.assert_equal(isinstance(ax.patches[0], Circle), True)


def test_DiskElectrode():
    with pytest.raises(TypeError):
        DiskElectrode(0, 0, 0, [1, 2])
    with pytest.raises(TypeError):
        DiskElectrode(0, np.array([0, 1]), 0, 1)
    # Invalid radius:
    with pytest.raises(ValueError):
        DiskElectrode(0, 0, 0, -5)
    # Check params:
    electrode = DiskElectrode(0, 1, 2, 100, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_equal(electrode.name, 'A001')
    # On the electrode surface (z=2, x^2+y^2<=100^2)
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(30, -30, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, 101, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, -99, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(100, 1, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(-100, 1, 2, 1), 1)
    # Right off the surface (z=2, x^2+y^2>100^2)
    npt.assert_almost_equal(electrode.electric_potential(0, 102, 2, 1), 0.910,
                            decimal=3)
    npt.assert_almost_equal(electrode.electric_potential(0, -100, 2, 1), 0.910,
                            decimal=3)
    # Some distance away from the electrode (z>2):
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 38, 1), 0.780,
                            decimal=3)
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)
    # Plots:
    ax = electrode.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.patches), 1)
    npt.assert_equal(isinstance(ax.patches[0], Circle), True)


def test_SquareElectrode():
    with pytest.raises(TypeError):
        SquareElectrode(0, 0, 0, [1, 2])
    with pytest.raises(TypeError):
        SquareElectrode(0, np.array([0, 1]), 0, 1)
    # Invalid radius:
    with pytest.raises(ValueError):
        SquareElectrode(0, 0, 0, -5)
    # Check params:
    electrode = SquareElectrode(0, 1, 2, 100, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.a, 100)
    npt.assert_equal(electrode.name, 'A001')
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)
    # Plots:
    ax = electrode.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.patches), 1)
    npt.assert_equal(isinstance(ax.patches[0], Rectangle), True)


def test_HexElectrode():
    with pytest.raises(TypeError):
        HexElectrode(0, 0, 0, [1, 2])
    with pytest.raises(TypeError):
        HexElectrode(0, np.array([0, 1]), 0, 1)
    # Invalid radius:
    with pytest.raises(ValueError):
        HexElectrode(0, 0, 0, -5)
    # Check params:
    electrode = HexElectrode(0, 1, 2, 100, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.a, 100)
    npt.assert_equal(electrode.name, 'A001')
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)
    # Plots:
    ax = electrode.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.patches), 1)
    npt.assert_equal(isinstance(ax.patches[0], RegularPolygon), True)


def _hex_extent(electrode, deactivated=False):
    """Return the (x, y) bounding-box size of a hexagon's plotted body"""
    kwargs = (electrode.plot_deactivated_kwargs if deactivated
              else electrode.plot_kwargs)
    patch = RegularPolygon((electrode.x, electrode.y), **kwargs)
    verts = patch.get_path().transformed(patch.get_patch_transform()).vertices
    return verts.max(axis=0) - verts.min(axis=0)


def test_HexElectrode_geometry():
    """``a`` is the apothem, so the flat-to-flat width is ``2 * a``"""
    a = 50
    # 'horizontal': flats face left/right, so the apothem is measured along x
    # and the hexagon is pointy-top:
    hexe = HexElectrode(0, 0, 0, a, orientation='horizontal')
    npt.assert_almost_equal(hexe.width, 2 * a)
    # Matplotlib sizes a RegularPolygon by its circumradius:
    npt.assert_almost_equal(hexe.plot_kwargs['radius'],
                            a / np.cos(np.radians(30)))
    npt.assert_almost_equal(hexe.plot_kwargs['orientation'], 0)
    npt.assert_almost_equal(_hex_extent(hexe),
                            [2 * a, 2 * a / np.cos(np.radians(30))])
    # 'vertical': flats face up/down, so the apothem is measured along y and
    # the hexagon is flat-top:
    vert = HexElectrode(0, 0, 0, a, orientation='vertical')
    npt.assert_almost_equal(vert.plot_kwargs['orientation'], np.radians(30))
    npt.assert_almost_equal(_hex_extent(vert),
                            [2 * a / np.cos(np.radians(30)), 2 * a])
    # Deactivated bodies have the same geometry, only a different color:
    npt.assert_almost_equal(_hex_extent(vert, deactivated=True),
                            _hex_extent(vert))
    with pytest.raises(ValueError):
        HexElectrode(0, 0, 0, a, orientation='diagonal')


@pytest.mark.parametrize('orientation', ('horizontal', 'vertical'))
def test_HexElectrode_rot(orientation):
    """``rot`` turns the hexagon body, not just the lattice"""
    a = 50
    unrot = HexElectrode(0, 0, 0, a, orientation=orientation)
    # A hexagon has 60 deg symmetry, so a 60 deg turn is a no-op:
    same = HexElectrode(0, 0, 0, a, orientation=orientation, rot=60)
    npt.assert_almost_equal(_hex_extent(same), _hex_extent(unrot))
    # ...and a 30 deg turn swaps pointy-top for flat-top:
    flipped = HexElectrode(0, 0, 0, a, orientation=orientation, rot=30)
    npt.assert_almost_equal(_hex_extent(flipped), _hex_extent(unrot)[::-1])
    # Positive `rot` is counter-clockwise, matching ElectrodeGrid:
    turned = HexElectrode(0, 0, 0, a, orientation=orientation, rot=15)
    npt.assert_almost_equal(turned.plot_kwargs['orientation'],
                            unrot.plot_kwargs['orientation'] +
                            np.radians(15))


def test_Electrode_units():
    """Equivalent spellings of a position must give the same electrode"""
    bare = DiskElectrode(1000, 0, 100, 200)
    unitful = DiskElectrode(1 * mm, 0 * mm, 0.1 * mm, 0.2 * mm)
    for attr in ('x', 'y', 'z', 'r'):
        npt.assert_allclose(getattr(unitful, attr), getattr(bare, attr),
                            rtol=1e-12)
        # Electrodes store plain numbers, whatever they were given:
        npt.assert_equal(isinstance(getattr(unitful, attr), Quantity), False)
    # Including conversions that do not land on a round number:
    awkward = DiskElectrode(0.0417 * mm, -8.3 * um, 0, 0.0083 * mm)
    npt.assert_allclose([awkward.x, awkward.y, awkward.r], [41.7, -8.3, 8.3],
                        rtol=1e-12)
    # Every electrode type takes a unitful size:
    npt.assert_allclose(SquareElectrode(0, 0, 0, 0.05 * mm).a, 50, rtol=1e-12)
    npt.assert_allclose(HexElectrode(0, 0, 0, 0.05 * mm).a, 50, rtol=1e-12)
    npt.assert_allclose(PointSource(1 * mm, 0, 0).x, 1000, rtol=1e-12)
    # A quantity wrapping an array is refused for the same reason a bare array
    # is, rather than being stored as one:
    with pytest.raises(TypeError):
        DiskElectrode(np.arange(3) * um, 0, 0, 100)
    with pytest.raises(TypeError):
        DiskElectrode(0, 0, 0, np.arange(1, 4) * um)


def test_Electrode_dimension_errors():
    for kwargs in ({'x': 5 * ms}, {'y': 5 * ms}, {'z': 5 * uA},
                   {'r': 10 * uA}):
        with pytest.raises(DimensionMismatchError):
            DiskElectrode(**{'x': 0, 'y': 0, 'z': 0, 'r': 100, **kwargs})
    for etype in (SquareElectrode, HexElectrode):
        with pytest.raises(DimensionMismatchError):
            etype(0, 0, 0, 2 * dva)
    # The message names the offending argument:
    with pytest.raises(DimensionMismatchError) as excinfo:
        DiskElectrode(0, 0, 0, 10 * uA)
    npt.assert_equal("Parameter 'r' expects length (um), got electric current"
                     in str(excinfo.value), True)


def test_Electrode_coordinates():
    elec = DiskElectrode(1000, 0, 100, 200)
    npt.assert_almost_equal(elec.coordinates(), [1000, 0, 100])
    npt.assert_allclose(elec.coordinates(mm), [1, 0, 0.1], rtol=1e-12)
    npt.assert_equal(elec.coordinate_unit, um)
    npt.assert_equal(isinstance(elec.coordinates(mm), np.ndarray), True)
    with pytest.raises(DimensionMismatchError):
        elec.coordinates(ms)


def test_electric_potential_units():
    """The point a potential is evaluated at is a position like any other"""
    disk = DiskElectrode(0, 0, 0, 100)
    npt.assert_allclose(disk.electric_potential(0.2 * mm, 0, 10 * um, 1),
                        disk.electric_potential(200, 0, 10, 1), rtol=1e-12)
    point = PointSource(0, 0, 0)
    npt.assert_allclose(point.electric_potential(0.2 * mm, 0, 0.01 * mm, 1, 1),
                        point.electric_potential(200, 0, 10, 1, 1), rtol=1e-12)
    for elec, args in [(disk, (1,)), (point, (1, 1))]:
        with pytest.raises(DimensionMismatchError):
            elec.electric_potential(1 * ms, 0, 0, *args)
        with pytest.raises(DimensionMismatchError):
            elec.electric_potential(0, 1 * uA, 0, *args)
