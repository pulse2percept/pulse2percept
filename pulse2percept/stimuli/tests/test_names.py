import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import ElectrodeNames, ImageStimulus, Stimulus


def test_ElectrodeNames():
    names = ElectrodeNames((3, 4))
    npt.assert_equal(len(names), 12)
    npt.assert_equal(names.shape, (12,))
    npt.assert_equal(names.size, 12)
    npt.assert_equal(names.ndim, 1)
    npt.assert_equal(names.grid_shape, (3, 4))
    npt.assert_equal(names.grid_size, 12)
    # Letters address the row, digits the column:
    npt.assert_equal(np.asarray(names),
                     ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4',
                      'C1', 'C2', 'C3', 'C4'])
    npt.assert_equal(names[0], 'A1')
    npt.assert_equal(names[6], 'B3')
    npt.assert_equal(names[-1], 'C4')
    npt.assert_equal(list(names), np.asarray(names).tolist())
    npt.assert_equal(names.tolist(), np.asarray(names).tolist())

    # Beyond 26 rows, names continue AA, AB, ... just like ElectrodeGrid:
    npt.assert_equal(ElectrodeNames((30, 2))[-1], 'AD2')

    # An empty grid is still a valid (empty) set of names:
    npt.assert_equal(len(ElectrodeNames((0, 4))), 0)
    npt.assert_equal(np.asarray(ElectrodeNames((0, 4))).size, 0)

    with pytest.raises(ValueError):
        ElectrodeNames((3,))
    with pytest.raises(ValueError):
        ElectrodeNames((2, 3, 4, 5))
    with pytest.raises(ValueError):
        ElectrodeNames((-1, 3))


def test_ElectrodeNames_channels():
    names = ElectrodeNames((2, 3, 3))
    npt.assert_equal(np.asarray(names)[:4], ['A1_R', 'A1_G', 'A1_B', 'A2_R'])
    npt.assert_equal(names[-1], 'B3_B')
    # A fourth channel is the alpha channel:
    npt.assert_equal(ElectrodeNames((2, 2, 4))[3], 'A1_A')
    # Anything else falls back to a numeric suffix, so that every channel
    # remains addressable:
    npt.assert_equal(ElectrodeNames((2, 2, 5))[4], 'A1_4')


@pytest.mark.parametrize('grid_shape', [(3, 4), (2, 3, 3), (5, 7, 4),
                                        (30, 50), (1, 1)])
def test_ElectrodeNames_roundtrip(grid_shape):
    # Every generated name must map back onto the electrode it names, which is
    # what lets `index` work without ever building the names:
    names = ElectrodeNames(grid_shape)
    materialized = np.asarray(names)
    npt.assert_equal(materialized.size, names.size)
    npt.assert_equal([names.index(n) for n in materialized],
                     list(range(names.size)))
    # No name may occur twice:
    npt.assert_equal(len(np.unique(materialized)), names.size)


def test_ElectrodeNames_index():
    names = ElectrodeNames((3, 4))
    npt.assert_equal(names.index('A1'), 0)
    npt.assert_equal(names.index('B3'), 6)
    npt.assert_equal('C4' in names, True)
    npt.assert_equal('D1' in names, False)
    npt.assert_equal('nonsense' in names, False)

    # Outside the grid:
    with pytest.raises(ValueError):
        names.index('D1')
    with pytest.raises(ValueError):
        names.index('A5')
    # Not a name at all:
    with pytest.raises(ValueError):
        names.index('12A')
    with pytest.raises(KeyError):
        names.index(3)
    # Channel suffix on a grid that has no channels:
    with pytest.raises(ValueError):
        names.index('A1_R')
    # ... and vice versa:
    with pytest.raises(ValueError):
        ElectrodeNames((3, 4, 3)).index('A1')
    with pytest.raises(ValueError):
        ElectrodeNames((3, 4, 3)).index('A1_X')


def test_ElectrodeNames_indexing():
    names = ElectrodeNames((3, 4))
    # A slice keeps the names it selects:
    npt.assert_equal(np.asarray(names[1:4]), ['A2', 'A3', 'A4'])
    npt.assert_equal(names[1:4].is_unique, True)
    # So does a boolean mask:
    mask = np.zeros(12, dtype=bool)
    mask[[1, 5]] = True
    npt.assert_equal(np.asarray(names[mask]), ['A2', 'B2'])
    npt.assert_equal(names[mask].is_unique, True)
    # Fancy indexing may repeat, so uniqueness has to be established:
    npt.assert_equal(np.asarray(names[[0, 0, 2]]), ['A1', 'A1', 'A3'])
    npt.assert_equal(names[[0, 0, 2]].check_unique(), False)
    npt.assert_equal(names[[0, 2]].check_unique(), True)
    # A name is not an index; callers fall back to `index` on KeyError:
    with pytest.raises(KeyError):
        names['A1']

    # Subsets renumber, but keep pointing at the original electrodes:
    sub = names[4:]
    npt.assert_equal(sub[0], 'B1')
    npt.assert_equal(sub.index('B1'), 0)
    npt.assert_equal(sub.index('C4'), 7)
    with pytest.raises(ValueError):
        sub.index('A1')


def test_ElectrodeNames_reshape():
    names = ElectrodeNames((4, 5))
    grid = names.reshape((4, 5))
    npt.assert_equal(grid.shape, (4, 5))
    npt.assert_equal(grid[1, 2], 'B3')
    # This is how `crop` carries the original names over:
    cropped = grid[1:3, 2:4].ravel()
    npt.assert_equal(np.asarray(cropped), ['B3', 'B4', 'C3', 'C4'])
    npt.assert_equal(cropped.is_unique, True)
    npt.assert_equal(np.asarray(names.reshape(2, 10)).shape, (2, 10))
    # ravel() of an already-flat container is a no-op:
    npt.assert_equal(names.ravel() is names, True)


def test_ElectrodeNames_equality():
    npt.assert_equal(np.all(ElectrodeNames((3, 4)) == ElectrodeNames((3, 4))),
                     True)
    npt.assert_equal(np.all(ElectrodeNames((3, 4)) == ElectrodeNames((4, 3))),
                     False)
    # Comparison is elementwise, so a stimulus can be indexed by
    # `stim[stim.electrodes != 'A1']`:
    names = ElectrodeNames((2, 2))
    npt.assert_equal(names == 'A1', [True, False, False, False])
    npt.assert_equal(names != 'A1', [False, True, True, True])
    # Two views of differently-shaped grids still compare by name:
    npt.assert_equal(np.all(ElectrodeNames((1, 2)) == np.array(['A1', 'A2'])),
                     True)


def test_ElectrodeNames_copy():
    names = ElectrodeNames((3, 4))[1:5]
    clone = names.copy()
    npt.assert_equal(np.asarray(clone), np.asarray(names))
    clone.indices[0] = 11
    npt.assert_equal(names[0], 'A2')


def test_ElectrodeNames_is_lazy():
    # The whole point of the structure: naming a million electrodes must not
    # cost a million strings. Building the names, copying them and looking one
    # up all have to stay independent of the size of the grid.
    names = ElectrodeNames((2000, 2000, 3))
    npt.assert_equal(names.size, 12000000)
    npt.assert_equal(names.indices.size, 12000000)
    npt.assert_equal(names[0], 'A1_R')
    npt.assert_equal(names[-1], 'BXX2000_B')
    npt.assert_equal(names.index('BXX2000_B'), names.size - 1)


def test_Stimulus_with_ElectrodeNames():
    data = np.arange(12, dtype=np.float32).reshape((-1, 1))
    stim = Stimulus(data, electrodes=ElectrodeNames((3, 4)))
    npt.assert_equal(stim.electrodes[6], 'B3')
    # Addressing an electrode by name goes through ElectrodeNames.index:
    npt.assert_almost_equal(stim['B3'], 6)
    npt.assert_almost_equal(stim['C4'], 11)
    with pytest.raises(ValueError):
        stim['D1']

    # Removing by name keeps the remaining names intact:
    stim.remove('B3')
    npt.assert_equal(len(stim.electrodes), 11)
    npt.assert_equal('B3' in stim.electrodes, False)
    npt.assert_almost_equal(stim['C4'], 11)

    # Duplicate names are still caught, even when they come from a lazy
    # container (a repeated index is the only way to produce them):
    with pytest.warns(UserWarning):
        Stimulus(np.zeros((3, 1)),
                 electrodes=ElectrodeNames((3, 4))[[0, 0, 1]])


def test_ImageStimulus_electrode_names(tmp_path):
    from skimage.io import imsave
    fname = str(tmp_path / 'test.png')
    imsave(fname, np.random.randint(0, 255, (5, 7), dtype=np.uint8))
    stim = ImageStimulus(fname)
    npt.assert_equal(isinstance(stim.electrodes, ElectrodeNames), True)
    npt.assert_equal(stim.electrodes[0], 'A1')
    npt.assert_equal(stim.electrodes[-1], 'E7')
    # A pixel keeps its name through an operation that preserves the shape:
    npt.assert_equal(stim.invert().electrodes[8], 'B2')
    # ... and through a crop, so that a pixel can still be identified:
    cropped = stim.crop(left=2, top=1)
    npt.assert_equal(cropped.img_shape, (4, 5))
    npt.assert_equal(cropped.electrodes[0], 'B3')
    npt.assert_equal(cropped.electrodes[-1], 'E7')
    npt.assert_almost_equal(cropped['B3'], stim['B3'])
