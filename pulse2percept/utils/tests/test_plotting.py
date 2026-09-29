import matplotlib.pyplot as plt
import numpy.testing as npt

from pulse2percept.utils._plotting import set_mm_ticks


def test_set_mm_ticks():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    set_mm_ticks(ax)
    ax.set_xlim(0, 36000)
    ax.set_zlim(-2500, 2500)
    fig.canvas.draw()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        fmt = axis.get_major_formatter()
        npt.assert_equal([fmt(t) for t in axis.get_majorticklocs()],
                         [f"{t / 1000:g}" for t in axis.get_majorticklocs()])
    # Clean labels: no '.0', float noise, or '-0'
    fmt = ax.xaxis.get_major_formatter()
    npt.assert_equal([fmt(v) for v in (0, 36000, 2500, 1e-10, -0.0)],
                     ['0', '36', '2.5', '0', '0'])
    plt.close(fig)
