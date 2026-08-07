# distutils: language = c++
# ^ needed for bool

from pulse2percept.utils._fast_math cimport c_isclose, float32
from libc.math cimport(fabs as c_abs, fmax as c_max)
from libcpp cimport bool
import numpy as np
cimport numpy as cnp
cnp.import_array()
ctypedef Py_ssize_t index_t


cpdef bool[::1] fast_compress_space(float32[:, ::1] data):
    """Compress a stimulus in space"""
    # In space, we only keep electrodes with nonzero activation values.
    # Note that `c_isclose(x, 0)` is an exact test, not a tolerant one: its
    # abs_tol is 0, so the comparison reduces to |x| <= 1e-9 * |x|, which
    # holds only for x == 0. Electrodes carrying a tiny but nonzero amplitude
    # are therefore kept, which is what "all-zero" above promises.
    cdef:
        index_t e, n_elec, t, n_time
        bool[::1] idx_space

    n_elec = data.shape[0]
    n_time = data.shape[1]
    idx_space = np.zeros(n_elec, dtype=np.bool_)  # Py overhead

    for e in range(n_elec):
        for t in range(n_time):
            if not c_isclose(data[e, t], 0):
                idx_space[e] = True
                break

    return np.asarray(idx_space)

cpdef bool[::1] fast_compress_time(float32[:, ::1] data):
    """Compress a stimulus in time"""
    # In time, we can't just remove empty columns. We need to walk
    # through each column and save all the "state transitions" along
    # with the points in time when they happened. For example, a
    # digital signal:
    # data = [0 0 1 1 1 1 0 0 0 1], time = [0 1 2 3 4 5 6 7 8 9]
    # becomes
    # data = [0 0 1 1 0 0 1],       time = [0 1 2 5 6 8 9].
    # You always need the first and last element. You also need the
    # high and low value (along with the time stamps) for every signal
    # edge.
    # Only the column indices matter here, so the time axis itself is not
    # needed: the caller applies the returned mask to it.
    # `c_isclose`'s rel_tol (1e-9) sits far below float32 eps (~1.2e-7), so
    # for float32 input this is exact equality for any pair of finite values.
    # It is not interchangeable with `!=` at the extremes, though:
    # c_isclose(inf, inf) is False, so a constant infinite signal registers an
    # edge at every column.
    cdef:
        index_t e, n_elec, t, n_time
        bool[::1] idx_time

    n_elec = data.shape[0]  # Py overhead
    n_time = data.shape[1]  # Py overhead
    idx_time = np.zeros(n_time, dtype=np.bool_)  # Py overhead

    for t in range(n_time):
        if t == 0 or t == n_time - 1:
            # We always need the first and last element:
            idx_time[t] = True
        else:
            # Determine if there is a signal edge (at least one element in
            # `t` must be different from `t-1`):
            for e in range(n_elec):
                if not c_isclose(data[e, t], data[e, t - 1]):
                    idx_time[t - 1] = True
                    idx_time[t] = True
                    break
    return np.asarray(idx_time)  # Py overhead
