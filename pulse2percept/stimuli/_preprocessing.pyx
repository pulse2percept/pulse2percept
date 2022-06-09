from ..utils._fast_math cimport c_isclose, float32, int32, uint32
from libc.math cimport(fabs as c_abs)
from skimage.filters import gaussian
import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef motion_saliency_x(image_gray, int32 N=4):
    """Calculates the motion saliency map in x direction 

    This function calculates the motion saliency map in x direction based on the algorithm in [Liu et al.1992]

    Parameters
    ----------
    image_gray : A 2D NumPy array 
      represents a (height, width) grayscale image
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt_x[j, x, y] will be determined by if there is motion in the block 
      containing the pixel [x,y] in j th degree
    """
    cdef:
        int32 j, u, v, x_, y_, x
        float32 denom
    Wt_x = np.zeros((N, image_gray.shape[0], image_gray.shape[1]), dtype=float)
    for j in range(N):
        for u in range(image_gray.shape[0]-4):
            for v in range(image_gray.shape[1]-4):
                # denom is F_k(u,v)
                denom = 0
                for x_ in range(N):
                    for y_ in range(N):
                        denom += image_gray[u+x_, v+y_]

                for x_ in range(N):
                    x = (u-1)+x_
                # v//N*N+i is trying to find corresponding four consecutive y from one y that is a multiple of N
                    Wt_x[j, u, v] += ((x-j) % N)*(image_gray[x, v//N*N]+image_gray[x,
                                                                                   v//N*N+1]+image_gray[x, v//N*N+2]+image_gray[x, v//N*N+3])/denom
    return Wt_x


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef motion_saliency_y(image_gray, int32 N=4):
    """Calculates the motion saliency map in y direction 

    This function calculates the motion saliency map in y direction using the Eq #2 in [Liu et al.1992]

    Parameters
    ----------
    image_gray : A 2D NumPy array 
      represents a (height, width) grayscale image
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt_y[j, x, y] will be determined by if there is motion in the block 
      containing the pixel [x,y] in j th degree
    """
    cdef:
        int32 j, u, v, x_, y_, y
        float32 denom
    Wt_y = np.zeros((N, image_gray.shape[0], image_gray.shape[1]))
    for j in range(N):
        for u in range(image_gray.shape[0]-4):
            for v in range(image_gray.shape[1]-4):
                # denom is F_k(u,v)
                denom = 0
                for x_ in range(N):
                    for y_ in range(N):
                        denom += image_gray[u+x_, v+y_]

                for y_ in range(N):
                    y = (v-1)+y_
                # v//N*N+i is trying to find corresponding four consecutive y from one y that is a multiple of N
                    Wt_y[j, u, v] += ((y-j) % N)*(image_gray[u//N*N, y]+image_gray[u //
                                                                                   N*N+1, y]+image_gray[u//N*N+2, y]+image_gray[u//N*N+3, y])/denom
    return Wt_y

cpdef temporal_saliency(image_gray, second_frame, int32 N=4, float32 boundary=0.5):
    """Calculates the temporal saliency map 

    This function calculates the temporal saliency map that used to identify moving objects based on the CSM-Based Algorithm in [Liu et al.1992]

    Parameters
    ----------
    image_gray : A 2D NumPy array 
      represents a (height, width) grayscale image
    second_frame : A 2D NumPy array
      represents another (height, width) grayscale image which is the previous or the next frame of image_gray
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt_x[x, y] will be determined by if there is motion in the block 
      containing the pixel [x,y]
    boundary : float, optional
      claims that there is a change occurs when the difference in the motion saliency map of two images is higher than the boundary
    """
    cdef:
        int32 u, v, j
    Wt = np.zeros((second_frame.shape[0], second_frame.shape[1]))
    Wt_x = motion_saliency_x(image_gray, N)
    Wt_y = motion_saliency_y(image_gray, N)
    Wt_2_x = motion_saliency_x(second_frame, N)
    Wt_2_y = motion_saliency_y(second_frame, N)
    for u in range(second_frame.shape[0]):
        for v in range(second_frame.shape[1]):
            for j in range(N):
                if (c_abs(Wt_x[j, u, v]-Wt_2_x[j, u, v]) > boundary or c_abs(Wt_y[j, u, v]-Wt_2_y[j, u, v]) > boundary):
                    Wt[u, v] = 1
    return Wt


cpdef seam_carving_energy(Wst, image_gray_shape):
    """Calculate the energy map for seam carving

    This function calculates the culmulative energy map for seam carving based on the spatio-temporal importance matrix

    Parameters
    ----------
    Wst : A 2D NumPy array 
      represents the spatio-temporal importance matrix
    image_gray_shape: A tuple of int
      represents the shape of the image
    """
    cdef:
        int32 i, j, id
    # This is implementing Eq. 8 from the paper:
    M_v = np.zeros(image_gray_shape)
    bt_v = np.zeros(image_gray_shape)
    for j in range(0, image_gray_shape[1]):
        for i in range(0, image_gray_shape[0]):
            if (i == 0):
                M_v[i, j] = Wst[i, j]
                continue
            if (j == 0):
                id = np.argmin(Wst[i-1, j:j+2])+j
                bt_v[i, j] = id
                M_v[i, j] = M_v[i-1, j]+Wst[i-1, id]
            elif (i == image_gray_shape[0]-1):
                id = np.argmin(Wst[i-1, j-1:j+1])+j-1
                bt_v[i, j] = id
                M_v[i, j] = M_v[i-1, j]+Wst[i-1, id]
            else:
                id = np.argmin(Wst[i-1, j-1:j+2])+j-1
                bt_v[i, j] = id
                M_v[i, j] = M_v[i-1, j]+Wst[i-1, id]
    return M_v, bt_v

cpdef seam_carvings(M_v, bt_v, Wst, image_gray_shape, int32 num=15):
    """Calculate the modified importance matrix with seam carving information

    This function calculates the updated importance matrix with seam carving based on the spatio-temporal importance matrix

    Parameters
    ----------
    M_v: A 2D NumPy array
      represents the culmulative energy map for seam carviing
    bt_v: A 2D NumPy array
      represents the backtrace information of the culmulative energy map
    Wst : A 2D NumPy array 
      represents the spatio-temporal importance matrix
    image_gray_shape: A tuple of int
      represents the shape of the image
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    cdef:
        int32 i, carving_end, carv_point, i_axis
        float32 min_weight, max_weight
    lowest = np.argpartition(
        M_v[image_gray_shape[0]-1, 0:image_gray_shape[1]], num)
    Wst_rescaled = Wst.copy()

    # This is implementing Eq. 10 from the paper:
    for i in range(num):
        carving_end = lowest[i]
        carv_point = carving_end
        i_axis = image_gray_shape[0]-1
        min_weight = 999
        max_weight = -999
        while (i_axis > 0):
            min_weight = min(min_weight, Wst_rescaled[i_axis, carv_point])
            max_weight = max(max_weight, Wst_rescaled[i_axis, carv_point])
            carv_point = bt_v[i_axis, carv_point]
            i_axis = i_axis-1
        i_axis = image_gray_shape[0]-1
        while (i_axis > 0):
            Wst_rescaled[i_axis, carv_point] = 0.001 * \
                (Wst_rescaled[i_axis, carv_point] -
                 min_weight)/(max_weight-min_weight)
            carv_point = bt_v[i_axis, carv_point]
            i_axis = i_axis-1
    return Wst_rescaled

cpdef importance_matrix(Wst, image_gray_shape, int32 L=5, int32 num=15):
    """Calculates final importance matrix

    This function calculates the final importance matrix from the modified importance matrix

    Parameters
    ----------
    Wst : np.array
      this is a blah doing bluh. values between 0 and 1
    image_gray_shape: A tuple of int
      represents the shape of the image
    L: int, optional
      a moving average window of size L would be applied to the importance matrix to preserve continuity between rows
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    cdef:
        int32 r_range, j, i
        float32 W_max
    M_v, bt_v = seam_carving_energy(Wst, image_gray_shape)
    Wst_rescaled = seam_carvings(M_v, bt_v, Wst, image_gray_shape, num)
    r_range = 2*image_gray_shape[1]//L
    new_W = np.zeros((image_gray_shape[0], image_gray_shape[1]))

    # This is implementing Eq. 11 and 12 from the paper:
    for j in range(image_gray_shape[1]):
        for i in range(L//2+1, image_gray_shape[0]-L//2):
            new_W[i, j] = np.average(Wst_rescaled[i-L//2:i+L//2+1, j])
        W_max = max(new_W[L//2+1:image_gray_shape[0]-L//2, j])
        for i in range(image_gray_shape[0]):
            new_W[i, j] = W_max

    # This is implementing Eq. 13 from the paper:
    W_fin = gaussian(new_W, 1)

    return W_fin


cpdef shrink_row(row, retain_factor, K):
    """shrink a row 

    This function shrink a row by the retain factor

    Parameters
    ----------
    row : 1d Numpy array
      represents one row in the image  
    retain_factor : 1d Numpy array
      represents the retain factor of each row
    K: int
      reduces the source image width by K pixels.
    """
    cdef:
        int32 i
        float32 rf, acc_index, acc_pixel, rf_current
    out = []
    i = 0
    acc_index = 0
    acc_pixel = 0
    while i < len(row) and len(out) < len(row)-K:
        rf = retain_factor[i]
        if acc_index+rf < 1:
            acc_index += rf
            acc_pixel += rf*row[i]
            i += 1
        else:
            rf_current = 1 - acc_index
            acc_index = rf - rf_current
            acc_pixel += rf_current*row[i]
            out.append(acc_pixel)
            acc_pixel = acc_index*row[i]
            i += 1
    while len(out) < len(row)-K:
        out.append(0)
    return np.array(out)
