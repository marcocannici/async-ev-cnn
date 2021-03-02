import numpy as np
cimport numpy as np
np.import_array()
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
# from cython.parallel import parallel, prange


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef data_to_np_int32(void * ptr, np.npy_intp dim, int shrink=0):
    if shrink:
        ptr = realloc(ptr, dim * sizeof(int))
    cdef np.ndarray[np.int32_t, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &dim, np.NPY_INT32, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def im2col_event(np.ndarray[np.float32_t, ndim=3] image, np.ndarray[np.int32_t, ndim=1] event_y,
                 np.ndarray[np.int32_t, ndim=1] event_x, int k_height, int k_width, int stride, int chan_as_cols=0):
    """
    Performs the im2col operation on the image extracting only the receptive fields (as columns) affected by the events.
    If two close events have overlapping receptive field, the columns are not extracted multiple times, but only once.
    The method returns the columns and a pair of arrays containing the y and x coordinates of the column in the output
    feature map.

    :param image: A float32 numpy ndarray of shape [in_channel, in_height, in_width] from which to extract columns
    :param event_y: An int32 numpy array containing num_events y coordinates of the events around which to extract cols
    :param event_x: An int32 numpy array containing num_events x coordinates of the events around which to extract cols
    :param k_height: Scalar int32, the height of the kernel
    :param k_width: Scalar int32, the width of the kernel
    :param stride: Scalar int32, the stride. It assumes equal stride in the x and y directions.
    :return: - out_cols: A float32 numpy ndarray od shape [in_channel * in_height * in_width, num_cols] containing
                the extracted columns
             - (out_cols_y, out_cols_x): A pair of int32 numpy arrays, containing the output coordinates on each
                column in the output feature map based on the provided kernel shape and stride.
    """

    cdef int num_events = event_y.shape[0]
    cdef int in_channel = image.shape[0]
    cdef int in_height = image.shape[1]
    cdef int in_width = image.shape[2]
    cdef int out_height = (in_height - k_height) // stride + 1
    cdef int out_width = (in_width - k_width) // stride + 1
    cdef np.ndarray[np.float32_t, ndim=2, mode='fortran'] out_cols
    cdef int* out_covered = <int *>malloc(out_height * out_width * sizeof(int))
    cdef int* out_cols_y = <int *>malloc(out_height * out_width * sizeof(int))
    cdef int* out_cols_x = <int *>malloc(out_height * out_width * sizeof(int))
    cdef int out_y_rf, out_x_rf
    cdef int next_out_idx = 0
    cdef int event_idx = 0
    cdef int y, x, y_min_rf, y_max_rf, x_min_rf, x_max_rf, out_len
    cdef int offset_y, offset_x, rf_offset_x, rf_offset_y, rf_chn, rf_y, rf_x, top_y, left_x, row_idx, col_idx

    out_covered = <int *>memset(out_covered, 0, out_height * out_width * sizeof(int))
    if chan_as_cols:
        out_cols = np.empty((k_height * k_width, in_channel * out_height * out_width), dtype=np.float32, order='F')
    else:
        out_cols = np.empty((in_channel * k_height * k_width, out_height * out_width), dtype=np.float32, order='F')

    # Loops through the events building the out_cols matrix while keeping track
    # of the output coordinates of each column
    for event_idx in range(num_events):
        y = event_y[event_idx]
        x = event_x[event_idx]

        # Compute the receptive field around the (y, x) coordinate
        if stride == 1:
            y_min_rf = int_max(0, y - (k_height - 1))
            y_max_rf = int_min(in_height, y + (k_height - 1) + 1)
            x_min_rf = int_max(0, x - (k_width - 1))
            x_max_rf = int_min(in_width, x + (k_width - 1) + 1)
        elif stride == k_width and stride == k_height:
            y_min_rf = (y // stride) * k_height
            y_max_rf = y_min_rf + k_height
            x_min_rf = (x // stride) * k_width
            x_max_rf = x_min_rf + k_width
        else:
            raise NotImplementedError("This method only support stride equal to 1 or to the kernel's dimensions.")

        # Computes how many kernels cover the receptive field of the current event
        num_height = (y_max_rf - y_min_rf - k_height) // stride + 1
        num_width = (x_max_rf - x_min_rf - k_width) // stride + 1

        # Loops through the receptive fields in the current patch
        offset_y = 0
        for offset_y from 0 <= offset_y < num_height by stride:
            offset_x = 0
            for offset_x from 0 <= offset_x < num_width by stride:
                # Computes the top left coordinates
                top_y = y_min_rf + offset_y
                left_x = x_min_rf + offset_x
                # Computes the coordinate where the current receptive field will map
                out_y_rf = top_y // stride
                out_x_rf = left_x // stride

                # If this position has not been covered yet
                if out_covered[out_y_rf * out_width + out_x_rf] == 0:
                    out_covered[out_y_rf * out_width + out_x_rf] = 1

                    out_cols_y[next_out_idx] = out_y_rf
                    out_cols_x[next_out_idx] = out_x_rf

                    # with nogil, parallel():
                    #     for rf_chn in prange(in_channel, schedule='guided'):
                    for rf_chn in range(in_channel):
                        for rf_offset_y in range(k_height):
                            rf_y = top_y + rf_offset_y
                            for rf_offset_x in range(k_width):
                                rf_x = left_x + rf_offset_x
                                if chan_as_cols:
                                    row_idx = rf_offset_y * k_width + rf_offset_x
                                    col_idx = in_channel * next_out_idx + rf_chn
                                else:
                                    row_idx = rf_chn * (k_height * k_width) + rf_offset_y * k_width + rf_offset_x
                                    col_idx = next_out_idx
                                out_cols[row_idx, col_idx] = image[rf_chn, rf_y, rf_x]

                    next_out_idx += 1

    free(out_covered)
    out_len = next_out_idx * in_channel if chan_as_cols else next_out_idx
    return out_cols[:, :out_len], (data_to_np_int32(out_cols_y, next_out_idx, shrink=1),
                                   data_to_np_int32(out_cols_x, next_out_idx, shrink=1))


@cython.boundscheck(False)
@cython.wraparound(False)
def min_argmax(np.ndarray[np.float32_t, ndim=2, mode='fortran'] max_arg,
               np.ndarray[np.float32_t, ndim=2, mode='fortran'] min_arg):
    """
    Computes the argmax of every column in 'max_arg' argument, if there are multiple maximum values, selects the one
    with the lowest value in 'min_arg' matrix. For each column, it determines if the selected position is also argmin
    of the column in the 'min_arg' matrix.

    :param max_arg: a float32 numpy ndarray of shape [num_rows, num_cols] used to compute the argmax
    :param min_arg: a float32 numpy ndarray of shape [num_rows, num_cols] used to compute the argmin
    :return: - argmax: a float32 array of shape [num_cols] representing the argmax indices
             - not_argmin_y: boolean array of shape [num_cols] containing a boolean value for each one of the indices,
               specifying if the selected argmax position is also argmin in 'min_arg'
    """

    cdef int col, row, argmax, argmin
    cdef int num_rows = max_arg.shape[0]
    cdef int num_cols = max_arg.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] argmax_cols = np.empty((num_cols), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] not_argmin = np.empty((num_cols), dtype=np.int32)

    # with nogil, parallel():
    #     for col in prange(num_cols, schedule='guided'):
    for col in range(num_cols):
        argmax = 0
        argmin = 0
        for row in range(num_rows):
            # Updates the argmax
            if max_arg[row, col] > max_arg[argmax, col]:
                argmax = row
            elif row > 0 and max_arg[row, col] == max_arg[argmax, col]:
                if min_arg[row, col] < min_arg[argmax, col]:
                    argmax = row

            # Updates the argmin
            if min_arg[row, col] < min_arg[argmin, col]:
                argmin = row

        argmax_cols[col] = argmax
        not_argmin[col] = min_arg[argmax, col] != min_arg[argmin, col]

    return argmax_cols, not_argmin
