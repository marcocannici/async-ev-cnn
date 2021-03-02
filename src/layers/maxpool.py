import numpy as np

from src.libs.cutils import im2col_event, min_argmax
from src.layers.functional import im2col
from src.layers.layer import Layer


class MaxPoolLayer(Layer):
    """
    Perform max pooling generating events whenever the maximum value in a region changes. The computation is
    performed incrementally, by storing the argmax indices of the previous call and updating them only where needed.
    """

    def __init__(self, prev_layer, ksize, stride):
        """
        :param prev_layer: The previous Layer object in the network
        :param ksize: The kernels size as a python list [k_height, k_width]
        :param stride: The stride
        """

        self._prev_layer = prev_layer
        self._ksize = ksize
        self._stride = stride

        in_channels, in_height, in_width = prev_layer.out_shape()
        k_height, k_width = ksize
        out_height = (np.floor((in_height - k_height) / stride) + 1).astype(np.int32)
        out_width = (np.floor((in_width - k_width) / stride) + 1).astype(np.int32)

        self._out_shape = [in_channels, out_height, out_width]
        prevlayer_surface = prev_layer.surface().reshape(in_channels, 1, in_height, in_width)
        surface_col, _ = im2col(prevlayer_surface, [in_channels, in_channels, *self._ksize], self._stride)
        self._init_idx_max = [surface_col.argmax(0).astype(np.int32),
                              np.arange(in_channels * out_height * out_width, dtype=np.int32)]
        self._idx_max = [self._init_idx_max[0].copy(), self._init_idx_max[1].copy()]
        self._recompute_coords = np.zeros([out_height, out_width], dtype=np.bool)

        self._cache_surface = None
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

    def surface(self):

        if self._cache_surface is None:
            in_channels, in_height, in_width = self._prev_layer.out_shape()
            prevlayer_surface = self._prev_layer.surface()

            # Makes the channels the batch dimension
            prevlayer_surface = prevlayer_surface.reshape(in_channels, 1, in_height, in_width)
            surface_col, _ = im2col(prevlayer_surface, [in_channels, in_channels, *self._ksize], self._stride)
            self._cache_surface = surface_col[tuple(self._idx_max)].reshape(self._out_shape)

        return self._cache_surface

    def layer_actfn(self):

        if self._cache_layer_actfn is None:
            in_channels, in_height, in_width = self._prev_layer.out_shape()
            prevlayer_actfn = self._prev_layer.layer_actfn()

            # Makes the channels the batch dimension
            prevlayer_actfn = prevlayer_actfn.reshape(in_channels, 1, in_height, in_width)
            actfn_col, _ = im2col(prevlayer_actfn, [in_channels, in_channels, *self._ksize], self._stride)
            self._cache_layer_actfn = actfn_col[tuple(self._idx_max)].reshape(self._out_shape)

        return self._cache_layer_actfn

    def conv_actfn(self):

        if self._cache_conv_actfn is None:
            in_channels, in_height, in_width = self._prev_layer.out_shape()
            prevlayer_conv_actfn = self._prev_layer.conv_actfn()

            # Makes the channels the batch dimension
            prevlayer_conv_actfn = prevlayer_conv_actfn.reshape(in_channels, 1, in_height, in_width)
            conv_actfn_col, _ = im2col(prevlayer_conv_actfn, [in_channels, in_channels, *self._ksize], self._stride)
            self._cache_conv_actfn = conv_actfn_col[tuple(self._idx_max)].reshape(self._out_shape)

        return self._cache_conv_actfn

    def out_shape(self):
        return self._out_shape

    def reset(self):
        _, out_height, out_width = self._out_shape
        self._idx_max = [self._init_idx_max[0].copy(), self._init_idx_max[1].copy()]
        self._recompute_coords = np.zeros([out_height, out_width], dtype=np.bool)
        self._cache_surface = None
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

    def _group_cols(self, array2d, num_groups, block_size):

        rows, cols = array2d.shape
        shape = [num_groups, cols // (num_groups * block_size), rows, block_size]
        strides = [block_size * rows, num_groups * block_size * rows, 1, rows]
        strides = np.array(strides) * array2d.itemsize

        grouped_array = np.lib.stride_tricks.as_strided(array2d, shape=shape, strides=strides)
        groups = np.split(grouped_array, num_groups, axis=0)
        groups = [np.concatenate(group[0], axis=1) for group in groups]

        return groups

    def compute(self, events, delta_leak):

        # Retrieves variables and shapes from previous layer
        prevlayer_surface = self._prev_layer.surface()
        prevlayer_actfn = self._prev_layer.conv_actfn()
        _, out_height, out_width = self._out_shape
        in_channels, _, _ = self._prev_layer.out_shape()

        k_height, k_width = self._ksize
        y, x = events

        # Remove the current events from the recompute list
        # Receptive fields to be recomputed are saved with their output coordinates
        out_y = y // self._stride
        out_x = x // self._stride
        self._recompute_coords[out_y, out_x] = False
        # Retrieves the events to be recomputed
        # Because they are saved with the output coords, we need to convert them back to the input space
        recomp_y, recomp_x = np.where(self._recompute_coords)
        recomp_events = [
            np.concatenate([y, recomp_y * self._stride]).astype(np.int32),
            np.concatenate([x, recomp_x * self._stride]).astype(np.int32)]

        # surface_col [k_height * k_width, img_channel * out_h * out_w]
        # Stacks prevlayer_surface and prevlayer_actfn in order to compute im2col just one time
        surface_col, (out_y, out_x) = im2col_event(prevlayer_surface,
                                                   *recomp_events, k_height,
                                                   k_width, self._stride,
                                                   chan_as_cols=True)
        actfn_col, _ = im2col_event(prevlayer_actfn, *recomp_events, k_height,
                                    k_width, self._stride,
                                    chan_as_cols=True)

        idx_max_patches, not_argmin = min_argmax(np.asfortranarray(surface_col),
                                                 np.asfortranarray(actfn_col))
        not_argmin = np.any(not_argmin.reshape(-1, in_channels).astype(np.bool),
                            axis=1)
        self._recompute_coords[out_y[not_argmin], out_x[not_argmin]] = True

        # Updates the indices
        ravel_indices = out_y * out_width + out_x
        channel_offsets = np.arange(0, in_channels * out_height * out_width,
                                    out_height * out_width)
        ravel_indices = (np.tile(ravel_indices.reshape(-1, 1),
                                 [1, in_channels]) + channel_offsets).reshape(
            -1)
        self._idx_max[0][ravel_indices] = idx_max_patches

        # The new events are those returned by im2col
        new_events = [out_y, out_x]

        # Resets the cached values
        self._cache_surface = None
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

        return new_events, delta_leak

    def compute_all(self, events, delta_leak=None):
        events, delta_leak = self._prev_layer.compute_all(events, delta_leak)
        return self.compute(events, delta_leak)
