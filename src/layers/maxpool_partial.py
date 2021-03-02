import numpy as np

from src.layers.functional import im2col
from src.layers.layer import Layer


class MaxPoolLayerPartiallyEvent(Layer):
    """
    Perform max pooling generating events whenever the maximum value in a region changes.
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
        self._init_idx_max = [np.zeros(in_channels * out_height * out_width, dtype=np.int32),
                              np.arange(in_channels * out_height * out_width, dtype=np.int32)]
        self._idx_max = [self._init_idx_max[0].copy(), self._init_idx_max[1].copy()]

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
            self._cache_surface = surface_col[self._idx_max].reshape(self._out_shape)

        return self._cache_surface

    def layer_actfn(self):

        if self._cache_layer_actfn is None:
            in_channels, in_height, in_width = self._prev_layer.out_shape()
            prevlayer_actfn = self._prev_layer.layer_actfn()

            # Makes the channels the batch dimension
            prevlayer_actfn = prevlayer_actfn.reshape(in_channels, 1, in_height, in_width)
            actfn_col, _ = im2col(prevlayer_actfn, [in_channels, in_channels, *self._ksize], self._stride)
            self._cache_layer_actfn = actfn_col[self._idx_max].reshape(self._out_shape)

        return self._cache_layer_actfn

    def conv_actfn(self):

        if self._cache_conv_actfn is None:
            in_channels, in_height, in_width = self._prev_layer.out_shape()
            prevlayer_conv_actfn = self._prev_layer.conv_actfn()

            # Makes the channels the batch dimension
            prevlayer_conv_actfn = prevlayer_conv_actfn.reshape(in_channels, 1, in_height, in_width)
            conv_actfn_col, _ = im2col(prevlayer_conv_actfn, [in_channels, in_channels, *self._ksize], self._stride)
            self._cache_conv_actfn = conv_actfn_col[self._idx_max].reshape(self._out_shape)

        return self._cache_conv_actfn

    def out_shape(self):
        return self._out_shape

    def reset(self):
        self._idx_max = [self._init_idx_max[0].copy(), self._init_idx_max[1].copy()]
        self._cache_surface = None
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

    def compute(self, events, delta_leak):

        # Retrieves variables and shapes from previous layer
        prevlayer_surface = self._prev_layer.surface()
        prevlayer_shape = self._prev_layer.out_shape()

        in_channels, in_height, in_width = prevlayer_shape
        k_height, k_width = self._ksize

        # Makes the channels the batch dimension
        prevlayer_surface = prevlayer_surface.reshape(in_channels, 1, in_height, in_width)
        # surface_col: [k_height * k_width, in_channels * out_h * out_w]
        surface_col, _ = im2col(prevlayer_surface, [in_channels, in_channels, k_height, k_width], self._stride)
        idx_max_patches = np.argmax(surface_col, axis=0)
        changed_idx_mask = np.not_equal(idx_max_patches, self._idx_max[0]).reshape(self._out_shape)
        changed_idx_mask = np.any(changed_idx_mask, axis=0)

        # Input events are always forwarded
        y, x = events
        changed_idx_mask[y // self._stride, x // self._stride] = True
        new_events = np.where(changed_idx_mask)

        # Updates the indices
        self._idx_max[0] = idx_max_patches

        # Resets the cached values
        self._cache_surface = None
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

        return new_events, delta_leak

    def compute_all(self, events, delta_leak=None):
        events, delta_leak = self._prev_layer.compute_all(events, delta_leak)
        return self.compute(events, delta_leak)