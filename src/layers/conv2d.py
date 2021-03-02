import numpy as np
#from Cython.Includes import numpy as np

from src.layers.layer import Layer
from src.libs.cutils import im2col_event
from src.layers.functional import im2col


class Conv2DLayer(Layer):
    """
    Performs the 2D convolution incrementally, recomputing the convolution only around the input events.
    This layer applies the leaky rectified linear activation function.
    """

    def __init__(self, prev_layer, kernel, bias, stride, alpha, padding='VALID'):
        """
        :param prev_layer: The previous Layer object in the network
        :param kernel: The kernel to be used as a float32 ndarray of shape [k_height, k_width, in_chan, out_chan]
        :param bias: The bias as a float32 ndarray of shape [out_chan]
        :param stride: The stride
        :param alpha: The alpha values to be used in the leaky rectified linear activation function.
        """

        self._prev_layer = prev_layer
        # Saves the kernel in continuous memory as [out_channels, in_channels, k_height, k_width]
        self._kernel = np.ascontiguousarray(kernel.transpose([3, 2, 0, 1]))
        self._bias = bias
        self._stride = stride
        self._alpha = alpha
        self._padding = padding

        in_channels, in_height, in_width = prev_layer.out_shape()
        out_channels, _, k_height, k_width = self._kernel.shape
        if padding == 'VALID':
            out_height = (np.floor((in_height - k_height) / stride) + 1).astype(np.int32)
            out_width = (np.floor((in_width - k_width) / stride) + 1).astype(np.int32)
            self._pad = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        elif padding == 'SAME':
            out_height = np.ceil(in_height / stride).astype(np.int32)
            out_width = np.ceil(in_width / stride).astype(np.int32)

            if in_height % stride == 0:
                pad_along_height = max(k_height - stride, 0)
            else:
                pad_along_height = max(k_height - (in_height % stride), 0)
            if in_width % stride == 0:
                pad_along_width = max(k_width - stride, 0)
            else:
                pad_along_width = max(k_width - (in_width % stride), 0)

            pad_top = pad_along_height // 2
            pad_left = pad_along_width // 2
            self._pad = {'top': pad_top, 'bottom': pad_along_height - pad_top,
                         'left': pad_left, 'right': pad_along_width - pad_left}
        else:
            raise ValueError("'padding' must be either 'SAME' or 'VALID', but %s has been provided." % padding)

        self._out_shape = [out_channels, out_height, out_width]
        self._init_fm = conv2d(self.pad_featuremap(prev_layer.surface() * prev_layer.layer_actfn()),
                               self._kernel, bias, stride).astype(np.float32)
        self._featuremap = self._init_fm.copy()
        self._init_conv_actfn = np.zeros(self._out_shape, np.float32)
        self._conv_actfn = self._init_conv_actfn.copy()

        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

    def pad_featuremap(self, featuremap):
        if self._pad['top'] > 0 or self._pad['bottom'] > 0:
            return np.pad(featuremap, ((0, 0), (self._pad['top'], self._pad['bottom']),
                                       (self._pad['left'], self._pad['right'])), mode='constant')
        return featuremap

    def pad_events(self, events):
        if self._pad['top'] > 0 or self._pad['bottom'] > 0:
            y, x = events
            return y + self._pad['top'], x + self._pad['left']
        return events

    def surface(self):
        return self._featuremap

    def layer_actfn(self):

        if self._cache_layer_actfn is None:
            positives = (self._featuremap > 0).astype(np.float32)
            self._cache_layer_actfn = positives + (1 - positives) * self._alpha
        return self._cache_layer_actfn

    def conv_actfn(self):

        if self._cache_conv_actfn is None:
            self._cache_conv_actfn = self._conv_actfn * self.layer_actfn()
        return self._cache_conv_actfn

    def out_shape(self):
        return self._out_shape

    def reset(self):
        self._featuremap = self._init_fm.copy()
        self._conv_actfn = self._init_conv_actfn.copy()
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

    def compute(self, events, delta_leak):

        # Retrieves variables and shapes from previous layer
        prevlayer_featuremap = self.pad_featuremap(self._prev_layer.featuremap())
        prevlayer_actfn = self.pad_featuremap(self._prev_layer.conv_actfn())
        events = self.pad_events(events)
        prevlayer_shape = self._prev_layer.out_shape()

        before_sign = self._featuremap >= 0
        # Applies the leak update to the previous feature map.
        self._featuremap -= self._conv_actfn * delta_leak

        # Updates the feature map around the events
        conv_cols, (out_y, out_x) = conv2d_event(prevlayer_featuremap, events, self._kernel, self._bias,
                                                 stride=self._stride)
        self._featuremap[:, out_y, out_x] = conv_cols
        conv_cols, (out_y, out_x) = conv2d_event(prevlayer_actfn, events, self._kernel,
                                                 stride=self._stride)
        self._conv_actfn[:, out_y, out_x] = conv_cols
        new_events_mask = np.zeros(self.out_shape()[1:], dtype=np.bool)  # [out_h, out_w]

        after_sign = self._featuremap >= 0
        # Boolean mask with True values in correspondence of changed signs
        changed_sign_mask = np.any(np.not_equal(before_sign, after_sign), axis=0)
        # Adds the input events
        changed_sign_mask[out_y, out_x] = True
        new_events = np.where(changed_sign_mask)

        # Resets cached values
        self._cache_layer_actfn = None
        self._cache_conv_actfn = None

        return new_events, delta_leak

    def compute_all(self, events, delta_leak=None):
        events, delta_leak = self._prev_layer.compute_all(events, delta_leak)
        return self.compute(events, delta_leak)


def conv2d_event(image, events, kernel, bias=None, padding='VALID', stride=1):
    """
    Computes the convolution of the receptive fields around the provided events.

    :param image: The image as a [batch, in_channel, height, width] numpy ndarray
    :param events: The events around which the convolution must be applied.
    :param kernel: The kernel to be convolved as a [out_channel, in_channel, k_height, k_width] ndarray
    :param bias: (optional) If provided, the bias will be added after the convolution operation.
    :param padding: (optional) A string representing the padding to be applied. 'VALID' is the only one currently
        supported.
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride.
    :return: - out_conv: The result of the convolution around the events as a flat array of shape [out_channels, num_rf]
                Each column is the result of the convolution
                over one of the receptive fields affected by one of the events. Its location is specified by the other
                return value of this method.
             - (out_y, out_x): The coordinates where each value in the 'conv' output has to be placed. You can place
                the obtained values in a featuremap with: featuremap[:, out_y, out_x] = out_conv
    """

    image = np.ascontiguousarray(image)

    out_channel, _, k_height, k_width = kernel.shape
    y_events, x_events = events
    # Makes sure that all the types are correct
    y_events = y_events.astype(np.int32)
    x_events = x_events.astype(np.int32)
    image = image.astype(np.float32)

    img_cols, out_events = im2col_event(image, y_events, x_events, k_height, k_width, stride)
    kernel_rows = kernel.reshape(out_channel, -1)

    conv = kernel_rows.dot(img_cols)
    if bias is not None:
        bias_rows = bias.reshape(out_channel, 1)
        conv = conv + bias_rows
    conv = conv.reshape(out_channel, -1)

    return conv, out_events


def conv2d(image, kernel, bias=None, padding='VALID', stride=1):
    """
    Computes the convolution of the kernel on the provided image.

    :param image: The image as a [batch, in_channel, height, width] (or [in_channel, height, width]) numpy ndarray
    :param kernel: The kernel to be convolved as a [out_channel, in_channel, k_height, k_width] ndarray
    :param bias: (optional) If provided, the bias will be added after the convolution operation.
    :param padding: (optional) A string representing the padding to be applied. 'VALID' is the only one currently
        supported.
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride.
    :return: The convolved image as a numpy's ndarray of shape [batch, out_channel, out_h, out_w]
    """

    image = np.ascontiguousarray(image)

    batch, img_channel, img_height, img_width = image.shape if image.ndim == 4 else [1, *image.shape]
    out_channel, _, k_height, k_width = kernel.shape

    if padding == 'SAME':
        if img_height % stride == 0:
            pad_along_height = max(k_height - stride, 0)
        else:
            pad_along_height = max(k_height - (img_height % stride), 0)
        if img_width % stride == 0:
            pad_along_width = max(k_width - stride, 0)
        else:
            pad_along_width = max(k_width - (img_width % stride), 0)

        pad_top = pad_along_height // 2
        pad_left = pad_along_width // 2
        image = np.pad(image, ((0, 0), (pad_top, pad_along_height - pad_top), (pad_left, pad_along_width - pad_left)),
                       mode='constant')

    img_cols, (out_h, out_w) = im2col(image, kernel.shape, stride)
    kernel_rows = kernel.reshape(out_channel, -1)

    conv = kernel_rows.dot(img_cols)
    if bias is not None:
        bias_rows = bias.reshape(out_channel, 1)
        conv = conv + bias_rows

    if image.ndim == 4:
        conv = conv.reshape(out_channel, batch, out_h, out_w).transpose(1, 0, 2, 3)
    else:
        conv = conv.reshape(out_channel, out_h, out_w)

    return conv