import numpy as np


def im2col(image, ksize, stride=1):
    """
    Rearranges the input's image into columns based on the receptive field define by ksize and stride

    :param image: A numpy's ndarray of shape [batch, in_channel, height, width] or [in_channel, height, width]
    :param ksize: A python's int32 list containing the shape of the kernel: [out_channel, in_channel, k_height, k_width]
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride
    :return: - img_cols: A numpy's ndarray with shape [img_channel * k_height * k_width, batch * out_h * out_w], where
                out_h and out_w depend of the provided ksize and stride
             - (out_h, out_w): The shape of the output feature map.
    """

    batch, img_channel, img_height, img_width = image.shape if image.ndim == 4 else [1, *image.shape]
    out_channel, _, k_height, k_width = ksize

    out_h = (img_height - k_height) // stride + 1
    out_w = (img_width - k_width) // stride + 1

    if image.ndim == 4:
        shape = (img_channel, k_height, k_width, batch, out_h, out_w)
        strides = (img_height * img_width, img_width, 1, img_channel * img_height * img_width, stride * img_width, stride)
    else:
        shape = (img_channel, k_height, k_width, out_h, out_w)
        strides = (img_height * img_width, img_width, 1, stride * img_width, stride)

    strides = image.itemsize * np.array(strides)
    img_stride = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    img_cols = np.ascontiguousarray(img_stride)
    img_cols.shape = (img_channel * k_height * k_width, batch * out_h * out_w)

    return img_cols, [out_h, out_w]


def make_leaky_rectified_actfn(alpha):
    """
    Returns the leaky rectified linear activation with the specified alpha as a function of one argument, the input.
    :param alpha: The alpha value to be used by the activation function
    :return: A method that takes one input (a numpy's array) and returns the array resulting from the application
        of the activation function.
    """
    def actfn(input):
        return np.maximum(input, input * alpha)

    return actfn


def flatten(input, batch_axis=None):
    """
    An operations that flats the input while maintaining the batch dimension.

    :param input: The input to be flatten
    :param batch_axis: The axis containing the batch dimension. If None, all the input will be flatten.
    :return: A numpy array on shape: - [batch, features...] if batch_axis is not None, [features...] otherwise
    """

    in_shape = list(input.shape)

    if batch_axis is not None:
        # Move the batch axis in the first position
        in_shape.insert(0, in_shape.pop(batch_axis))
        batch_first = input.transpose(in_shape)
        # Flatten the other dimensions
        flat_input = batch_first.reshape([in_shape[0], -1])
    else:
        flat_input = input.reshape(-1)

    return flat_input


def fully_connected(input, weight, bias, activation_fn=make_leaky_rectified_actfn(0.1)):
    """
    A fully connected layer.

    :param input: a numpy array of shape [batch, in_channels] or [in_channels]
    :param weight: The weight parameters of the layer as a [in_channels, out_channels] array
    :param bias: The bias parameter as a [out_channels] array
    :param activation_fn: The activation function to be applied, or None if no function must be applied. The function
        must have the following signature (ndarray) -> ndarray
    :return: The output resulting from the application of the fully connected layer. A [batch, out_channels] or
        [out_channels] ndarray depending on the provided input array.
    """

    fc = np.matmul(input, weight) + bias

    if activation_fn is not None:
        fc = activation_fn(fc)
    return fc