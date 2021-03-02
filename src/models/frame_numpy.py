import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from src.layers.conv2d import conv2d
from src.layers.functional import im2col, make_leaky_rectified_actfn, flatten, fully_connected


class YoloFrameNumpy:
    def __init__(self, h_frame, w_frame, num_classes, cnn_layers,
                 cnn_padding, h_cells, w_cells, num_bbox,
                 alpha, leak, checkpoint, sess):

        self._h_frame = h_frame
        self._w_frame = w_frame
        self._num_classes = num_classes
        self._cnn_layers = cnn_layers
        self._padding = cnn_padding
        self._weights = {}

        self._h_cells = h_cells
        self._w_cells = w_cells
        self._num_bbox = num_bbox
        self._alpha = alpha
        self._leak = leak

        self._sess = sess
        self._checkpoint = checkpoint
        self.restore(checkpoint)

    def restore(self, checkpoint_path, restrict_vars=None):
        layers_dict = {}

        if os.path.isdir(checkpoint_path):
            restore_checkpoint = tf.train.latest_checkpoint(checkpoint_path, latest_filename=None)
        else:
            restore_checkpoint = checkpoint_path
        ckpt_reader = checkpoint_utils.load_checkpoint(restore_checkpoint)

        var_to_shape_map = ckpt_reader.get_variable_to_shape_map()
        ckpt_vars = {key: ckpt_reader.get_tensor(key) for key in var_to_shape_map}
        layers_dict.update(ckpt_vars)

        if restrict_vars:
            for key, value in layers_dict.items():
                if key not in restrict_vars:
                    del layers_dict[key]
        self._weights.update(layers_dict)

    def maxpool(self, input, ksize, stride):
        in_channels, in_height, in_width = input.shape
        k_height, k_width = ksize

        input = input.reshape(in_channels, 1, in_height, in_width)
        frame_cols, (out_h, out_w) = im2col(input, [in_channels, in_channels, k_height, k_width], stride)
        argmax_idx = np.argmax(frame_cols, axis=0)
        frame_pool = frame_cols[argmax_idx, np.arange(in_channels * out_h * out_w, dtype=np.int32)]
        frame_pool = frame_pool.reshape(in_channels, out_h, out_w)

        return frame_pool

    def cnn(self, input, actfn):
        shapes = self._cnn_layers
        weights = self._weights

        prev_layer = input
        transposed = False
        for name, size in shapes.items():

            if 'fc' in name:
                prev_layer = fully_connected(prev_layer, weights['w_'+name], weights['b_'+name],
                                             activation_fn=actfn)
            elif 'conv' in name:
                prev_layer = conv2d(prev_layer, weights['w_'+name], weights['b_'+name], self._padding)
                prev_layer = actfn(prev_layer)
            elif 'pool' in name:
                prev_layer = self.maxpool(prev_layer, size, size[0])
                prev_layer = actfn(prev_layer)
            elif 'flatten' in name:
                transposed = True
                prev_layer = prev_layer.transpose(1, 2, 0)  # [height, width, channel]
                prev_layer = flatten(prev_layer)
                prev_layer = actfn(prev_layer)

        if not transposed:
            prev_layer = prev_layer.transpose(1, 2, 0)

        return prev_layer

    def build_graph(self, _):

        leaky_actfn = make_leaky_rectified_actfn(self._alpha)

        weights = self._weights
        for k, v in weights.items():
            if 'w_conv' in k:
                weights[k] = np.ascontiguousarray(weights[k].transpose([3, 2, 0, 1]))

        def graph(frame):
            frame = np.expand_dims(frame, axis=0)
            last = self.cnn(frame, leaky_actfn)
            output = np.reshape(last, [self._h_cells, self._w_cells, self._num_classes + self._num_bbox * 5])

            return output

        return graph
