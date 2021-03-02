import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

from src.layers.conv2d import Conv2DLayer
from src.layers.integration import IntegrationLayer
from src.layers.maxpool import MaxPoolLayer
from src.layers.functional import make_leaky_rectified_actfn, flatten, fully_connected


class YoloEventNumpy:
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

    def build_cnn_layers(self):
        shapes = self._cnn_layers
        weights = self._weights

        prev_layer = IntegrationLayer(self._leak, self._h_frame, self._w_frame)
        event_layers = [prev_layer]
        non_event_layers = []

        for name, size in shapes.items():

            if 'conv' in name:
                prev_layer = Conv2DLayer(prev_layer, weights['w_'+name], weights['b_'+name], 1,
                                         self._alpha, self._padding)
                event_layers.append(prev_layer)
            elif 'pool' in name:
                prev_layer = MaxPoolLayer(prev_layer, size, size[0])
                event_layers.append(prev_layer)
            else:
                non_event_layers.append((name, size))

        return event_layers, non_event_layers

    def cnn(self, events, last_event_layers, non_event_layers, actfn):
        layers = self._cnn_layers
        delta_leak = None
        last_event_layers.compute_all(events, delta_leak)
        prev_layer = last_event_layers.featuremap().transpose(1, 2, 0)

        for name, size in non_event_layers:
            if 'fc' in name:
                prev_layer = fully_connected(prev_layer, layers['w_'+name], layers['b_'+name],
                                             activation_fn=actfn)
            elif 'flatten' in name:
                prev_layer = flatten(prev_layer)

        return prev_layer

    def build_graph(self, _):
        event_layers, non_event_layers = self.build_cnn_layers()
        leaky_actfn = make_leaky_rectified_actfn(self._alpha)

        def graph(input, reset):

            if reset:
                for layer in event_layers:
                    layer.reset()

            last = self.cnn(input, event_layers[-1], non_event_layers, leaky_actfn)
            output = np.reshape(last, [self._h_cells, self._w_cells, self._num_classes + self._num_bbox * 5])

            return output

        return graph
