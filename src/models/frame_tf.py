import os
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils


def leaky_activation(tensor, leak=0.1):
    with tf.name_scope('leaky_activation'):
        return tf.maximum(tensor, leak * tensor)


class YoloFrameTf:
    def __init__(self, h_frame, w_frame, num_classes, cnn_layers,
                 cnn_padding, h_cells, w_cells, num_bbox,
                 alpha, leak, checkpoint, sess, add_last_fc=False):

        self._h_frame = h_frame
        self._w_frame = w_frame
        self._num_classes = num_classes
        self._cnn_layers = cnn_layers
        self._cnn_padding = cnn_padding
        self._add_last_fc = add_last_fc

        self._h_cells = h_cells
        self._w_cells = w_cells
        self._num_bbox = num_bbox
        self._alpha = alpha
        self._leak = leak

        self._sess = sess
        self._checkpoint = checkpoint

    def restore(self, checkpoint_path, restrict_vars=None):

        if os.path.isdir(checkpoint_path):
            restore_checkpoint = tf.train.latest_checkpoint(checkpoint_path, latest_filename=None)
        else:
            restore_checkpoint = checkpoint_path

        # Retrieves the variables inside 'restore_checkpoint'
        ckpt_vars = [name for name, shape in checkpoint_utils.list_variables(restore_checkpoint)]

        if restrict_vars and len(restrict_vars) > 0:
            restore_vars = list(set(ckpt_vars).intersection(restrict_vars))
        else:
            # If no list is provided, all the variables contained in the checkpoint will be restored
            uninit_vars = [bs.decode("utf-8") for bs in self._sess.run(tf.report_uninitialized_variables(tf.global_variables()))]
            restore_vars = list(set(ckpt_vars).intersection(uninit_vars))

        restore_variables = []
        # Retrieves the variables to be restored from their name
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for name in restore_vars:
                restore_variables.append(tf.get_variable(name, dtype=self._sess.graph.get_tensor_by_name(name+":0").dtype))

        restore_vars_saver = tf.train.Saver(var_list=restore_variables)
        restore_vars_names = map(lambda v: v.name, restore_variables)
        restore_vars_names = ''.join("%s, " % v for v in restore_vars_names)[:-2]

        print("Restoring variables ({}) from '{}'... ".format(restore_vars_names, restore_checkpoint), end="")
        restore_vars_saver.restore(self._sess, restore_checkpoint)
        print("Done.")

    def fully_connected(self, input, name, w_init=None, b_init=None, shape=None, activation_fn=None):
        fc_w = tf.get_variable(name="w_" + name,
                               initializer=tf.contrib.layers.xavier_initializer() if w_init is None else w_init,
                               shape=shape)
        fc_b = tf.get_variable(name="b_" + name,
                               initializer=tf.constant(0.1, shape=[shape[1]]) if b_init is None else b_init)
        h_fc = tf.matmul(input, fc_w) + fc_b

        if activation_fn is not None:
            h_fc = activation_fn(h_fc)
        return h_fc

    def conv2d(self, input, ksize, strides, padding, name, activation_fn=None, trainable=True):
        w_conv = tf.get_variable(name="w_" + name, shape=ksize, initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=trainable)
        b_conv = tf.get_variable(name="b_" + name, initializer=tf.constant(0.1, shape=ksize[-1:]), trainable=trainable)

        h_conv = tf.nn.conv2d(input, w_conv, strides=strides, padding=padding)

        if activation_fn is not None:
            h_conv = activation_fn(h_conv + b_conv)
        return h_conv

    def cnn(self, input, layers, actfn=tf.nn.relu, padding='VALID', keep_prob=None, trainable=True):

        prev_layer = input
        for name, size in layers:

            if 'fc' in name:
                prev_layer = self.fully_connected(prev_layer, name=name, shape=size, activation_fn=actfn)
            elif 'softmax' in name:
                prev_layer = self.fully_connected(prev_layer, name=name, shape=size, activation_fn=None)
            elif 'conv' in name:
                prev_layer = self.conv2d(prev_layer, ksize=size, strides=[1, 1, 1, 1], padding=padding,
                                         name=name, activation_fn=actfn, trainable=trainable)
            elif 'pool' in name:
                prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, *size, 1], strides=[1, *size, 1],
                                            padding=padding, name=name)
            elif 'flatten' in name:
                prev_layer = tf.contrib.layers.flatten(prev_layer)
            elif 'drop' in name:
                if keep_prob is None:
                    raise ValueError("'%s' layer requires a keep_prob parameter." % name)
                prev_layer = tf.nn.dropout(prev_layer, keep_prob=keep_prob, name=name)

        return prev_layer

    def build_graph(self, frame):

        frames_channel = tf.reshape(frame, [1, self._h_frame, self._w_frame, 1])

        layers = list(self._cnn_layers.items())
        fc_name, fc_shape = next(layer for layer in reversed(layers) if 'drop' not in layer[0])
        # Adds a layer based on the number of cells and bboxes per cell
        if self._add_last_fc:
            layers += [(fc_name[:-1] + str(int(fc_name[-1]) + 1),
                        [fc_shape[1], (self._h_cells*self._w_cells) * (self._num_classes+self._num_bbox*5)])]

        last = self.cnn(frames_channel, layers, padding=self._cnn_padding, actfn=leaky_activation, trainable=False)

        # [flat_batch_size, S, S, C + 5 * B]
        output = tf.reshape(last, [self._h_cells, self._w_cells, self._num_classes + self._num_bbox * 5])

        self.restore(self._checkpoint)
        return output
