import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.libs.cutils import *
from src.libs.viz import integrate_frame
from src.layers.conv2d import Conv2DLayer
from src.layers.integration import IntegrationLayer
from src.layers.maxpool import MaxPoolLayer

from collections import OrderedDict


class TestNetworkEvents:

    def __init__(self, frame_shape, weights, leak=0.1, alpha_relu=0.1, padding="SAME"):
        self.leak = leak
        self.alpha_relu = alpha_relu

        self.weights = weights
        self.padding = padding
        self.frame_height, self.frame_width = frame_shape

        self.intgr = IntegrationLayer(leak, self.frame_height, self.frame_width)
        self.conv1 = Conv2DLayer(self.intgr, weights['conv1_k'], weights['conv1_b'], 1, alpha_relu, padding)
        self.pool1 = MaxPoolLayer(self.conv1, [2, 2], 2)
        self.conv2 = Conv2DLayer(self.pool1, weights['conv2_k'], weights['conv2_b'], 1, alpha_relu, padding)
        self.pool2 = MaxPoolLayer(self.conv2, [2, 2], 2)

    def forward(self, events):

        intgr_ev, delta_leak = self.intgr.compute(events, None)
        conv1_ev, delta_leak = self.conv1.compute(intgr_ev, delta_leak)
        pool1_ev, delta_leak = self.pool1.compute(conv1_ev, delta_leak)
        conv2_ev, delta_leak = self.conv2.compute(pool1_ev, delta_leak)
        pool2_ev, delta_leak = self.pool2.compute(conv2_ev, delta_leak)

        retvals = OrderedDict([
            ("intgr", {"fm": self.intgr.featuremap().transpose(1, 2, 0),
                       "out_events": np.stack(intgr_ev, axis=-1)}),
            ("conv1", {"fm": self.conv1.featuremap().transpose(1, 2, 0),
                       "actfn": self.conv1.conv_actfn().transpose(1, 2, 0),
                       "out_events": np.stack(conv1_ev, axis=-1)}),
            ("pool1", {"fm": self.pool1.featuremap().transpose(1, 2, 0),
                       "actfn": self.pool1.conv_actfn().transpose(1, 2, 0),
                       "out_events": np.stack(pool1_ev, axis=-1)}),
            ("conv2", {"fm": self.conv2.featuremap().transpose(1, 2, 0),
                       "actfn": self.conv2.conv_actfn().transpose(1, 2, 0),
                       "out_events": np.stack(conv2_ev, axis=-1)}),
            ("pool2", {"fm": self.pool2.featuremap().transpose(1, 2, 0),
                       "actfn": self.pool2.conv_actfn().transpose(1, 2, 0),
                       "out_events": np.stack(pool2_ev, axis=-1)})
        ])

        return retvals


class TestNetworkFrames:

    def __init__(self, frame_shape, weights, leak=0.1, alpha_relu=0.1, padding="SAME", ):
        self.leak = leak
        self.alpha_relu = alpha_relu

        self.weights = weights
        self.padding = padding
        self.frame_height, self.frame_width = frame_shape

    def build_graph(self, frames):

        tf_conv1 = tf.nn.conv2d(frames, self.weights['conv1_k'],
                                strides=[1, 1, 1, 1], padding=self.padding) + self.weights['conv1_b']
        tf_conv1 = tf.maximum(tf_conv1, self.alpha_relu * tf_conv1)
        tf_pool1 = tf.nn.max_pool(tf_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        tf_conv2 = tf.nn.conv2d(tf_pool1, self.weights['conv2_k'],
                                strides=[1, 1, 1, 1], padding=self.padding) + self.weights['conv2_b']
        tf_conv2 = tf.maximum(tf_conv2, self.alpha_relu * tf_conv2)
        tf_pool2 = tf.nn.max_pool(tf_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)

        retvals = OrderedDict([
            ("frame", frames),
            ("conv1", tf_conv1),
            ("pool1", tf_pool1),
            ("conv2", tf_conv2),
            ("pool2", tf_pool2)])

        return retvals


def test(max_iterations=1000, verbose=True):

    np.set_printoptions(precision=5, linewidth=200)

    padding = "SAME"
    leak = 0.1
    alpha = 0.1
    frame_height, frame_width = 8, 8
    b1,  b2 = 10, 10
    k1 = k2 = np.array([[-2, -1, 1],
                        [-2, -1, 1],
                        [-2, -1, 1]]).reshape([3, 3, 1, 1])
    weights = {"conv1_k": k1, "conv1_b": np.array([b1]),
               "conv2_k": k2, "conv2_b": np.array([b2])}

    # Event based net
    # ===============
    net_events = TestNetworkEvents(frame_shape=(frame_height, frame_width),
                                   weights=weights, leak=leak,
                                   alpha_relu=alpha, padding=padding)

    # Frame based net
    # ===============
    frames = tf.placeholder(shape=[1, frame_height, frame_width, 1], dtype=tf.float32, name="frame")
    net_frames = TestNetworkFrames(frame_shape=(frame_height, frame_width),
                                   weights=weights, leak=leak,
                                   alpha_relu=alpha, padding=padding)
    graph_frames = net_frames.build_graph(frames)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ts = np.array([0], dtype=np.int32)
    y = np.random.randint(0, frame_height, size=1)
    x = np.random.randint(0, frame_width, size=1)
    prev_frame_state = None

    for i in tqdm(range(max_iterations)):
        events = np.stack([y, x, ts], axis=-1).astype(np.int32)
        frame, prev_ts = integrate_frame(events, leak, frame_height, frame_width, prev_frame_state)
        b_frame = frame.reshape([1, frame_height, frame_width, 1])

        ev_results = net_events.forward(events)
        tf_results = sess.run(graph_frames, feed_dict={frames: b_frame})

        close = np.allclose(ev_results["conv1"]["fm"], tf_results["conv1"][0]) and \
                np.allclose(ev_results["pool1"]["fm"], tf_results["pool1"][0]) and \
                np.allclose(ev_results["conv2"]["fm"], tf_results["conv2"][0]) and \
                np.allclose(ev_results["pool2"]["fm"], tf_results["pool2"][0])

        if verbose or not close:
            print("\n")
            print("event: {}, ts: {}".format([y, x], ts))

            for layer_name, layer_res in ev_results.items():
                print("{} output:".format(layer_name))
                print(np.squeeze(ev_results[layer_name]["fm"]))
                if "actfn" in ev_results[layer_name]:
                    print("{} actfn:".format(layer_name))
                    print(np.squeeze(ev_results[layer_name]["actfn"]))
                print("{} events:".format(layer_name))
                print([list(e) for e in ev_results[layer_name]["out_events"]])

            print("\n")
            for layer_name, layer_res in tf_results.items():
                print("{} output:".format(layer_name))
                print(np.squeeze(layer_res))

        if not close:
            print("\n ERROR: Difference found in last iteration")
            exit()

        # Generate ew events
        num_events = 5  # np.random.randint(1, 5)
        ts = np.sort(np.random.randint(1, 10, size=(num_events))) + prev_ts
        y = np.random.randint(0, frame_height, size=(num_events))
        x = np.random.randint(0, frame_width, size=(num_events))
        prev_frame_state = [frame, prev_ts]

    print("\nSUCCESS: No difference found between event-based and tensorflow results!")


if __name__ == "__main__":
    test(max_iterations=10000, verbose=False)
