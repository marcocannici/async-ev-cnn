import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np

from src.scripts.config import config
from src.readers import detection_reader
from src.models.frame_tf import YoloFrameTf
from src.models.frame_numpy import YoloFrameNumpy
from src.models.event_numpy import YoloEventNumpy
from src.libs.runner import NumpyEventRunner, NumpyFrameRunner, TfFrameRunner


if __name__ == "__main__":

    # Read configs
    # ============
    args = config()

    # Create Tf session
    # =================
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=sess_config)

    # Create reader
    # =============
    reader = detection_reader.factory(args.input_data_dir, file_format=args.file_format)

    # Build network and initialize from checkpoint
    # ============================================
    frame = tf.placeholder(tf.float32, [args.frame_h, args.frame_w], name="frame")
    network_class = locals()[args.network]
    network = network_class(args.frame_h, args.frame_w, reader.num_classes(),
                            args.yolo_cnn_layers, args.yolo_cnn_padding,
                            args.yolo_num_cells_h, args.yolo_num_cells_w,
                            args.yolo_num_bbox, 0.1, args.leak, args.restore_net, sess)
    network_graph = network.build_graph(frame)

    # Initializes the remaining variables
    # ===================================
    uninit_vars = [bs.decode("utf-8") for bs in sess.run(tf.report_uninitialized_variables(tf.global_variables()))]
    var_list = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for name in uninit_vars:
            var_list.append(tf.get_variable(name, dtype=sess.graph.get_tensor_by_name(name + ":0").dtype))
    sess.run(tf.variables_initializer(var_list))

    # Run the network
    # ===============
    if args.network == "YoloFrameNumpy":
        runner_class = NumpyFrameRunner
    elif args.network == "YoloEventNumpy":
        runner_class = NumpyEventRunner
    elif args.network == "YoloFrameTf":
        runner_class = TfFrameRunner

    runner = runner_class(args, reader)
    runner.run(network_graph, sess, frame)
