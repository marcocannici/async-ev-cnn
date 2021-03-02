import os
import configargparse as argparse
from collections import OrderedDict


def layers_dict(text):
    try:
        return OrderedDict([(x.split('=')[0],
                             [int(y) for y in x.split('=')[1].split(',')]) for x in text.split(' ')])
    except:
        raise argparse.ArgumentTypeError("Format must be 'name1=h1,w1,i1,o1 name2=h2,w2,12,02 name3=i3,o3"
                                         " name4=i4,o4 ...'")


def boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def config():
    parser = argparse.ArgumentParser()

    parser.add('-c', '--config', required=True, is_config_file=True, help='config file path')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch of a single tower (GPU).  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--reader_threads',
        type=int,
        default=4,
        help='The number of parallel threads to be used by the reader.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '../data/nmnist'),
        help='Directory where the dataset is located.'
    )
    parser.add_argument(
        '--file_format',
        type=str,
        default='n-data',
        help='The file format of the dataset: \'n-data\' or \'aer-data\'.'
    )
    parser.add_argument(
        '--restore_net',
        type=str,
        default=None,
        help='This option allows you to restore the network from a previous state. You can either provide the folder'
             'containing the checkpoints, in this case the last one will be used, or a specific checkpoint file.'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='YoloEventNumpy',
        help="The implementation to be used, either 'YoloEventNumpy', 'YoloFrameNumpy' or 'YoloFrameTf'"
    )
    parser.add_argument(
        '--frame_h',
        type=int,
        default=124,
        help="The height size of the YOLO input frames"
    )
    parser.add_argument(
        '--frame_w',
        type=int,
        default=124,
        help="The width size of the YOLO input frames"
    )
    parser.add_argument(
        '--example_h',
        type=int,
        default=124,
        help="The height size of dataset examples"
    )
    parser.add_argument(
        '--example_w',
        type=int,
        default=124,
        help="The width size of dataset examples"
    )
    parser.add_argument(
        '--leak',
        type=float,
        default=0.00015,
        help="The leak to be applied."
    )
    parser.add_argument(
        '--frame_delay',
        type=int,
        default=50,
        help='Delay to wait when showing frames'
    )
    parser.add_argument(
        '--yolo_cnn_layers',
        type=layers_dict,
        default=None,
        help="A list of lists specifying the dimensions of the layers. The format is 'name1=h1,w1,i1,o1"
             "name2=h2,w2,i2,o2 name3=i3,o3 name4=i4,o4 where ' ' separates different layers and ',' separates"
             " dimensions of the same layer. Spaces are allowed only as layers' separators."
    )
    parser.add_argument(
        '--yolo_cnn_padding',
        type=str,
        default='VALID',
        help="The padding mode to be used for CNN layers."
    )
    parser.add_argument(
        '--yolo_num_cells_h',
        type=int,
        default=4,
        help="The numbed of cells on height edge."
    )
    parser.add_argument(
        '--yolo_num_cells_w',
        type=int,
        default=4,
        help="The numbed of cells on width edge."
    )
    parser.add_argument(
        '--yolo_num_bbox',
        type=int,
        default=2,
        help="The number of bounding boxes for each cell in the grid",
    )
    parser.add_argument(
        '--batch_event_size',
        type=int,
        default=1,
        help="How many events inside each batch"
    )
    parser.add_argument(
        '--batch_event_usec',
        type=int,
        default=None,
        help="The temporal length of each batch of events. If provided, it overrides batch_event_size"
    )

    args, unparsed = parser.parse_known_args()
    return args
