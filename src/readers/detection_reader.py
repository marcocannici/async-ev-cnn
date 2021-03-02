import numpy as np
import tensorflow as tf
import glob
import os
import re

from multiprocessing import Value
from src.readers.file_reader import NReader, AerReader
from src.readers.event_reader import EventReader
from src.readers.file_reader import FileReader


class FileAnnotationsReader(FileReader):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def read_annotation(self, filename):
        bboxes = np.load(filename)
        return bboxes

    def read_example_and_annotation(self, events_filename):
        filename, ext = os.path.splitext(events_filename)
        bboxes_filename = os.path.join(os.path.dirname(filename), "annotations", os.path.basename(filename) + ".npy")

        l, x, y, ts, p = self.read_example(events_filename)
        bboxes = self.read_annotation(bboxes_filename)

        return l, x, y, ts, p, bboxes


class DetectionReader(EventReader, FileAnnotationsReader):
    """
    A modified version of the reader that considers labels as features allowing padding. The main difference is
    the fact that next_batch_** functions will return [length, labels, features...] instead of
    [labels, length, features...], and the fact that preprocessing function have (length, label, x, y, ts, p) arguments
    """

    def _init(self, path, validation_size=None, test_size=None, tmp_dir=None, seed=1234):

        # Sets the seed
        np.random.seed(seed)
        self._seed = seed
        self._path = path
        self._tmp_dir = tmp_dir

        # Loads the filenames of the whole dataset
        test_filenames = glob.glob(os.path.join(path, 'test', '*.*'))
        train_filenames = glob.glob(os.path.join(path, 'train', '*.*'))
        validation_filenames = glob.glob(os.path.join(path, 'validation', '*.*'))
        params_filename = os.path.join(path, 'params.npz')
        if len(test_filenames) == 0 or len(train_filenames) == 0 or len(validation_filenames) == 0 or \
                not os.path.exists(params_filename):
            raise Exception("The provided path does not contain any data or the directory structure is not valid.")

        params_values= np.load(params_filename)
        num_classes = np.asscalar(params_values['num_classes'])
        # Computes a mapping between directory names and integer labels
        self._labels = list(params_values['label_to_idx'])
        self._dir_to_label = dict(zip(params_values['label_to_idx'], np.arange(num_classes)))

        # Shuffles the filenames in-place
        np.random.shuffle(train_filenames)
        np.random.shuffle(validation_filenames)
        np.random.shuffle(test_filenames)

        self._train_filenames = train_filenames
        self._validation_filenames = validation_filenames
        self._test_filenames = test_filenames

        # Variables used to keep track of the current position in the dataset
        self._train_pos_shared = Value('i', 0)
        self._validation_pos_shared = Value('i', 0)
        self._test_pos_shared = Value('i', 0)

        self._train_queue_workers = []
        self._validation_queue_workers = []
        self._test_queue_workers = []
        self._train_workers_started = np.array([False], dtype=np.bool)
        self._validation_workers_started = np.array([False], dtype=np.bool)
        self._test_workers_started = np.array([False], dtype=np.bool)
        self._train_queue = None
        self._validation_queue = None
        self._test_queue = None

        self._train_size = len(self._train_filenames)
        self._validation_size = len(self._validation_filenames)
        self._test_size = len(self._test_filenames)

    def feature_to_pad_size(self, feature_list, axis):

        feature_to_size = lambda f: None if f.ndim == 1 else f.shape[axis]
        size_to_max = lambda list: None if list[0] is None else max(list)

        lengths = []
        for feature in feature_list:
            sizes = list(map(feature_to_size, feature))
            max_length = size_to_max(sizes)
            lengths.append(max_length)

        return lengths

    def next_batch_single_thread(self, batch_size, dataset, preprocessing_fn=None, features_to_pad_mask=None,
                                 concat_features=False, multiple_examples=1, cache_preprocessed=False):

        lengths, feature_list = [], []
        mult_filenames = [self._get_next_filenames(batch_size, dataset) for _ in range(multiple_examples)]

        # Aggregates the examples in pairs of multiple_examples examples and reads them
        for pair_fn in zip(*mult_filenames):
            pair_values = {}
            rel_filename = os.path.relpath(pair_fn[0], start=self._path)
            cached_fn = os.path.splitext(os.path.join(self._tmp_dir, rel_filename))[0] if self._tmp_dir else None
            if preprocessing_fn is not None and cached_fn is not None and cache_preprocessed \
                    and os.path.exists(cached_fn + '.npz'):
                cached = np.load(cached_fn + '.npz')
                features = [cached['feature_%d' % k] for k in range(len(cached.keys()) - 1)]
                length = np.asscalar(cached['length'])
                cached.close()
            else:
                # Reads the selected files
                for filename in pair_fn:
                    # Extracts the events
                    length, x, y, ts, p, bboxes = self.read_example_and_annotation(filename)
                    pair_values['length'] = self._concat_list(pair_values.get('length', None), length)
                    pair_values['x'] = self._concat_list(pair_values.get('x', None), x)
                    pair_values['y'] = self._concat_list(pair_values.get('y', None), y)
                    pair_values['ts'] = self._concat_list(pair_values.get('ts', None), ts)
                    pair_values['p'] = self._concat_list(pair_values.get('p', None), p)
                    pair_values['bboxes'] = self._concat_list(pair_values.get('bboxes', None), bboxes)

                # Applies the preprocessing function if provided
                if preprocessing_fn is not None:
                    # The preprocessing function can produce a different number of features
                    r_features = preprocessing_fn(pair_values['length'], pair_values['x'], pair_values['y'],
                                                  pair_values['ts'], pair_values['p'], pair_values['bboxes'])
                    # The preprocessing function must always return the length as first value
                    length = r_features[0]
                    features = list(r_features[1:])

                    if cache_preprocessed and cached_fn is not None:
                        sorted_features = {('feature_%d' % k): f for k, f in enumerate(features)}
                        sorted_features['length'] = length
                        os.makedirs(os.path.dirname(cached_fn), exist_ok=True)
                        np.savez(cached_fn + '.npz', **sorted_features)
                else:
                    if multiple_examples > 1:
                        raise ValueError("You must specify a preprocessing function if multiple_examples > 1")

                    features = [pair_values['x'], pair_values['y'], pair_values['ts'], pair_values['p'],
                                pair_values['bboxes']]
                    length = pair_values['length']
                    if features_to_pad_mask is None:
                        features_to_pad_mask = [0, 0, 0, 0, 0]

            # Initializes the feature_list vector based on the number of features returned by preprocessing_fn
            if len(feature_list) == 0:
                feature_list = [[]] * len(features)

            for i, f in enumerate(features):
                feature_list[i] = feature_list[i] + [f]

            lengths.append(length)

        # If it is a list of scalars, adds a nesting level. In the most general case, in fact, the preprocessing
        # function can return a list of lengths as first return value that can be used to pad return values to
        # different lengths. If this is not the case, we extend it to the general case (list of lists).
        if isinstance(lengths[0], int):
            lengths = map(lambda x: [x], lengths)

        # Pairs lengths element-wise and compute their max. We need to compute the maximum of each
        # length in the list returned by the preprocessing functions
        paired_lengths = list(zip(*lengths))
        max_length_list = list(map(max, paired_lengths))
        stacked_features = []

        # Applies padding and reshape arrays
        # If the features must be concatenated, no padding will be applied, even if provided, otherwise, if
        # features must be stacked, but no padding mask is provided, all the features are padded with the first length
        # in the list
        if concat_features is True:
            features_to_pad_mask = [-1] * len(feature_list)
        elif features_to_pad_mask is None:
            features_to_pad_mask = [0] * len(feature_list)

        for pad_idx, feature in zip(features_to_pad_mask, feature_list):
            if pad_idx >= 0:
                feature = self._pad_list(feature, max_length_list[pad_idx])
            stacked_feature = np.concatenate(feature, axis=0) if concat_features else np.stack(feature)
            stacked_features.append(stacked_feature)

        lengths = list(map(np.array, paired_lengths))

        return lengths + stacked_features

    def next(self, dataset, preprocessing_fn=None, multiple_examples=1, return_key=False, cache_preprocessed=False):

        # Retrieves the next filename to be read. _get_next_filenames returns a list of filenames even if
        # the provided batch_size is 1
        mult_filenames, first_pos = self._get_next_filenames(multiple_examples, dataset, return_start_pos=True)

        pair_values = {}
        rel_filename = os.path.relpath(mult_filenames[0], start=self._path)
        cached_fn = os.path.splitext(os.path.join(self._tmp_dir, rel_filename))[0] if self._tmp_dir else None
        if preprocessing_fn is not None and cached_fn is not None and cache_preprocessed \
                and os.path.exists(cached_fn + '.npz'):
            cached = np.load(cached_fn + '.npz')
            features = [cached['feature_%d' % k] for k in range(len(cached.keys()) - 1)]
            length = np.asscalar(cached['length'])
            cached.close()
        else:
            for filename in mult_filenames:
                # Extracts the events
                length, x, y, ts, p, bboxes = self.read_example_and_annotation(filename)
                pair_values['length'] = self._concat_list(pair_values.get('length', None), length)
                pair_values['x'] = self._concat_list(pair_values.get('x', None), x)
                pair_values['y'] = self._concat_list(pair_values.get('y', None), y)
                pair_values['ts'] = self._concat_list(pair_values.get('ts', None), ts)
                pair_values['p'] = self._concat_list(pair_values.get('p', None), p)
                pair_values['bboxes'] = self._concat_list(pair_values.get('bboxes', None), bboxes)

            # Applies the preprocessing function if provided
            if preprocessing_fn is not None:
                # The preprocessing function can produce a different number of features
                r_features = preprocessing_fn(pair_values['length'], pair_values['x'], pair_values['y'], pair_values['ts'],
                                              pair_values['p'], pair_values['bboxes'])
                length = r_features[0]
                features = list(r_features[1:])

                if cache_preprocessed and cached_fn is not None:
                    sorted_features = {('feature_%d' % k): f for k, f in enumerate(features)}
                    sorted_features['length'] = length
                    os.makedirs(os.path.dirname(cached_fn), exist_ok=True)
                    np.savez(cached_fn + '.npz', **sorted_features)
            else:
                if multiple_examples > 1:
                    raise ValueError("You must specify a preprocessing function if multiple_examples > 1")

                features = [pair_values['x'], pair_values['y'], pair_values['ts'], pair_values['p'], pair_values['bboxes']]
                length = pair_values['length']

        result = [length] + features
        result += [first_pos] if return_key else []
        return result

    def next_op(self, dataset, preprocessing_fn=None, multiple_examples=1, data_types=None, return_key=True,
                cache_preprocessed=False):

        if data_types is None:
            data_types = [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]
        return super().next_op(dataset, preprocessing_fn, multiple_examples, data_types, return_key, cache_preprocessed)

    def get_dequeue_op(self, batch_size, dataset, preprocessing_fn=None, multiple_examples=1, data_types=None,
                       data_shapes=None, queue_capacity=1000, cache_preprocessed=False, threads=2):

        if data_types is None and data_shapes is None:
            data_types = [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]
            data_shapes = [[], [None], [None], [None], [None], [None, 6]]
        return super().get_dequeue_op(batch_size, dataset, preprocessing_fn, multiple_examples, data_types,
                                      data_shapes, queue_capacity, cache_preprocessed, threads)


class NDetectionReader(DetectionReader, NReader):
    """A class for reading N-MNIST and N-Caltech101 datasets"""

    def __init__(self, path, validation_size=0, tmp_dir=None, seed=1234):
        super().__init__(path, validation_size=validation_size, tmp_dir=tmp_dir, seed=seed)


class AerDetectionReader(DetectionReader, AerReader):
    """A class for reading AER datasets"""

    def __init__(self, path, validation_size=0, tmp_dir=None, seed=1234, camera='DVS128'):
        """
        Different cameras may have different data format. This argument is ignored if the dataset has been saved
        with the 3.1 format. In this case the events' format is specified in the header of each event.
        Further info: https://inilabs.com/support/software/fileformat/
        """
        super().__init__(path, validation_size=validation_size, tmp_dir=tmp_dir, seed=seed, camera=camera)


def factory(path, file_format, tmp_dir=None, validation_size=10000, seed=1234, **kargs):
    """
    This function returns the proper reader based on the provided file format
    :param path: if this argument is a file, it is interpreted as the file containing the save from where to
                restore the reader, otherwise it is the path where the dataset is located. The expected directory
                structure must have a sub-directory for each label containing all the examples of the corresponding
                label:

                    path
                    ├──0
                    |   ├──example1.bin
                    |   ├──example2.bin
                    |   └──...
                    ├──1
                    |   └──...
                    |
    :param file_format: the files' format of the dataset (values in {'n-data', 'aer-data', 'aer-data_DVS128', ...})
    :param validation_size: number of examples to be taken from the original training set in order to build
        the validation set.
    :param tmp_dir: (optional) A directory used by the reader to save temporary files. It can be used to store
            preprocessed files to be reused.
    :param seed: the random seed
    :return: A DVSReader's implementation object.
    """

    if file_format == 'n-data':
        return NDetectionReader(path=path, validation_size=validation_size, tmp_dir=tmp_dir, seed=seed)
    elif file_format.startswith('aer-data'):
        format = file_format.split('_')
        try:
            camera = format[1]
        except IndexError:
            camera = None
        return AerDetectionReader(path=path, validation_size=validation_size, tmp_dir=tmp_dir, seed=seed, camera=camera)
    else:
        raise ValueError('The provided file format ({}) is unknown'.format(file_format))


def restore(save_filename, file_format):
    """
    This method is used to restore a previously saved reader from its save
    :param save_filename: The save file from which to restore the reader
    :return: a DVSReader object
    """

    if file_format == 'n-data':
        return NDetectionReader(path=save_filename)
    elif file_format == 'aer-data':
        return AerDetectionReader(path=save_filename)
    else:
        raise ValueError('The provided file format ({}) is unknown'.format(file_format))