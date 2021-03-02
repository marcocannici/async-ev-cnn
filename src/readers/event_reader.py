import os
import time
import glob
import traceback
import numpy as np
import tensorflow as tf
from multiprocessing import Lock, Queue, Process, Value
from concurrent.futures import ThreadPoolExecutor
from src.readers.file_reader import FileReader, NReader, AerReader, NumpyReader

import dill
dill.settings['recurse'] = True


class EventReader(FileReader):
    """A class for reading datasets coming from DVS sensors. This is an abstract class, that is, the actual function
    used for reading files (read_example) is not implemented because it depends on the file format. You should use
    one of the extensions of this class, or inherit from this class and overwriting the read_example method."""

    # A lock used to access self._##_pos variables
    position_lock = Lock()

    def __init__(self, path, validation_size=None, test_size=None, tmp_dir=None, seed=1234, **kargs):
        """Constructor.

        :param path: if this argument is a file, it is interpreted as the file containing the save from where to
            restore the reader, otherwise it is the path where the dataset is located. The expected directory
            structure must have a sub-directory for each label containing all the examples of the corresponding
            label:

            train
                ├── class0
                |   ├──example1.bin
                |   ├──example2.bin
                |   └──...
                ├── class1
                |   └──...
                |
            test
                └──...
            validation
                └──...

        :param tmp_dir: (optional) A directory used by the reader to save temporary files. It can be used to store
            preprocessed files to be reused. The tmp_dir can also be set afterwards by using the set_tmp_dir() method.
        :param validation_size: number of examples to be taken from the original training set in order to build
        the validation set.
        :param seed: the random seed"""

        super().__init__(**kargs)

        self._train_pos_shared = Value('i', 0)
        self._validation_pos_shared = Value('i', 0)
        self._test_pos_shared = Value('i', 0)

        if os.path.isfile(path):
            self._restore(path)
        else:
            self._init(path, validation_size, test_size, tmp_dir, seed)

        self._train_queue_workers = []
        self._validation_queue_workers = []
        self._test_queue_workers = []
        self._train_workers_started = np.array([False], dtype=np.bool)
        self._validation_workers_started = np.array([False], dtype=np.bool)
        self._test_workers_started = np.array([False], dtype=np.bool)
        self._train_queue = None
        self._validation_queue = None
        self._test_queue = None

    def train_size(self):
        return self._train_size

    def validation_size(self):
        return self._validation_size

    def test_size(self):
        return self._test_size

    def num_classes(self):
        return len(self._labels)

    def label_to_idx(self):
        return self._dir_to_label

    def set_tmp_dir(self, dir):
        self._tmp_dir = dir

    @property
    def _train_pos(self):
        return self._train_pos_shared.value

    @_train_pos.setter
    def _train_pos(self, value):
        self._train_pos_shared.value = value

    @property
    def _validation_pos(self):
        return self._validation_pos_shared.value

    @_validation_pos.setter
    def _validation_pos(self, value):
        self._validation_pos_shared.value = value

    @property
    def _test_pos(self):
        return self._test_pos_shared.value

    @_test_pos.setter
    def _test_pos(self, value):
        self._test_pos_shared.value = value

    @staticmethod
    def _pad_list(list_arrays, length, axis=0, value=0):
        """Given a list of numpy arrays, adds to them padding values at the end of the first axis so that all of them
        will have the specified length

        :param list_arrays: the list of labels to be padded
        :param length: the length that all the arrays must have after the padding has been applied
        :param value: the value to be added at the end of each array
        :param axis: the axis where to apply padding.
        :return A list containing the padded arrays"""

        n_axis = list_arrays[0].ndim
        pad_list = [(0, 0)] * (n_axis-1)
        return map(lambda x: np.pad(x, tuple(pad_list[:axis] + [(0, length - x.shape[axis])] + pad_list[axis:]),
                                    'constant', constant_values=value),
                   list_arrays)

    def _one_hot(self, label_list):
        """Given a list of labels, returns a list of arrays containing the labels in one-hot representation

        :param label_list: a list of integers representing the labels to be converted
        :return A list of arrays which are the one-hot representation of the given labels"""

        # Creates an identity matrix and selects the rows according to the labels. A label is just
        # a particular row of the matrix
        return np.eye(self.num_classes(), dtype=np.float32)[label_list]

    def _get_next_filenames(self, batch_size, dataset, return_start_pos=False):
        """
        Retrieves the next batch of filenames to be read. If there aren't enough remaining files, the remaining are
        taken from the beginning of the list. When the list finishes the position of the dataset is reset to 0 and the
        dataset is shuffled.

        :param batch_size: the size of he batch
        :param dataset: the dataset from which the filenames must be taken from.
        :param return_start_pos: a Boolean, if true the method will return also the position of the current element in
            the minibatch wrt the current data ordering.
        :return: A list of filenames of size 'batch_size' from 'dataset'
        """
        filenames, start_pos = None, None

        with self.position_lock:
            if dataset == 'train':
                start_pos = self._train_pos
                filenames = self._train_filenames[self._train_pos:self._train_pos+batch_size]
                if len(filenames) < batch_size:
                    filenames += self._train_filenames[:batch_size - len(filenames)]

                self._train_pos += batch_size
                if self._train_pos >= self._train_size:
                    self._train_pos = 0
                    np.random.shuffle(self._train_filenames)
            elif dataset == 'validation':
                start_pos = self._validation_pos
                filenames = self._validation_filenames[self._validation_pos:self._validation_pos+batch_size]
                if len(filenames) < batch_size:
                    filenames += self._validation_filenames[:batch_size - len(filenames)]

                self._validation_pos += batch_size
                if self._validation_pos >= self._validation_size:
                    self._validation_pos = 0
            elif dataset == 'test':
                start_pos = self._test_pos
                filenames = self._test_filenames[self._test_pos:self._test_pos+batch_size]
                if len(filenames) < batch_size:
                    filenames += self._test_filenames[:batch_size - len(filenames)]

                self._test_pos += batch_size
                if self._test_pos >= self._test_size:
                    self._test_pos = 0
            else:
                raise ValueError("'{}' is not a valid argument.".format(dataset))

        return [filenames, start_pos] if return_start_pos else filenames

    def _feature_to_pad_size(self, feature_list, axis):
        """
        Returns a list containing the maximum length of each feature in feature_list

        :param feature_list: A list of lists. Each list in the inner level contains ndarrays representing the same
            feature
        :param axis: The axis to be used to compute the length.
        :return: A list of lengths, one for each feature
        """

        feature_to_size = lambda f: None if f.ndim == 1 else f.shape[axis]
        size_to_max = lambda list: None if list[0] is None else max(list)

        lengths = []
        for feature in feature_list:
            sizes = list(map(feature_to_size, feature))
            max_length = size_to_max(sizes)
            lengths.append(max_length)

        return lengths

    def _concat_list(self, lst, new_value):
        """
        An helper function used to concatenate values in a list. The function concatenate list and new_value, if the
        resulting list has only one element (list is None), the returned value will be a scalar (new_value). If list
        is a scalar, the returned value will be [list, new_value]. In the most general case, list is actually a list, so
        the returned value in this case will be list + [new_value]

        :param list: None, a scalar or a list
        :param new_value: a scalar value
        :return: The list obtained by concatenating list and new_value. If the obtained list has only one element, the
            returned value will be a scalar, not a list
        """

        if lst is not None:
            if isinstance(lst, list):
                return lst + [new_value]
            else:
                return [lst, new_value]
        else:
            return new_value

    def start_queue_workers(self, batch_size, dataset, processing_fn, features_to_pad_mask, concat_features,
                            multiple_examples, cache_preprocessed, threads, queue_size):

        f_data = dill.dumps(processing_fn)

        if dataset == 'train':
            self._train_queue = Queue(queue_size) if self._train_queue is None else self._train_queue
            workers_list, queue, started = self._train_queue_workers, self._train_queue, self._train_workers_started
        elif dataset == 'validation':
            self._validation_queue = Queue(queue_size) if self._validation_queue is None else self._validation_queue
            workers_list, queue, started = self._validation_queue_workers, self._validation_queue, \
                                           self._validation_workers_started
        elif dataset == 'test':
            self._test_queue = Queue(queue_size)if self._test_queue is None else self._test_queue
            workers_list, queue, started = self._test_queue_workers, self._test_queue, self._test_workers_started
        else:
            raise ValueError("The set '%s' does not exist!" % dataset)

        if False in started:
            for i in range(threads):
                p = Process(target=self._queue_writer,
                            args=(queue, batch_size, dataset, f_data, features_to_pad_mask, concat_features,
                                  multiple_examples, cache_preprocessed),
                            daemon=True,
                            name='writer_process_{}'.format(i))
                p.start()
                workers_list.append(p)

            while(not queue.full()):
                time.sleep(1)
            started[0] = True

    def _queue_writer(self, queue, batch_size, dataset, f_data, features_to_pad_mask, concat_features,
                      multiple_examples, cache_preprocessed):

        preprocessing_fn = dill.loads(f_data)
        # continuously reads examples and put them on the proper queue
        while True:
            batch = self.next_batch_single_thread(batch_size=batch_size,
                                                  dataset=dataset,
                                                  preprocessing_fn=preprocessing_fn,
                                                  features_to_pad_mask=features_to_pad_mask,
                                                  concat_features=concat_features,
                                                  multiple_examples=multiple_examples,
                                                  cache_preprocessed=cache_preprocessed)
            # waits if the queue is full
            queue.put(batch)

    def _next_batch_multithread(self, batch_size, dataset, preprocessing_fn=None, features_to_pad_mask=None, concat_features=False,
                                multiple_examples=1, cache_preprocessed=False, threads=2):
        """Returns the next batch of the dataset. This is a multi-thread implementation that creates 'threads' workers
        which will execute the single thread version next_batch_single_thread()

        :param batch_size: the size of the batch to be extracted.
        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: A function to be applied to each example before including it in the batch. The
            first value returned by the function must always be the length of the new sequence in order to allow
            automatic padding of features in the batch. All the features are automatically padded to the maximum
            length returned by this function across the examples in the same batch. If the features returned by the
            function must be padded with different lengths you can return a list of lengths, instead of a single
            value, and then use the features_to_pad_mask argument to specify to which length each feature must be
            padded with. The reader will compute the maximum of each length separately and pad features according to
            features_to_pad_mask.
            The signature of preprocessing_fn must be: (length, label, x, y, ts, p) -> (lengths, features...)
        :param features_to_pad_mask: A list of integers for each feature returned by the preprocessing function
            (except for the first one, the length). Each integer is an index to the length return list and specifies
            to which length each feature must be padded with. If this argument is not specified, all the feature
             will be padded with the first (or the only one) length contained in the first returned value of
             preprocessing_fn.
        :param concat_features: (optional, default False) A boolean specifying if the features must be stacked on a new
            axis, False option, or concatenated on the first axis. If this option is True, no padding will be applied
            assuming that features of the same type have that same dimensions except for the first axis
            (eg: [num_features_example_i, feature_dim], resulting after concat: [num_tot_features, feature_dim]). When
            you choose to stack features instead, the first dimension is padded to the maximum length based on
            features_to_pad_mask (the resulting array will be [num:examples, num_features_padded, feature_dim]).
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
        :param threads: The number of parallel threads to be used when reading files.
        :return a list containing the next batch's values from the specified 'set'"""

        futures = []
        issued_batches = 0
        small_batch_size = int(np.ceil(batch_size / threads))

        # Starts the executor and waits until all the workers have finished
        with ThreadPoolExecutor(max_workers=threads) as executor:
            while issued_batches < batch_size:

                # The last worker processes the remaining batches. If batch_size is evenly
                # divisible by threads small_batch_size will be the same also for the last
                # worker, otherwise a smaller one will be issued
                if issued_batches + small_batch_size > batch_size:
                    th_batch_size = batch_size - issued_batches
                    issued_batches += th_batch_size
                else:
                    th_batch_size = small_batch_size
                    issued_batches += small_batch_size

                future = executor.submit(self.next_batch_single_thread, th_batch_size, dataset,
                                         preprocessing_fn, features_to_pad_mask, concat_features,
                                         multiple_examples, cache_preprocessed)
                # Collect the results
                futures.append(future)

        res_by_features = list(zip(*[f.result() for f in futures]))
        lengths_by_feature = self._feature_to_pad_size(res_by_features, axis=1)

        # Each batch from the workers will be padded wrt its maximum length. When merging the results from
        # the workers we must extend the padding of the batches to match the maximum length.
        stacked_features = []
        for pad_length, feature in zip(lengths_by_feature, res_by_features):
            if pad_length is not None and concat_features is False:
                # Padding is applied over the feature dimension (axis=1)
                feature = self._pad_list(feature, pad_length, axis=1)
            stacked_feature = np.concatenate(list(feature), axis=0)
            stacked_features.append(stacked_feature)

        return stacked_features

    def next_batch_queue(self, batch_size, dataset, preprocessing_fn=None, features_to_pad_mask=None, concat_features=False,
                         multiple_examples=1, cache_preprocessed=False, threads=4, queue_size=10):
        """Returns the next batch of the dataset. This is a multi-thread implementation that creates 'threads' workers
                which will execute the single thread version next_batch_single_thread()

                :param batch_size: the size of the batch to be extracted.
                :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
                :param preprocessing_fn: A function to be applied to each example before including it in the batch. The
                    first value returned by the function must always be the length of the new sequence in order to allow
                    automatic padding of features in the batch. All the features are automatically padded to the maximum
                    length returned by this function across the examples in the same batch. If the features returned by the
                    function must be padded with different lengths you can return a list of lengths, instead of a single
                    value, and then use the features_to_pad_mask argument to specify to which length each feature must be
                    padded with. The reader will compute the maximum of each length separately and pad features according to
                    features_to_pad_mask.
                    The signature of preprocessing_fn must be: (length, label, x, y, ts, p) -> (lengths, features...)
                :param features_to_pad_mask: A list of integers for each feature returned by the preprocessing function
                    (except for the first one, the length). Each integer is an index to the length return list and specifies
                    to which length each feature must be padded with. If this argument is not specified, all the feature
                     will be padded with the first (or the only one) length contained in the first returned value of
                     preprocessing_fn.
                :param concat_features: (optional, default False) A boolean specifying if the features must be stacked on a new
                    axis, False option, or concatenated on the first axis. If this option is True, no padding will be applied
                    assuming that features of the same type have that same dimensions except for the first axis
                    (eg: [num_features_example_i, feature_dim], resulting after concat: [num_tot_features, feature_dim]). When
                    you choose to stack features instead, the first dimension is padded to the maximum length based on
                    features_to_pad_mask (the resulting array will be [num:examples, num_features_padded, feature_dim]).
                :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
                    process them together with the provided preprocessing function to obtain a unique set of features. This
                    allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
                    example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
                    (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
                :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
                    function for each example; if the examples are requested multiple times, the next epoch for instance, the
                    cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
                    function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
                :param threads: The number of parallel threads to be used when reading files.
                :return a list containing the next batch's values from the specified 'set'"""

        # Starts the
        self.start_queue_workers(batch_size, dataset, preprocessing_fn, features_to_pad_mask, concat_features,
                                 multiple_examples, cache_preprocessed, threads, queue_size)

        if dataset == 'train':
            return self._train_queue.get()
        elif dataset == 'validation':
            return self._validation_queue.get()
        elif dataset == 'test':
            return self._test_queue.get()
        else:
            raise ValueError("The set '%s' does not exist!" % dataset)

    def next_batch(self, batch_size, dataset, preprocessing_fn=None, features_to_pad_mask=None, concat_features=False,
                   multiple_examples=1, cache_preprocessed=False, threads=2, use_queue=True, queue_size=10):
        """Returns the next batch of the dataset. This is a multi-thread implementation that creates 'threads' workers
        which will execute the single thread version next_batch_single_thread()

        :param batch_size: the size of the batch to be extracted.
        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: A function to be applied to each example before including it in the batch. The
            first value returned by the function must always be the length of the new sequence in order to allow
            automatic padding of features in the batch. All the features are automatically padded to the maximum
            length returned by this function across the examples in the same batch. If the features returned by the
            function must be padded with different lengths you can return a list of lengths, instead of a single
            value, and then use the features_to_pad_mask argument to specify to which length each feature must be
            padded with. The reader will compute the maximum of each length separately and pad features according to
            features_to_pad_mask.
            The signature of preprocessing_fn must be: (length, label, x, y, ts, p) -> (lengths, features...)
        :param features_to_pad_mask: A list of integers for each feature returned by the preprocessing function
            (except for the first one, the length). Each integer is an index to the length return list and specifies
            to which length each feature must be padded with. If this argument is not specified, all the feature
             will be padded with the first (or the only one) length contained in the first returned value of
             preprocessing_fn.
        :param concat_features: (optional, default False) A boolean specifying if the features must be stacked on a new
            axis, False option, or concatenated on the first axis. If this option is True, no padding will be applied
            assuming that features of the same type have that same dimensions except for the first axis
            (eg: [num_features_example_i, feature_dim], resulting after concat: [num_tot_features, feature_dim]). When
            you choose to stack features instead, the first dimension is padded to the maximum length based on
            features_to_pad_mask (the resulting array will be [num:examples, num_features_padded, feature_dim]).
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
        :param threads: The number of parallel threads to be used when reading files.
        :return a list containing the next batch's values from the specified 'set'"""

        if use_queue:
            return self.next_batch_queue(batch_size, dataset, preprocessing_fn, features_to_pad_mask, concat_features,
                                         multiple_examples, cache_preprocessed, threads, queue_size)
        else:
            return self._next_batch_multithread(batch_size, dataset, preprocessing_fn, features_to_pad_mask,
                                                concat_features, multiple_examples, cache_preprocessed, threads)

    def next_batch_single_thread(self, batch_size, dataset, preprocessing_fn=None, features_to_pad_mask=None,
                                 concat_features=False, multiple_examples=1, cache_preprocessed=False):
        """Returns the next batch of the dataset. Single thread version of next_batch(), this function is called by
        the multi-thread implementation next_batch().

        :param batch_size: the size of the batch to be extracted.
        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: A function to be applied to each example before including it in the batch. The
            first value returned by the function must always be the length of the new sequence in order to allow
            automatic padding of features in the batch. All the features are automatically padded to the maximum
            length returned by this function across the examples in the same batch. If the features returned by the
            function must be padded with different lengths you can return a list of lengths, instead of a single
            value, and then use the features_to_pad_mask argument to specify to which length each feature must be
            padded with. The reader will compute the maximum of each length separately and pad features according to
            features_to_pad_mask.
            The signature of preprocessing_fn must be: (length, label, x, y, ts, p) -> (lengths, features...)
        :param features_to_pad_mask: A list of integers for each feature returned by the preprocessing function
            (except for the first one, the length). Each integer is an index to the length return list and specifies
            to which length each feature must be padded with. If this argument is not specified, all the feature
             will be padded with the first (or the only one) length contained in the first returned value of
             preprocessing_fn.
        :param concat_features: (optional, default False) A boolean specifying if the features must be stacked on a new
            axis, False option, or concatenated on the first axis. If this option is True, no padding will be applied
            assuming that features of the same type have that same dimensions except for the first axis
            (eg: [num_features_example_i, feature_dim], resulting after concat: [num_tot_features, feature_dim]). When
            you choose to stack features instead, the first dimension is padded to the maximum length based on
            features_to_pad_mask (the resulting array will be [num:examples, num_features_padded, feature_dim]).
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
        :param threads: The number of parallel threads to be used when reading files.
        :return a list containing the next batch's values from the specified 'set'"""

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
                    label = self._dir_to_label[os.path.basename(os.path.dirname(filename))]
                    label = self._one_hot(label)
                    # Extracts the events
                    length, x, y, ts, p = self.read_example(filename)
                    pair_values['length'] = self._concat_list(pair_values.get('length', None), length)
                    pair_values['label'] = self._concat_list(pair_values.get('label', None), label)
                    pair_values['x'] = self._concat_list(pair_values.get('x', None), x)
                    pair_values['y'] = self._concat_list(pair_values.get('y', None), y)
                    pair_values['ts'] = self._concat_list(pair_values.get('ts', None), ts)
                    pair_values['p'] = self._concat_list(pair_values.get('p', None), p)

                # Applies the preprocessing function if provided
                if preprocessing_fn is not None:
                    # The preprocessing function can produce a different number of features
                    try:
                        r_features = preprocessing_fn(pair_values['length'], pair_values['label'], pair_values['x'],
                                                      pair_values['y'], pair_values['ts'], pair_values['p'])
                    except:
                        no_error = False
                        filenames = pair_fn
                        while no_error != True:
                            traceback.print_exc()
                            print("\nError reading files: %s\nSkipping batch.\n" % filenames)
                            pair_values = {}
                            filenames = self._get_next_filenames(multiple_examples, dataset)
                            for filename in filenames:
                                label = self._dir_to_label[os.path.basename(os.path.dirname(filename))]
                                label = self._one_hot(label)
                                length, x, y, ts, p = self.read_example(filename)
                                pair_values['length'] = self._concat_list(pair_values.get('length', None), length)
                                pair_values['label'] = self._concat_list(pair_values.get('label', None), label)
                                pair_values['x'] = self._concat_list(pair_values.get('x', None), x)
                                pair_values['y'] = self._concat_list(pair_values.get('y', None), y)
                                pair_values['ts'] = self._concat_list(pair_values.get('ts', None), ts)
                                pair_values['p'] = self._concat_list(pair_values.get('p', None), p)
                            try:
                                r_features = preprocessing_fn(pair_values['length'], pair_values['label'],
                                                              pair_values['x'],
                                                              pair_values['y'], pair_values['ts'], pair_values['p'])
                                no_error = True
                            except:
                                no_error = False

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

                    features = [pair_values['label'], pair_values['x'], pair_values['y'], pair_values['ts'],
                                pair_values['p']]
                    length = pair_values['length']
                    if features_to_pad_mask is None:
                        features_to_pad_mask = [-1, 0, 0, 0, 0]

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
        """Returns the next example of the dataset.

        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: A function to be applied to each example before including it in the batch.
            The signature of preprocessing_fn must be: (length, label, x, y, ts, p) -> (lengths, features...)
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param return_key: A boolean, if true the method will return also an integer which is a unique identifier of
            the example inside the current batch (basically its position inside the current data ordering)
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.

        :return the next example from the specified 'set' and the key if specified"""

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
                # Retrieves the label from the dir name
                label = self._dir_to_label[os.path.basename(os.path.dirname(filename))]
                label = self._one_hot([label])[0]
                # Extracts the events
                length, x, y, ts, p = self.read_example(filename)
                pair_values['length'] = self._concat_list(pair_values.get('length', None), length)
                pair_values['label'] = self._concat_list(pair_values.get('label', None), label)
                pair_values['x'] = self._concat_list(pair_values.get('x', None), x)
                pair_values['y'] = self._concat_list(pair_values.get('y', None), y)
                pair_values['ts'] = self._concat_list(pair_values.get('ts', None), ts)
                pair_values['p'] = self._concat_list(pair_values.get('p', None), p)

            # Applies the preprocessing function if provided
            if preprocessing_fn is not None:
                # The preprocessing function can produce a different number of features
                r_features = preprocessing_fn(pair_values['length'], pair_values['label'], pair_values['x'],
                                              pair_values['y'], pair_values['ts'], pair_values['p'])
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

                features = [pair_values['label'], pair_values['x'], pair_values['y'], pair_values['ts'],
                            pair_values['p']]
                length = pair_values['length']

        result = [length] + features
        result += [first_pos] if return_key else []
        return result

    def next_op(self, dataset, preprocessing_fn=None, multiple_examples=1, data_types=None, return_key=True,
                cache_preprocessed=False):
        """
        Returns a tensorflow operation that provides a new example every time it is executed

        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: (optional) A function to be applied to each example before including it in the batch.
        :param data_types: (optional) a list of tensorflow's data types indicating the type of the values returned by
            the reader. This argument must be specified if a preprocessing function is provided that changes the number
            or type of the values returned by the reader.
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param return_key: if a unique key must be returned along with the features of the reader
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
        :return: The example's features and a unique identifier, if return_key is set
        """

        if data_types is None:
            data_types = [tf.int32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32]

        if return_key:
            data_types += [tf.int32]

        r_features = tf.py_func(lambda: self.next(dataset, preprocessing_fn, multiple_examples, return_key=return_key,
                                                  cache_preprocessed=cache_preprocessed),
                                [], data_types)

        length = r_features[0]
        key = r_features[-1]
        features = r_features[1:-1]

        reshaped_features = []
        for f in features:
            reshaped_features += [tf.reshape(f, [-1, 1])]

        return [length] + reshaped_features + [key]

    def get_dequeue_op(self, batch_size, dataset, preprocessing_fn=None, multiple_examples=1, data_types=None,
                       data_shapes=None, queue_capacity=1000, cache_preprocessed=False, threads=2):
        """
        Creates a tensorflow's operation that dequeues examples.

        :param batch_size: The size of the batch that has to be returned from the queue every time the operation is
        executed
        :param dataset: the set from which the examples should be extracted. Values: ['train', 'validation', 'test']
        :param preprocessing_fn: (optional) A function to be applied to each example before including it in the batch.
        :param multiple_examples: (optional, default 1) Eventually you can read multiple examples at a time and
            process them together with the provided preprocessing function to obtain a unique set of features. This
            allows you to merge multiple examples in one (eg: placing 2 digits in the same canvas to build a new bigger
            example). In this case, the preprocessing function will be called with pairs ov values instead of scalars
            (eg: if multiple_examples = 2, pre_fn((len1, len2), (lb1, lb2), (x1, x2), (y1, y2), (ts1, ts2), (p1, p2)) )
        :param data_types: (optional) a list of tensorflow's data types indicating the type of the values returned by
            the reader. This argument must be specified if a preprocessing function is provided that changes the number
            or type of the values returned by the reader.
        :param data_shapes: (optional) a list of shapes indicating the shapes of the values returned by
            the reader. This argument must be specified if a preprocessing function is provided that changes the number
            or shapes of the values returned by the reader.
        :param queue_capacity: (optional) the capacity of the queue
        :param cache_preprocessed: (optional, default False) If the reader must cache the results of the preprocessing
            function for each example; if the examples are requested multiple times, the next epoch for instance, the
            cached results will be loaded instead of being recomputed. This option is only available if a preprocessing
            function and the tmp directory (using the reader's constructor or set_tmp_dir() method) have been provided.
        :param threads: (optional) the number of threads that has to be used to enqueue examples

        :return: A tensorflow operation that dequeue examples
        """

        if data_types is None and data_shapes is None:
            data_types = [tf.int32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32]
            data_shapes = [[], [self.num_classes()], [None], [None], [None], [None]]

        # Creates the operation that reads a single example from the dataset
        example = tf.py_func(lambda: self.next(dataset, preprocessing_fn, multiple_examples,
                                               cache_preprocessed=cache_preprocessed), [], data_types)

        # Creates a queue and an op that enqueues examples one at a time in the queue.
        queue = tf.PaddingFIFOQueue(capacity=queue_capacity, dtypes=data_types, shapes=data_shapes)
        enqueue_op = queue.enqueue(example)

        # Creates the dequeue operation which is used as input to the tensorflow model
        inputs = queue.dequeue_many(batch_size, name="dequeue_{}".format(dataset))

        # Create a queue runner that will run threads in parallel to enqueue examples.
        qr = tf.train.QueueRunner(queue, [enqueue_op] * threads)
        tf.train.add_queue_runner(qr)

        return inputs

    def save(self, filename):
        """Saves the state of the reader so that it can be restored.

        :param filename: the filename where to write the reader's state"""

        np.savez(filename,
                 train_filenames=self._train_filenames,
                 train_pos=self._train_pos,
                 train_size=self._train_size,
                 validation_filenames=self._validation_filenames,
                 validation_pos=self._validation_pos,
                 validation_size=self._validation_size,
                 test_filenames=self._test_filenames,
                 test_pos=self._test_pos,
                 test_size=self._test_size,
                 labels=self._labels,
                 seed=self._seed,
                 tmp_dir=self._tmp_dir,
                 path=self._path)

    def _restore(self, filename):
        """Restore the reader's state from a checkpoint saved with save() method.

        :param filename: the filename of the checkpoint file."""

        save_file = np.load(filename)

        self._train_filenames = save_file['train_filenames'].tolist()
        self._train_pos = np.asscalar(save_file['train_pos'])
        self._train_size = np.asscalar(save_file['train_size'])
        self._validation_filenames = save_file['validation_filenames'].tolist()
        self._validation_pos = np.asscalar(save_file['validation_pos'])
        self._validation_size = np.asscalar(save_file['validation_size'])
        self._test_filenames = save_file['test_filenames'].tolist()
        self._test_pos = np.asscalar(save_file['test_pos'])
        self._test_size = np.asscalar(save_file['test_size'])
        self._seed = np.asscalar(save_file['seed'])
        self._tmp_dir = np.asscalar(save_file['tmp_dir'])
        self._path = np.asscalar(save_file['path'])
        self._labels = save_file['labels'].tolist()
        self._dir_to_label = dict(zip(self._labels, range(len(self._labels))))

        save_file.close()

    def _split_dataset_stratified(self, label_dirs, splits):
        """
        Split a dataset into subsets using stratified sampling.
        :param label_dirs: a list of strings representing, for each label in the dataset, the directory containing
        examples of the corresponding label
        :param splits: a list of integers containing the amount of samples in each split
        :return: a list containing one list of samples for each split. The last list contain any remaining sample
        """

        num_splits = len(splits)
        label_samples = [glob.glob(os.path.join(dir, '*.*')) for dir in label_dirs]
        num_samples = sum([len(samples) for samples in label_samples])
        num_labels = len(label_dirs)

        splits_filenames = [[] for _ in range(num_splits + 1)]

        # For every label subfolder
        for lb, filenames in enumerate(label_samples):
            np.random.shuffle(filenames)
            label_prop = len(filenames) / num_samples
            start = 0
            for i in range(num_splits):
                n = int(np.ceil(splits[i] * label_prop))
                if lb + 1 == num_labels:
                    n = splits[i] - len(splits_filenames[i])
                splits_filenames[i] = splits_filenames[i] + filenames[start:start + n]
                start += n
                if start > len(filenames):
                    raise ValueError("Not enough samples")
            splits_filenames[-1] = splits_filenames[-1] + filenames[start:]

        return splits_filenames

    def _init(self, path, validation_size=None, test_size=None, tmp_dir=None, seed=1234):
        """Initializes the reader given path where the dataset is located.
        :param path: the path where the dataset is located. The expected directory structure must
            have a sub-directory for each label containing all the examples of the corresponding label:

            train
                ├── class0
                |   ├──example1.bin
                |   ├──example2.bin
                |   └──...
                ├── class1
                |   └──...
                |
            test
                └──...
            validation
                └──...

        :param validation_size: number of examples to be taken from the original training set in order to build the
            validation set.
        :param tmp_dir: (optional) A directory used by the reader to save temporary files. It can be used to store
            preprocessed files to be reused.
        :param seed: the random seed"""

        # Sets the seed
        np.random.seed(seed)
        self._seed = seed
        self._path = path
        self._tmp_dir = tmp_dir

        if os.path.exists(os.path.join(path, 'train')):
            train_label_paths = glob.glob(os.path.join(path, 'train', '*'))
        else:
            # if no 'train' directory exists, it assumes samples are inside the provided path
            train_label_paths = glob.glob(os.path.join(path, '*'))

        if len(train_label_paths) == 0:
            raise Exception("The provided path does not contain any data or the directory structure is not valid.")

        # Imposes a specific ordering, glob returns arbitrarily ordered names
        train_label_paths = sorted(train_label_paths)

        # Computes a mapping between directory names and integer labels
        indices = range(len(train_label_paths))
        self._labels = list(map(lambda p: os.path.basename(p), train_label_paths))
        self._dir_to_label = dict(zip(self._labels, indices))

        # If validation or test directories do not exist, create val and test sets from the training set
        create_val = validation_size is not None and not os.path.exists(os.path.join(path, 'validation'))
        create_test = test_size is not None and not os.path.exists(os.path.join(path, 'test'))

        if create_val and create_test:
            splits = self._split_dataset_stratified(train_label_paths, [validation_size, test_size])
            validation_filenames, test_filenames, train_filenames = splits
        elif create_val:
            splits = self._split_dataset_stratified(train_label_paths, [validation_size])
            validation_filenames, train_filenames = splits
            test_filenames = glob.glob(os.path.join(path, 'test', '*/*.*'))
        elif create_test:
            splits = self._split_dataset_stratified(train_label_paths, [test_size])
            test_filenames, train_filenames = splits
            validation_filenames = glob.glob(os.path.join(path, 'validation', '*/*.*'))
        else:
            train_filenames = glob.glob(os.path.join(path, 'train', '*/*.*'))
            validation_filenames = glob.glob(os.path.join(path, 'validation', '*/*.*'))
            test_filenames = glob.glob(os.path.join(path, 'test', '*/*.*'))

        # Shuffles the filenames in-place
        np.random.shuffle(train_filenames)
        np.random.shuffle(validation_filenames)
        np.random.shuffle(test_filenames)

        self._train_filenames = train_filenames
        self._validation_filenames = validation_filenames
        self._test_filenames = test_filenames

        # Variables used to keep track of the current position in the dataset
        self._train_size = len(self._train_filenames)
        self._validation_size = len(self._validation_filenames)
        self._test_size = len(self._test_filenames)

    def reset(self, seed=1234):
        self._init(self._path, validation_size=self._validation_size, test_size=self._test_size,
                   tmp_dir=self._tmp_dir, seed=seed)


class NEventReader(EventReader, NReader):
    """A class for reading N-MNIST and N-Caltech101 datasets"""

    def __init__(self, path, validation_size=0, test_size=0, tmp_dir=None, seed=1234):
        super().__init__(path, validation_size=validation_size, test_size=test_size, tmp_dir=tmp_dir, seed=seed)


class AerEventReader(EventReader, AerReader):
    """A class for reading AER datasets"""

    def __init__(self, path, validation_size=0, test_size=0, tmp_dir=None, seed=1234, camera="DVS128"):
        """
        Different cameras may have different data format. This argument is ignored if the dataset has been saved
        with the 3.1 format. In this case the events' format is specified in the header of each event.
        Further info: https://inilabs.com/support/software/fileformat/
        """
        super().__init__(path, validation_size=validation_size, test_size=test_size, tmp_dir=tmp_dir,
                         seed=seed, camera=camera)


class NumpyEventReader(EventReader, NumpyReader):
    """A class for reading Prophesee datasets"""

    def __init__(self, path, validation_size=0, test_size=0, tmp_dir=None, seed=1234):
        super().__init__(path, validation_size=validation_size, test_size=test_size, tmp_dir=tmp_dir,
                         seed=seed)


def factory(path, file_format, validation_size=10000, test_size=10000, tmp_dir=None, seed=1234, **kargs):
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
        return NEventReader(path=path, validation_size=validation_size, test_size=test_size, tmp_dir=tmp_dir, seed=seed)
    elif file_format.startswith('aer-data'):
        format = file_format.split('_')
        try:
            camera = format[1]
        except IndexError:
            camera = None
        return AerEventReader(path=path, validation_size=validation_size, test_size=test_size, tmp_dir=tmp_dir,
                              seed=seed, camera=camera)
    else:
        raise ValueError('The provided file format ({}) is unknown'.format(file_format))


def restore(save_filename, file_format):
    """
    This method is used to restore a previously saved reader from its save
    :param save_filename: The save file from which to restore the reader
    :return: a DVSReader object
    """

    if file_format == 'n-data':
        return NEventReader(path=save_filename)
    elif file_format == 'aer-data':
        return AerEventReader(path=save_filename)
    elif file_format == 'numpy-data':
        return NumpyEventReader(path=save_filename)
    else:
        raise ValueError('The provided file format ({}) is unknown'.format(file_format))

