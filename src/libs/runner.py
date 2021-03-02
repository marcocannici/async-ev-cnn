import cv2
import time

import numpy as np
from functools import partial

from src.libs.utils import center_crop
from src.libs.viz import draw_bboxes, integrate_frame


class Runner:

    def __init__(self, args, reader, profile_integration=False):
        self.args = args
        self.reader = reader
        self.num_classes = self.reader.num_classes()
        self.profile_integration = profile_integration

        label_k = np.array(list(reader.label_to_idx().keys()))
        label_v = np.array(list(reader.label_to_idx().values()))
        self.idx_to_label = label_k[np.argsort(label_v)]

    @staticmethod
    def data_transform(l, x, y, ts, p, bboxes, args):
        ts = ts - ts[0]
        if args.frame_h != args.example_h or \
                args.frame_w != args.example_w:
            l, x, y, ts, p, bboxes = center_crop(l, x, y, ts, p, bboxes,
                                                 (args.example_h, args.example_w),
                                                 (args.frame_h, args.frame_w))

        events = np.stack([y, x, ts], axis=-1)
        return l, events

    def show_frames(self, net_out, frames, *args, **kwargs):
        drawn_frames = draw_bboxes(net_out, frames, self.args.yolo_num_cells_h, self.args.yolo_num_cells_w,
                                   self.num_classes, idx_to_label=self.idx_to_label, conf_threshold=0.1,
                                   nms_threshold=0., use_nms=True,
                                   max_thickness=1, highlight_top_n=2, resize_ratio=5)

        for frame in drawn_frames:
            cv2.imshow('Predictions', frame)
            cv2.waitKey(self.args.frame_delay)
        cv2.waitKey(1)

    def feed_network(self, network, events, frames, reset_state, *args, **kwargs):
        raise NotImplementedError()

    def run(self, network, *args, **kwargs):

        n = 0
        ex_time = []

        # Test loop: forward pass through all the test examples
        for i in range(int(np.ceil(self.reader.test_size() / self.args.batch_size))):
            start_read = time.time()
            _, events = self.reader.next_batch(self.args.batch_size, dataset='test',
                                               preprocessing_fn=partial(self.data_transform, args=self.args),
                                               concat_features=False,
                                               threads=self.args.reader_threads)
            end_reading = time.time()

            loop_state = None
            reset_state = True
            if self.args.batch_event_usec is not None:
                bins = np.arange(0, events[:, -1][-1], self.args.batch_event_usec)
                bin_ids = np.digitize(events[:, -1], bins)
                split_indices = np.where(bin_ids[:-1] != bin_ids[1:])[0] + 1
                event_batches = np.array_split(events, indices_or_sections=split_indices, axis=0)
            else:
                num_event_batches = int(np.ceil(events.shape[0] / self.args.batch_event_size))
                event_batches = np.array_split(events, indices_or_sections=num_event_batches, axis=0)

            for events_batch in event_batches:
                # Reconstruct frame
                # =================
                if self.profile_integration:
                    start_fw = time.time()
                frames, prev_ts = integrate_frame(events_batch, self.args.leak,
                                                 self.args.frame_h, self.args.frame_w,
                                                 loop_state)
                loop_state = [frames, prev_ts]
                if not self.profile_integration:
                    start_fw = time.time()

                # Network forward step
                # ====================
                net_out = self.feed_network(network, events, frames, reset_state, *args, **kwargs)
                time_fw = time.time() - start_fw
                ex_time.append(time_fw)
                n += 1
                print("Test batch {:<2} - sec/example: {:.3f}  reading: {:.3f} sec"
                      "".format(i + 1, time_fw, end_reading - start_read))

                if n % 1000 == 0:
                    print("Mean fw time ({} runs): {}".format(n, np.mean(ex_time)))

                # Show frames
                # ====================
                self.show_frames(net_out, frames, loop_state)
                reset_state = False

        cv2.destroyAllWindows()


class TfFrameRunner(Runner):
    def __init__(self, args, reader):
        super().__init__(args, reader, profile_integration=True)

    def feed_network(self, network, events, frames, reset_state, sess, placeholder, *args, **kwargs):
        return sess.run(network, feed_dict={placeholder: frames})


class NumpyFrameRunner(Runner):
    def __init__(self, args, reader):
        super().__init__(args, reader, profile_integration=True)

    def feed_network(self, network, events, frames, reset_state, *args, **kwargs):
        return network(frames)


class NumpyEventRunner(Runner):
    def __init__(self, args, reader):
        super().__init__(args, reader, profile_integration=False)

    def feed_network(self, network, events, frames, reset_state, *args, **kwargs):
        return network(events, reset_state)

