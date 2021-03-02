import numpy as np
import bitstring
import time

from numba import njit


class FileReader:

    def __init__(self, **kargs):
        super().__init__()

    def read_example(self, filename):
        raise Exception("This function is not implemented.\n"
                        "You are directly using the FileReader abstract class, instead you should use one of its "
                        "implementations through the factory() method")

    def save_example(self, filename, x, y, ts, p, version):
        raise Exception("This function is not implemented.\n"
                        "You are directly using the FileReader abstract class, instead you should use one of its "
                        "implementations through the factory() method")


class NReader(FileReader):
    """A class for reading N-MNIST and N-Caltech101 datasets"""

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def read_example(self, filename):
        """Reads the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""

        f = open(filename, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        x = np.array(all_x[td_indices], dtype=np.int32)
        y = np.array(all_y[td_indices], dtype=np.int32)
        ts = np.array(all_ts[td_indices], dtype=np.int32)
        p = np.array(all_p[td_indices], dtype=np.int32)
        length = len(x)

        return length, x, y, ts, p

    def save_example(self, filename, x, y, ts, p, version):

        final_x = x.astype(dtype=np.uint64) << 32
        final_y = y.astype(dtype=np.uint64) << 24
        final_p = p.astype(dtype=np.uint64) << 23
        final_ts = ts.astype(dtype=np.uint64)

        final_all = final_x + final_y + final_p + final_ts

        bits = bitstring.Bits().join(bitstring.Bits(uint=val_uint64, length=40) for val_uint64 in np.nditer(final_all))
        bytes = bits.tobytes()

        f = open(filename, 'wb')
        f.write(bytes)
        f.close()


class AerReader(FileReader):
    """A class for reading AER datasets"""

    def __init__(self, camera="DVS128", **kargs):
        """
        :param camera: (optional) This argument is used to specify the camera format used to store the events.
        Different cameras may have different data format. This argument is ignored if the dataset has been saved
        with the 3.1 format. In this case the events' format is specified in the header of each event.
        Further info: https://inilabs.com/support/software/fileformat/
        """
        super().__init__(**kargs)
        self._camera = camera

    def _get_camera_format(self):

        if self._camera is None:
            raise ValueError("No camera format has been specified. If you are using the factory method to obtain the "
                             "reader object, make sure to specify also the camera type in the file_format argument "
                             "(eg. 'aer-data_DVS128)'")
        elif self._camera == "DVS128":
            x_mask = 0xFE
            x_shift = 1
            y_mask = 0x7F00
            y_shift = 8
            p_mask = 0x1
            p_shift = 0
        else:
            raise ValueError("Unsupported camera: {}".format(self._camera))

        return x_mask, x_shift, y_mask, y_shift, p_mask, p_shift

    def _read_aedat20_events(self, f):

        raw_data = np.fromfile(f, dtype=np.int32).newbyteorder('>')
        f.close()

        all_data = raw_data[0::2]
        all_ts = raw_data[1::2]

        # Events' representation depends of the camera format
        x_mask, x_shift, y_mask, y_shift, p_mask, p_shift = self._get_camera_format()

        all_x = ((all_data & x_mask) >> x_shift).astype(np.int32)
        all_y = ((all_data & y_mask) >> y_shift).astype(np.int32)
        all_p = ((all_data & p_mask) >> p_shift).astype(np.int32)
        all_ts = all_ts.astype(np.int32)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def _read_aedat31_events(self, f):

        # WARNING: This function assumes that all the events are of type POLARITY_EVENT and so
        # each packet has a fixed size and structure. If your dataset may contain other type of events you
        # must write a function to properly handle different packets' sizes and formats.
        # See: https://inilabs.com/support/software/fileformat/#h.w7vjqzw55d5b

        raw_data = np.fromfile(f, dtype=np.int32)
        f.close()

        all_x, all_y, all_ts, all_p = [], [], [], []

        while raw_data.size > 0:

            # Reads the header
            block_header, raw_data = raw_data[:7], raw_data[7:]
            eventType = block_header[0] >> 16
            eventSize, eventTSOffset, eventTSOverflow, eventCapacity, eventNumber, eventValid = block_header[1:]
            size_events = eventNumber * eventSize // 4
            events, raw_data = raw_data[:size_events], raw_data[size_events:]

            if eventValid and eventType == 1:
                data = events[0::2]
                ts = events[1::2]

                x = ((data >> 17) & 0x1FFF).astype(np.int32)
                y = ((data >> 2) & 0x1FFF).astype(np.int32)
                p = ((data >> 1) & 0x1).astype(np.int32)
                valid = (data & 0x1).astype(np.bool)
                ts = ((eventTSOverflow.astype(np.int64) << 31) | ts).astype(np.int64)

                # The validity bit can be used to invalidate events. We filter out the invalid ones
                if not np.all(valid):
                    x = x[valid]
                    y = y[valid]
                    ts = ts[valid]
                    p = p[valid]

                all_x.append(x)
                all_y.append(y)
                all_ts.append(ts)
                all_p.append(p)

        all_x = np.concatenate(all_x, axis=-1)
        all_y = np.concatenate(all_y, axis=-1)
        all_ts = np.concatenate(all_ts, axis=-1)
        all_p = np.concatenate(all_p, axis=-1)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def read_example(self, filename):
        f = open(filename, 'rb')

        # If comment section is not present, version 1.0 is assumed by the standard
        version = "1.0"
        prev = 0
        line = f.readline().decode("utf-8", "ignore")
        # Reads the comments and extracts the aer format version
        while line.startswith('#'):
            if line[0:9] == '#!AER-DAT':
                version = line[9:12]
            prev = f.tell()
            line = f.readline().decode("utf-8", "ignore")
        # Repositions the pointer at the beginning of data section
        f.seek(prev)

        if version == "2.0":
            length, all_x, all_y, all_ts, all_p = self._read_aedat20_events(f)
        elif version == "3.1":
            length, all_x, all_y, all_ts, all_p = self._read_aedat31_events(f)
        else:
            raise NotImplementedError("Reader for version {} has not yet been implemented.".format(version))

        return length, all_x, all_y, all_ts, all_p

    def _save_aedat20_events(self, filename, x, y, ts, p):

        header = "#!AER-DAT2.0\r\n" \
                 "# This is a raw AE data file - do not edit\r\n" \
                 "# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n" \
                 "# Timestamps tick is 1 us\r\n" \
                 "# created "+time.ctime()+"\r\n"

        # Events' representation depends of the camera format
        _, x_shift, _, y_shift, _, p_shift = self._get_camera_format()

        final_x = (x.astype(dtype=np.uint32) & 0x7F) << x_shift
        final_y = (y.astype(dtype=np.uint32) & 0x7F) << y_shift
        final_p = (p.astype(dtype=np.uint32) & 0x7F) << p_shift
        final_ts = ts.astype(dtype=np.uint32)

        final_data = np.bitwise_or.reduce((final_y, final_x, final_p))
        final_all = np.stack([final_data, final_ts], axis=-1)

        f = open(filename, 'wb')
        f.write(header.encode())
        uint32 = np.dtype(np.uint32).newbyteorder('>')
        f.write(final_all.astype(uint32).tobytes())
        f.close()

    def _save_aedat31_events(self, filename, x, y, ts, p):

        header = "#!AER-DAT3.1\r\n" \
                 "#Format: RAW\r\n" \
                 "#Source 1: " + self._camera + "\r\n" \
                 "#Start-Time: " + time.strftime("%Y-%m-%d %H:%M:%S (TZ%z)", time.localtime()) + "\r\n" \
                 "#!END-HEADER\r\n"

        # Computes the int32 timestamps' overflows
        overflows = (ts >> 31) & 0x7FFFFFFF
        idx_blocks = np.where(overflows[:-1] != overflows[1:])[0]
        idx_blocks = [len(ts) - 1] if len(idx_blocks) == 0 else idx_blocks
        start_idx = 0
        bytes = b''
        for end_idx in idx_blocks:
            block_x = x[start_idx:end_idx + 1]
            block_y = y[start_idx:end_idx + 1]
            block_ts = ts[start_idx:end_idx + 1] & 0x7FFFFFFF
            block_p = p[start_idx:end_idx + 1]

            # Constructs header for Polarity Event data
            num_events = len(block_x)
            eventTypeSource = 1 << 16 | 1  # Type: Polarity Event (1), Source: device 0
            eventSize = 8
            eventTSOffset = 4
            eventTSOverflow = (ts[start_idx] >> 31) & 0x7FFFFFFF
            eventCapacity = num_events
            eventNum = num_events
            eventValid = num_events

            event_header = np.array([eventTypeSource, eventSize, eventTSOffset, eventTSOverflow,
                                     eventCapacity, eventNum, eventValid], dtype=np.int32)
            all_data = np.expand_dims(np.bitwise_or.reduce([block_x << 17, block_y << 2, block_p << 1, 1]), axis=-1)
            all_ts = np.expand_dims(block_ts, axis=-1)
            final_all = np.concatenate([all_data, all_ts], axis=-1)
            bytes += event_header.astype(np.int32).tobytes()
            bytes += final_all.astype(np.int32).tobytes()
            start_idx = end_idx + 1

        f = open(filename, 'wb')
        f.write(header.encode())
        f.write(bytes)
        f.close()

    def save_example(self, filename, x, y, ts, p, version):

        if version == "2.0":
            self._save_aedat20_events(filename, x, y, ts, p)
        elif version == "3.1":
            self._save_aedat31_events(filename, x, y, ts, p)
        else:
            raise NotImplementedError("A saver for this data format has not yet been implemented.")


class NumpyReader(FileReader):
    """A class for reading .npy files"""

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def read_example(self, filename):

        with open(filename, 'rb') as f:
            events = np.load(f)
        all_x, all_y, all_ts, all_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        length = len(all_x)
        return length, all_x, all_y, all_ts, all_p

    def save_example(self, filename, x, y, ts, p, version):

        all = np.stack([x, y, ts, p], axis=-1)
        np.save(filename, all)
