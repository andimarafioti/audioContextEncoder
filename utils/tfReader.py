__author__ = 'Andres'

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError


class TFReader(object):
    def __init__(self, path_to_tfRecord_file, window_size, batchSize, num_epochs=10, capacity=100000):
        self._path_to_tfRecord_file = path_to_tfRecord_file
        self._capacity = capacity
        self._batchSize = batchSize
        self._window_size = window_size
        self._audios = self._read_and_decode(tf.train.string_input_producer([path_to_tfRecord_file],
                                                                            num_epochs=num_epochs))

    def start(self):
        self._coordinator = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(coord=self._coordinator)

    def dataOperation(self, session):
        try:
            audios = session.run(self._audios)
            return audios
        except OutOfRangeError:
            raise StopIteration

    def finish(self):
        self._coordinator.request_stop()
        self._coordinator.join(self._threads)

    def _read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={'valid/windows': tf.FixedLenFeature([], tf.string)})

        windows = tf.decode_raw(features['valid/windows'], tf.float32)
        windows = tf.reshape(windows, [self._window_size])

        audios = tf.train.shuffle_batch([windows], batch_size=self._batchSize,
                                        min_after_dequeue=int(self._capacity * 0.5),
                                        capacity=self._capacity,
                                        num_threads=4)
        return audios
