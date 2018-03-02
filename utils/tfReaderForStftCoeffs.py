__author__ = 'Andres'

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError


class TFReaderForStftCoeffs(object):
    def __init__(self, path_to_tfRecord_file, shape_of_stft_coeffs, num_epochs=10, capacity=100000):
        self._path_to_tfRecord_file = path_to_tfRecord_file
        self._capacity = capacity
        self._shape_of_stft_coeffs = shape_of_stft_coeffs
        self._audios = self._read_and_decode(tf.train.string_input_producer([path_to_tfRecord_file],
                                                                            num_epochs=num_epochs))

    def start(self):
        self._coordinator = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(coord=self._coordinator)

    def dataOperation(self, session):
        try:
            sides, gaps = session.run(self._audios)
            return sides, gaps
        except OutOfRangeError:
            raise StopIteration

    def finish(self):
        self._coordinator.request_stop()
        self._coordinator.join(self._threads)

    def _read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={'valid/stft': tf.FixedLenFeature([], tf.string)})

        stftCoeffs = tf.decode_raw(features['valid/stft'], tf.float32)
        stftCoeffs = tf.reshape(stftCoeffs, self._shape_of_stft_coeffs)

        batchs = tf.train.shuffle_batch(stftCoeffs, batch_size=256, min_after_dequeue=int(self._capacity * 0.5),
                                        capacity=self._capacity,
                                        num_threads=4)
        return batchs
