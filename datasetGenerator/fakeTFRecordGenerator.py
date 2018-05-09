import tensorflow as tf
import numpy as np
import time
import os
import sys
from datasetGenerator.tfRecordGenerator import TFRecordGenerator

__author__ = 'Andres'


class FakeTFRecordGenerator(TFRecordGenerator):
    def generateDataset(self):
        start = time.time()

        train_filename = self.name() + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(train_filename)

        print("start:", start)
        count = 0
        total = 0

        _sampling_rate = 16000
        _window_size = 5120
        _time = np.arange(0, _window_size / _sampling_rate, 1 / _sampling_rate)
        _low_freq = np.arange(0, 2000, 40)
        _mid_low_freq = np.arange(2000, 4000, 40)
        _mid_high_freq = np.arange(4000, 6000, 40)
        _high_freq = np.arange(6000, 8000, 40)

        for low_freq in _low_freq:
            for mid_low_freq in _mid_low_freq:
                for mid_high_freq in _mid_high_freq:
                    for high_freq in _high_freq:
                        audio = np.sin(2 * np.pi * low_freq * _time) + np.sin(2 * np.pi * mid_low_freq * _time) + \
                                np.sin(2 * np.pi * mid_high_freq * _time) + np.sin(2 * np.pi * high_freq * _time)

                        self._createFeature(audio, writer)

                        count, total = self._notifyIfNeeded(count + 1, total)
                        sys.stdout.flush()
        writer.close()
        end = time.time() - start

        print("there were: ", total + count)
        print("wow, that took", end, "seconds... might want to change that to mins :)")


    def _filenameShouldBeLoaded(self, filename):
        raise NotImplementedError("We fake bro")
