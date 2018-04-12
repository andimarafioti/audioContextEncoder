import librosa
import numpy as np
import tensorflow as tf
import time
import os
import sys

from audioread import NoBackendError

__author__ = 'Andres'


class TFRecordGenerator(object):
    """To generate a Dataset, instantiate this class with its arguments and call generateDataset()"""

    def __init__(self, baseName, pathToDataFolder, exampleProcessor, targetSamplingRate=16000, notifyEvery=10000):
        self._pathToDataFolder = pathToDataFolder
        self._exampleProcessor = exampleProcessor
        self._notifyEvery = notifyEvery
        self._targetSamplingRate = targetSamplingRate
        self._baseName = baseName

    def name(self):
        return self._baseName + self._exampleProcessor.describe()

    def generateDataset(self):
        start = time.time()

        train_filename = self.name() + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(train_filename)

        print("start:", start)
        count = 0
        total = 0

        for file_name in os.listdir(self._pathToDataFolder):
            if self._filenameShouldBeLoaded(file_name):
                try:
                    audio, sr = librosa.load(self._pathToDataFolder + '/' + file_name, sr=self._targetSamplingRate)
                except NoBackendError:
                    print("No backend for file:", file_name)
                    continue
                sides, gaps = self._exampleProcessor.process(audio)
                if sides.shape[0] is 0:
                    print("Got a completely silenced signal! with path:", file_name)
                    continue

                for side, gap in zip(sides, gaps):
                    self._createFeature(side, gap, writer)

                count, total = self._notifyIfNeeded(count + len(sides), total)
                sys.stdout.flush()
        writer.close()
        end = time.time() - start

        print("there were: ", total + count)
        print("wow, that took", end, "seconds... might want to change that to mins :)")

    def _createFeature(self, side, gap, writer):
        side_bytes = side.astype(np.float32).tostring()
        gap_bytes = gap.astype(np.float32).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'valid/sides': self._bytes_feature(side_bytes),
            'valid/gaps': self._bytes_feature(gap_bytes)}))

        writer.write(example.SerializeToString())

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _filenameShouldBeLoaded(self, filename):
        raise NotImplementedError("Subclass Responsibility")

    def _notifyIfNeeded(self, count, total):
        if count > self._notifyEvery:
            count -= self._notifyEvery
            total += self._notifyEvery
            print(self._notifyEvery, "plus!", time.time())
            return count, total
        return count, total
