import librosa
import numpy as np
import tensorflow as tf
import time
import os
import sys

__author__ = 'Andres'


class TFRecordGenerator(object):
    """To generate a Dataset, instanciate this class with its arguments and call generateDataset()"""
    def __init__(self, baseName, pathToDataFolder, window_size, gapLength, hopSize, notifyEvery=10000):
        self._pathToDataFolder = pathToDataFolder
        self._windowSize = window_size
        self._gapLength = gapLength
        self._hopSize = hopSize

        self._notifyEvery = notifyEvery
        self._baseName = baseName

    def name(self):
        return self._baseName + "_w" + str(self._windowSize) + '_g' + str(self._gapLength) + '_h' + str(self._hopSize)

    def generateDataset(self):
        start = time.time()

        train_filename = self.name() + '.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)

        time_per_loaded = []
        time_to_HD = []
        print("start:", start)
        i = 0
        total = 0
        for file_name in os.listdir(self._pathToDataFolder):
            if file_name.endswith('.wav'):
                now = time.time()
                audio, sr = librosa.load(self._pathToDataFolder + '/' + file_name, sr=None)
                time_to_HD.append(time.time() - now)
                sides, gaps = self._generate_audio_input(audio)
                if sides.shape[0] is 0:
                    #             print("Got a completely silenced signal! with path:", file_name)
                    continue

                for side, gap in zip(sides, gaps):
                    # Create a feature
                    i += 1
                    gap = np.reshape(gap, [self._gapLength])
                    side = np.reshape(side, [self._windowSize - self._gapLength])
                    side_bytes = side.astype(np.float32).tostring()
                    gap_bytes = gap.astype(np.float32).tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'valid/sides': self._bytes_feature(side_bytes),
                        'valid/gaps': self._bytes_feature(gap_bytes)}))

                    writer.write(example.SerializeToString())

                if i > self._notifyEvery:
                    i -= self._notifyEvery
                    total += self._notifyEvery
                    print(self._notifyEvery, "plus!", time.time())
                sys.stdout.flush()
                time_per_loaded.append(time.time() - now)
        writer.close()
        print("there were: ", total + i)

        end = time.time() - start
        print("wow, that took", end, "seconds... might want to change that to mins :)")
        print("On average, it took", np.average(time_per_loaded), "per loaded file")
        print("With an standard deviation of", np.std(time_per_loaded))
        print("On average, it took", np.average(time_to_HD), "per loaded file")
        print("With an standard deviation of", np.std(time_to_HD))

    def _generate_audio_input(self, audio_signal):
        audio_without_silence_at_beginning_and_end = self._trim_silence(audio_signal, frame_length=self._gapLength)
        splited_audio = self._windower(audio_without_silence_at_beginning_and_end, self._windowSize,
                                       hop_size=self._hopSize)
        sides, gaps = self._divide_data_into_data_and_gaps(splited_audio, self._windowSize, gap_length=self._gapLength)
        return sides, gaps

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _divide_data_into_data_and_gaps(self, data, window_size, gap_length):
        begin = int(np.floor((window_size - gap_length) / 2))
        end = int(np.floor((window_size + gap_length) / 2))
        processed_data = np.concatenate((data[:, :begin], data[:, end:]), axis=1)
        processed_data = np.reshape(processed_data, [-1, window_size - gap_length, 1])
        gaps = data[:, begin:end]
        gaps = np.reshape(gaps, [-1, gap_length])

        processed_data, gaps = self._remove_examples_with_average_sample_below(processed_data, gaps, gap_length,
                                                                               threshold=1e-4)
        return processed_data, gaps

    def _remove_examples_with_average_sample_below(self, processed_data, gaps, gap_length, threshold):
        mask = np.where(np.sum(np.abs(gaps), axis=1) < gap_length * threshold)
        processed_data = np.delete(processed_data, mask, axis=0)
        gaps = np.delete(gaps, mask, axis=0)

        return processed_data, gaps

    def _trim_silence(self, audio, threshold=0.01, frame_length=1024):
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rmse(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    def _windower(self, audio_signal, window_size, hop_size=-1):
        if hop_size == -1:
            hop_size = window_size
        window_count = int((len(audio_signal) - window_size) / hop_size)

        windowed_audios = np.array([])
        for window_index in range(int(window_count)):
            initial_index = int(window_index * hop_size)
            windowed_audios = np.append(windowed_audios, audio_signal[initial_index:initial_index + window_size])
        windowed_audios = np.reshape(windowed_audios, (-1, window_size))
        return windowed_audios
