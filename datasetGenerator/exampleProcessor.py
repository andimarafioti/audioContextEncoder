import numpy as np
import librosa

__author__ = 'Andres'


class ExampleProcessor(object):
    def __init__(self, gapLength, sideLength, hopSize, gapMinRMS=1e-3):
        self._sideLength = sideLength
        self._gapLength = gapLength
        self._totalLength = gapLength + 2*sideLength
        self._hopSize = hopSize
        self._gapMinRMS = gapMinRMS

    def gapLength(self):
        return self._gapLength

    def sideLength(self):
        return self._sideLength

    def describe(self):
        return "_w" + str(self._totalLength) + '_g' + str(self._gapLength) \
               + '_h' + str(self._hopSize) + '_t' + str(self._gapMinRMS)

    def process(self, audio_signal):
        audio_without_silence_at_beginning_and_end = self._trim_silence(audio_signal, frame_length=self._gapLength)
        splited_audio = self._window(audio_without_silence_at_beginning_and_end)
        sides, gaps = self._split_audio_into_sides_and_gaps(splited_audio)
        return sides, gaps

    def _trim_silence(self, audio, frame_length=1024):
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rmse(audio, frame_length=frame_length)
        frames = np.nonzero(energy > self._gapMinRMS*frame_length)
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    def _window(self, audio_signal):
        window_count = int((len(audio_signal) - self._totalLength) / self._hopSize)

        windowed_audios = np.array([])
        for window_index in range(int(window_count)):
            initial_index = int(window_index * self._hopSize)
            windowed_audios = np.append(windowed_audios, audio_signal[initial_index:initial_index + self._totalLength])
        windowed_audios = np.reshape(windowed_audios, (-1, self._totalLength))
        return windowed_audios

    def _split_audio_into_sides_and_gaps(self, audio):
        begin = int(np.floor((self._totalLength - self._gapLength) / 2))
        end = int(np.floor((self._totalLength + self._gapLength) / 2))
        sides = np.concatenate((audio[:, :begin], audio[:, end:]), axis=1)
        sides = np.reshape(sides, [-1, self._totalLength - self._gapLength, 1])
        gaps = audio[:, begin:end]
        gaps = np.reshape(gaps, [-1, self._gapLength])

        sides, gaps = self._remove_examples_with_low_energy_in_gap(sides, gaps)
        return sides, gaps

    def _remove_examples_with_low_energy_in_gap(self, processed_data, gaps):
        mask = np.where(np.sum(np.abs(gaps), axis=1) < self._gapLength * self._gapMinRMS)
        processed_data = np.delete(processed_data, mask, axis=0)
        gaps = np.delete(gaps, mask, axis=0)

        return processed_data, gaps

