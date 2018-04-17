import numpy as np
import librosa

__author__ = 'Andres'


class ExampleProcessor(object):
    def __init__(self, gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3):
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
        return "_w" + str(self._totalLength) + '_g' + str(self._gapLength) + '_h' + str(self._hopSize)

    def process(self, audio_signal):
        audio_without_silence_at_beginning_and_end = self._trim_silence(audio_signal, frame_length=self._gapLength)
        windowed_audio = self._window(audio_without_silence_at_beginning_and_end)
        processed_windows = self._remove_examples_with_low_energy_in_gap(windowed_audio)
        return processed_windows

    def _trim_silence(self, audio, frame_length=1024):
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rmse(audio, frame_length=frame_length)
        frames = np.nonzero(energy > self._gapMinRMS * 10)
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

    def _remove_examples_with_low_energy_in_gap(self, windows):
        begin = int(np.floor((self._totalLength - self._gapLength) / 2))
        end = int(np.floor((self._totalLength + self._gapLength) / 2))
        gaps = windows[:, begin:end]

        mask = np.where(np.sum(np.abs(gaps), axis=1) < self._gapLength * self._gapMinRMS)
        processed_windows = np.delete(windows, mask, axis=0)

        return processed_windows

