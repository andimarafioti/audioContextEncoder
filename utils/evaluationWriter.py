import pandas as pd
import numpy as np

__author__ = 'Andres'


class EvaluationWriter(object):
    def __init__(self, excelFileName):
        self._writer = pd.ExcelWriter(excelFileName)
        self._index = 0

    def evaluate(self, reconstructed, original_gaps, step):
        assert (len(original_gaps) == len(reconstructed))

        fake_a = reconstructed
        gap = original_gaps

        SNRs = self._pavlovs_SNR(gap, fake_a)

        norm_orig = self._squaredEuclideanNorm(gap) / 5
        error = gap - fake_a
        reconstruction_loss = 0.5 * np.sum(np.square(error), axis=1) * (1 + 1 / norm_orig)

        df = pd.DataFrame({'SNRs ' + str(step): SNRs, 'reconstruction_loss ' + str(step): reconstruction_loss})
        df.describe().to_excel(self._writer, sheet_name='general', startcol=self._index, index=not self._index)
        self._index += 3
        return np.mean(SNRs)

    def _pavlovs_SNR(self, y_orig, y_inp):
        norm_y_orig = self._squaredEuclideanNorm(y_orig)
        norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp)
        return 10 * np.log10(norm_y_orig / norm_y_orig_minus_y_inp)

    def _squaredEuclideanNorm(self, vector):
        squared = np.square(vector)
        summed = np.sum(squared, axis=1)
        return summed

    def save(self):
        self._writer.save()
