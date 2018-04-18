# import pandas as pd
import numpy as np

__author__ = 'Andres'


class EvaluationWriter(object):
    def __init__(self, excelFileName):
        # self._writer = pd.ExcelWriter(excelFileName)
        self._index = 0

    def evaluate(self, reconstructed, original_gaps, step):
        assert (len(original_gaps) == len(reconstructed))

        SNRs = self._pavlovs_SNR(original_gaps, reconstructed)

        norm_orig = self._squaredEuclideanNorm(original_gaps) / 5
        error = original_gaps - reconstructed
        reconstruction_loss = 0.5 * np.sum(np.square(error), axis=1) * (1 + 1 / norm_orig)

        # df = pd.DataFrame({'SNRs ' + str(step): SNRs, 'reconstruction_loss ' + str(step): reconstruction_loss})
        # df.describe().to_excel(self._writer, sheet_name='general', startcol=self._index, index=not self._index)
        self._index += 3
        return np.mean(SNRs)

    def evaluateImages(self, reconstructed, original_gaps, step):
        print('original_gaps:', original_gaps.shape)
        print('reconstructed:', reconstructed.shape)
        assert (original_gaps.shape == reconstructed.shape)

        SNRs = self._pavlovs_SNR(original_gaps, reconstructed, onAxis=(1, 2, 3))

        # norm_orig = self._squaredEuclideanNorm(original_gaps, onAxis=(1, 2, 3)) / 5
        # error = original_gaps - reconstructed
        # reconstruction_loss = 0.5 * np.sum(np.square(error), axis=(1, 2, 3)) * (1 + 1 / norm_orig)
        #
        # # df = pd.DataFrame({'SNRs ' + str(step): SNRs, 'reconstruction_loss ' + str(step): reconstruction_loss})
        # # df.describe().to_excel(self._writer, sheet_name='general', startcol=self._index, index=not self._index)
        # self._index += 3
        return np.mean(SNRs)

    def _pavlovs_SNR(self, y_orig, y_inp, onAxis=(1,)):
        norm_y_orig = self._squaredEuclideanNorm(y_orig, onAxis)
        norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp, onAxis)
        return 10 * np.log10(norm_y_orig / norm_y_orig_minus_y_inp)

    def _squaredEuclideanNorm(self, vector, onAxis=(1,)):
        squared = np.square(vector)
        print('squared:', squared.shape)
        summed = np.sum(squared, axis=onAxis)
        return summed

    def save(self):
        pass
        # self._writer.save()
