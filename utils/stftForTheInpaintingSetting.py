import tensorflow as tf

__author__ = 'Andres'


class StftForTheInpaintingSetting(object):
    def __init__(self, signal_length, gap_length, fft_window_length, fft_hop_size):
        super(StftForTheInpaintingSetting, self).__init__()
        self._signalLength = signal_length
        self._gapLength = gap_length
        self._fftWindowLength = fft_window_length
        self._fftHopSize = fft_hop_size

    def signalLength(self):
        return self._signalLength

    def gapLength(self):
        return self._gapLength

    def fftWindowLenght(self):
        return self._fftWindowLength

    def fftHopSize(self):
        return self._fftHopSize

    def padding(self):
        return self._fftWindowLength - self._fftHopSize

    def addStftForGapTo(self, aModel):
        signal = aModel.output()
        assert len(signal.shape) == 2
        signalWithoutExtraSides = self._removeExtraSidesForSTFTOfGap(signal)
        aModel.setOutputTo(signalWithoutExtraSides)
        aModel.addSTFT(frame_length=self._fftWindowLength, frame_step=self._fftHopSize)

    def addStftForTheContextTo(self, aModel):
        aBatchOfSignals = aModel.output()
        assert len(aBatchOfSignals.shape) == 2
        signalWithoutGap = self._removeGap(aBatchOfSignals)
        contextOfTheSignalPadded = self._addPaddingForStftOfContext(signalWithoutGap)
        aModel.setOutputTo(contextOfTheSignalPadded)
        aModel.addSTFT(frame_length=self._fftWindowLength, frame_step=self._fftHopSize)

    def _gapBeginning(self):
        return (self._signalLength-self._gapLength)//2

    def _gapEnding(self):
        return self._gapBeginning()+self._gapLength

    def _removeExtraSidesForSTFTOfGap(self, batchOfSignals):
        return batchOfSignals[:, self._gapBeginning() - self.padding(): self._gapEnding() + self.padding()]

    def _removeGap(self, batchOfSignals):
        leftSide = batchOfSignals[:, :self._gapBeginning()]
        rightSide = batchOfSignals[:, self._gapEnding():]
        return tf.stack((leftSide, rightSide), axis=1)

    def _addPaddingForStftOfContext(self, batchOfSides):
        """batchOfSides should contain the left side on the first dimension and the right side on the second"""
        batchSize = batchOfSides.shape.as_list()[0]
        leftSidePadded = tf.concat((batchOfSides[:, 0], tf.zeros((batchSize, self.padding()))), axis=1)
        rightSidePadded = tf.concat((batchOfSides[:, 1], tf.zeros((batchSize, self.padding()))), axis=1)
        return tf.stack((leftSidePadded, rightSidePadded), axis=1)
