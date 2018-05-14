import numpy as np

__author__ = 'Andres'


class FullyLayerParams(object):
    def __init__(self, inputShape, outputShape, name):
        assert inputShape[0] == outputShape[0], 'Batch size is expected to be the first element in the shapes'

        self._inputShape = inputShape
        self._outputShape = outputShape
        self._name = name

    def inputShape(self):
        return self._inputShape

    def outputShape(self):
        return self._outputShape

    def name(self):
        return self._name

    def batchSize(self):
        return self._inputShape[0]

    def inputChannels(self):
        return np.prod(self._inputShape[1:])

    def outputChannels(self):
        return np.prod(self._outputShape[1:])
