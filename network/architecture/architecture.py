import tensorflow as tf

__author__ = 'Andres'


class Architecture(object):
    def __init__(self, targetShape):
        self._input = tf.placeholder(tf.float32, shape=self._inputShape(), name='input_data')
        self._target = tf.placeholder(tf.float32, shape=targetShape, name='target_data')
        self._loss = self._lossGraph()

    def input(self):
        return self._input

    def target(self):
        return self._target

    def loss(self):
        return self._loss

    def _preprocessData(self, data):
        return data

    def _postprocessData(self, data):
        return data

    def _lossGraph(self):
        return self._lossFunction(self._postprocessData(self._network(self._preprocessData(self._input))))

    def _lossFunction(self, processedData):
        raise NotImplementedError("Subclass Responsibility")

    def _network(self, data):
        raise NotImplementedError("Subclass Responsibility")

    def _inputShape(self):
        raise NotImplementedError("Subclass Responsibility")

    def _targetShape(self):
        raise NotImplementedError("Subclass Responsibility")
