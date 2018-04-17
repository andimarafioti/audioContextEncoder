import tensorflow as tf
from system.dnnSystem import DNNSystem
from utils.tfReader import TFReader

__author__ = 'Andres'


class ContextEncoderSystem(DNNSystem):
    def __init__(self, architecture, batchSize, anStftForTheContextEncoder, name):
        self._windowSize = anStftForTheContextEncoder.signalLength()
        self._batchSize = batchSize
        self._audio = tf.placeholder(tf.float32, shape=(batchSize, self._windowSize), name='audio_data')
        self._stftForGap = anStftForTheContextEncoder.stftForGapOf(self._audio)
        self._stftForContext = anStftForTheContextEncoder.stftForTheContextOf(self._audio)
        super().__init__(architecture, name)

    def optimizer(self, learningRate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self._architecture.loss())

    def _trainingFeedDict(self, data, sess):
        net_input, net_target = sess.run([self._stftForContext, self._stftForGap], feed_dict={self._audio: data})
        return {self._architecture.input(): net_input, self._architecture.target(): net_target,
                self._architecture.isTraining(): True}

    def _evaluate(self, summariesDict, feed_dict, validReader, sess):
        raise NotImplementedError("Subclass Responsibility")

    def _loadReader(self, dataPath):
        return TFReader(dataPath, self._windowSize, batchSize=self._batchSize, capacity=int(2e5), num_epochs=400)

    def _evaluationSummaries(self):
        raise NotImplementedError("Subclass Responsibility")
