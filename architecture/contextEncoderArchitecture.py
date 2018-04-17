import tensorflow as tf

from architecture.architecture import Architecture
from network.tfGraph import TFGraph

__author__ = 'Andres'


class ContextEncoderArchitecture(Architecture):
    def __init__(self, inputShape, encoderParams, decoderParams, fullyParams):
        with tf.variable_scope("ContextEncoderArchitecture"):
            self._inputShape = inputShape
            self._encoderParams = encoderParams
            self._decoderParams = decoderParams
            self._fullyParams = fullyParams
            super().__init__()

    def inputShape(self):
        return self._inputShape

    def _lossGraph(self):
        with tf.variable_scope("Loss"):
            targetSquaredNorm = tf.reduce_sum(tf.square(self._target), axis=[1, 2, 3])

            error = self._target - self._output
            error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])

            reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (targetSquaredNorm+1e-4)))
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-2
            total_loss = tf.add_n([reconstruction_loss, lossL2])

            total_loss_summary = tf.summary.scalar("total_loss", total_loss)
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

            return total_loss

    def _network(self, data):
        encodedData = self._encode(data)
        connectedData = self._fullyConnect(encodedData)
        decodedData = self._decode(connectedData)
        return decodedData

    def _encode(self, data):
        with tf.variable_scope("Encoder"):
            encoder = TFGraph(data, self._isTraining, "Encoder")

            encoder.addSeveralConvLayers(filter_shapes=self._encoderParams.filterShapes(),
                                         input_channels=self._encoderParams.inputChannels(),
                                         output_channels=self._encoderParams.outputChannels(),
                                         strides=self._encoderParams.strides(),
                                         names=self._encoderParams.convNames())
            return encoder.output()

    def _fullyConnect(self, data):
        with tf.variable_scope("Fully"):
            fullyConnected = TFGraph(data, self._isTraining, "Fully")

            fullyConnected.addReshape((self._fullyParams.batchSize(), self._fullyParams.inputChannels()))
            fullyConnected.addFullyConnectedLayer(self._fullyParams.inputChannels(),
                                                  self._fullyParams.outputChannels(),
                                                  'Fully')
            fullyConnected.addRelu()
            fullyConnected.addBatchNormalization()
            fullyConnected.addReshape(self._fullyParams.outputShape())
            return fullyConnected.output()

    def _decode(self, data):
        with tf.variable_scope("Decoder"):
            decoder = TFGraph(data, self._isTraining, "Decoder")

            decoder.addSeveralDeconvLayers(filter_shapes=self._decoderParams.filterShapes()[0:-2],
                                           input_channels=self._decoderParams.inputChannels()[0:-2],
                                           output_channels=self._decoderParams.outputChannels()[0:-2],
                                           strides=self._decoderParams.strides()[0:-2],
                                           names=self._decoderParams.convNames()[0:-2])

            currentShape = decoder.outputShape()
            constantForReshape = int(4 * currentShape[1] / currentShape[2])
            decoder.addReshape((currentShape[0], int(currentShape[1] / constantForReshape),
                                currentShape[3], currentShape[2] * constantForReshape))

            decoder.addDeconvLayer(filter_shape=self._decoderParams.filterShapes()[-2],
                                   input_channels=currentShape[2] * constantForReshape,
                                   output_channels=self._decoderParams.outputChannels()[-2],
                                   stride=self._decoderParams.strides()[-2],
                                   name=self._decoderParams.convNames()[-2])
            decoder.addBatchNormalization()

            currentShape = decoder.outputShape()
            constantForReshape = int(self._decoderParams.strides()[-2][2])

            decoder.addReshape((currentShape[0], currentShape[3],
                                int(currentShape[2] / constantForReshape),
                                currentShape[1] * constantForReshape))

            decoder.addDeconvLayerWithoutNonLin(filter_shape=self._decoderParams.filterShapes()[-1],
                                                input_channels=currentShape[1] * constantForReshape,
                                                output_channels=self._decoderParams.outputChannels()[-1],
                                                stride=self._decoderParams.strides()[-1],
                                                name=self._decoderParams.convNames()[-1])
            return decoder.output()
