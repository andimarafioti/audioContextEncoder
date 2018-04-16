import tensorflow as tf
from network.architecture.architecture import Architecture
from network.tfGraph import TFGraph

__author__ = 'Andres'


class ContextEncoder(Architecture):
    def __init__(self, inputShape, targetShape, encoderParams, decoderParams, fullyParams):
        self._inputShape = inputShape
        self._encoderParams = encoderParams
        self._decoderParams = decoderParams
        self._fullyParams = fullyParams
        super().__init__(targetShape)

    def _inputShape(self):
        return self._inputShape

    def _lossFunction(self, processedData):
        with tf.variable_scope("Loss"):
            gap_stft = self._target_model.output()

            norm_orig = self._squaredEuclideanNorm(gap_stft, onAxis=[1, 2, 3])
            norm_orig_summary = tf.summary.scalar("norm_orig", tf.reduce_min(norm_orig))

            error = gap_stft - self._reconstructed_input_data
            # Nati comment: here you should use only one reduce sum function
            error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])

            reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (norm_orig+1e-2)))

            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e-2
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            total_loss_summary = tf.summary.scalar("total_loss", total_loss)

            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, norm_orig_summary, total_loss_summary])

            return total_loss

    def _network(self, data):
        encodedData = self._encode(data)
        connectedData = self._fullyConnect(encodedData)
        decodedData = self._decode(connectedData)
        return decodedData

    def _encode(self, data):
        with tf.variable_scope("Encoder"):
            encoder = TFGraph(data, "Encoder")

            encoder.addSeveralConvLayers(filter_shapes=self._encoderParams.filterShapes(),
                                         input_channels=self._encoderParams.inputChannels(),
                                         output_channels=self._encoderParams.outputChannels(),
                                         strides=self._encoderParams.strides(),
                                         names=self._encoderParams.convNames())
            return encoder.output()

    def _fullyConnect(self, data):
        with tf.variable_scope("Fully"):
            fullyConnected = TFGraph(data, "Fully")

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
            decoder = TFGraph(data, "Decoder")

            decoder.addSeveralDeconvLayers(filter_shapes=self._decoderParams.filterShapes()[0:3],
                                           input_channels=self._decoderParams.inputChannels()[0:3],
                                           output_channels=self._decoderParams.outputChannels()[0:3],
                                           strides=self._decoderParams.strides()[0:3],
                                           names=self._decoderParams.convNames()[0:3])

            currentShape = decoder.outputShape()
            decoder.addReshape((currentShape[0], currentShape[1] / 4, currentShape[3], currentShape[2] * 4))

            decoder.addDeconvLayer(filter_shape=self._decoderParams.filterShapes()[3],
                                   input_channels=self._decoderParams.inputChannels()[3],
                                   output_channels=self._decoderParams.outputChannels()[3],
                                   stride=self._decoderParams.strides()[3],
                                   name=self._decoderParams.convNames()[3])
            decoder.addBatchNormalization()

            currentShape = decoder.outputShape()
            decoder.addReshape((currentShape[0], currentShape[3], currentShape[2], currentShape[1]))

            decoder.addDeconvLayerWithoutNonLin(filter_shape=self._decoderParams.filterShapes()[4],
                                                input_channels=self._decoderParams.inputChannels()[4],
                                                output_channels=self._decoderParams.outputChannels()[4],
                                                stride=self._decoderParams.strides()[4],
                                                name=self._decoderParams.convNames()[4])
            return decoder.output()
