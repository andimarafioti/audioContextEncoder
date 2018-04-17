import tensorflow as tf

__author__ = 'Andres'


class TFGraph(object):
    """
    This class is meant to represent a tensorflow graph.
    It is initialized empty and one can add different types of layers to it.
    The output of the network is accessed with output()
    The input of the function is a placeholder and can be set with input()

    input_shape : Shape of the input (with batch size)
    """

    def __init__(self, inputSignal, isTraining, name):
        self._name = name
        self._input = inputSignal
        self._isTraining = isTraining
        self._description = "---------\n" + name + "\n---------"
        self._outputSetter(self._input)

    def input(self):
        return self._input

    def output(self):
        print(self.description())
        return self._output

    def outputShape(self):
        return self._output.get_shape().as_list()

    def setOutputTo(self, value):
        self._outputSetter(value)

    def description(self):
        return self._description

    def addSeveralConvLayers(self, filter_shapes, input_channels, output_channels, strides, names, padding="SAME"):
        assert (len(filter_shapes) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_shape, input_channels, output_channels, stride, name in \
                zip(filter_shapes, input_channels, output_channels, strides, names):
            self.addConvLayer(filter_shape, input_channels, output_channels, stride, name, padding)
            self.addBatchNormalization()

    def addSeveralConvLayersWithSkip(self, filter_shapes, input_channels, output_channels, strides, names, padding="SAME"):
        assert (len(filter_shapes) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_shape, input_channels, output_channels, stride, name in \
                zip(filter_shapes, input_channels, output_channels, strides, names):
            self.addConvLayerWithSkip(filter_shape, input_channels, output_channels, stride, name, padding)
            self.addBatchNormalization()

    def addConvLayer(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        self._outputSetter(self._convLayerWithoutNonLin(self._output, filter_shape, input_channels, output_channels,
                                           stride, name, padding))
        self.addRelu()

    def addConvLayerWithSkip(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        temp1 = self._convLayerWithoutNonLin(self._output, filter_shape, input_channels, input_channels,
                                          [1,1,1,1], name+'_1', "SAME")
        self._outputSetter(tf.nn.relu(temp1) + self._output)
        self.addConvLayer(filter_shape, input_channels, output_channels, stride, name+'_2', padding)

    def addConvLayerWithoutNonLin(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        self._outputSetter(self._convLayerWithoutNonLin(self._output, filter_shape, input_channels, output_channels,
                                                        stride, name, padding))

    def addSeveralDeconvLayers(self, filter_shapes, input_channels, output_channels, strides, names, padding="SAME"):
        assert (len(filter_shapes) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_shape, input_channels, output_channels, stride, name in \
                zip(filter_shapes, input_channels, output_channels, strides, names):
            self.addDeconvLayer(filter_shape, input_channels, output_channels, stride, name, padding)
            self.addBatchNormalization()

    def addSeveralDeconvLayersWithSkip(self, filter_shapes, input_channels, output_channels, strides, names, padding="SAME"):
        assert (len(filter_shapes) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_shape, input_channels, output_channels, stride, name in \
                zip(filter_shapes, input_channels, output_channels, strides, names):
            self.addDeconvLayerWithSkip(filter_shape, input_channels, output_channels, stride, name, padding)
            self.addBatchNormalization()

    def addDeconvLayer(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        self._outputSetter(self._deconvLayerWithoutNonLin(self._output, filter_shape, input_channels, output_channels,
                                             stride, name, padding))
        self.addRelu()

    def addDeconvLayerWithSkip(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        temp1 = self._deconvLayerWithoutNonLin(self._output, filter_shape, input_channels, input_channels,
                                                          [1,1,1,1], name+'_1', "SAME")
        self._outputSetter(tf.nn.relu(temp1) + self._output)
        self.addDeconvLayer(filter_shape, input_channels, output_channels, stride, name+'_2', padding)

    def addDeconvLayerWithoutNonLin(self, filter_shape, input_channels, output_channels, stride, name, padding="SAME"):
        self._outputSetter(self._deconvLayerWithoutNonLin(self._output, filter_shape, input_channels, output_channels,
                                                          stride, name, padding))

    def addReshape(self, output_shape):
        self._outputSetter(tf.reshape(self._output, output_shape))

    def addFullyConnectedLayer(self, input_size, output_size, name):
        self._outputSetter(self._linearLayer(self._output, input_size, output_size, name))

    def addDropout(self, rate):
        dropout = tf.layers.dropout(self._output, rate=rate, training=self._isTraining, name='dropout_'+str(rate))
        self._outputSetter(dropout)

    def addRelu(self):
        self._outputSetter(tf.nn.relu(self._output))

    def addBatchNormalization(self):
        self._outputSetter(tf.layers.batch_normalization(self._output, training=self._isTraining))

    def addSTFT(self, frame_length, frame_step):
        with tf.name_scope('stft'):
            self._outputSetter(tf.contrib.signal.stft(signals=self._output,
                                                      frame_length=frame_length, frame_step=frame_step))

    def addAbs(self):
        self._outputSetter(tf.abs(self._output))

    def divideComplexOutputIntoRealAndImaginaryParts(self):
        real_part = tf.real(self._output)
        imag_part = tf.imag(self._output)
        stacked = tf.stack([real_part, imag_part], axis=-1, name='divideComplexOutputIntoRealAndImaginaryParts')
        self._outputSetter(stacked)

    def divideComplexOutputIntoMagAndPhase(self):
        mag = tf.abs(self._output)
        phase = tf.angle(self._output)
        stacked = tf.stack([mag, phase], axis=-1, name='divideComplexOutputIntoMagAndPhaseParts')
        self._outputSetter(stacked)

    def divideComplexOutputIntoMagAndMaskedPhase(self, threshold=1e-3):
        mag = tf.abs(self._output)
        phase = tf.angle(self._output)

        shape = mag.shape.as_list()
        shape[0] = 1
        axis_to_reduce = [x for x in range(1, len(shape))]
        maximum_per_example = tf.reduce_max(mag, axis=axis_to_reduce, keep_dims=True)  # Denkers run on tf 1.4
        maximum_extruded = tf.tile(maximum_per_example, shape)

        mask = maximum_extruded * threshold
        masked_phase = tf.where(mag < mask, phase, phase * 0)

        stacked = tf.stack([mag, masked_phase], axis=-1, name='divideComplexOutputIntoMagAndMaskedPhaseParts')
        self._outputSetter(stacked)

    def _outputSetter(self, value):
        self._output = value
        self._description += "\n" + str(value)

    def _convLayerWithoutNonLin(self, input_signal, filter_shape, input_channels, output_channels, stride, name,
                                padding="SAME"):
        assert(len(filter_shape) == 2), "filter must have 2 dimensions!"
        with tf.variable_scope(name, reuse=False):
            layers_filters = self._weight_variable([filter_shape[0], filter_shape[1], input_channels, output_channels])
            conv = tf.nn.conv2d(input_signal, layers_filters, strides=stride, padding=padding)
            return conv

    def _deconvLayerWithoutNonLin(self, input_signal, filter_shape, input_channels, output_channels, strides,
                                  name, padding="SAME"):
        assert(len(filter_shape) == 2), "filter must have 2 dimensions!"

        input_signal_shape = input_signal.get_shape().as_list()
        output_shape = input_signal_shape[:-1] + [output_channels]
        for index, stride in enumerate(strides):
            output_shape[index] = output_shape[index] * stride

        with tf.variable_scope(name, reuse=False):
            layers_filters = self._weight_variable([filter_shape[0], filter_shape[1], output_channels, input_channels])
            deconv = tf.nn.conv2d_transpose(input_signal, layers_filters, strides=strides, padding=padding,
                                            output_shape=output_shape)
            return deconv

    def _linearLayer(self, input_signal, input_size, output_size, name):
        with tf.variable_scope(name, reuse=False):
            weights = self._weight_variable([input_size, output_size])
            linear_function = tf.matmul(input_signal, weights)
            return linear_function

    def _weight_variable(self, shape):
        return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable('b', shape, initializer=tf.contrib.layers.xavier_initializer())
