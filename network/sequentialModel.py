import tensorflow as tf

__author__ = 'Andres'


class SequentialModel(object):
    """
    This class is meant to represent a Sequential Neural Network Model.
    It is initialized empty and one can add different types of layers to it.
    The output of the network is accessed with the output() function

    input_shape : Shape of the input (with batch size)
    """

    def __init__(self, shapeOfInput, name):
        self._name = name
        self._input = tf.placeholder(tf.float32, shape=shapeOfInput, name='input_data')
        self._isTraining = tf.placeholder(tf.bool, name='is_training')
        self._description = "---------\n" + name + "\n---------"
        self._outputSetter(self._input)

    def input(self):
        return self._input

    def isTraining(self):
        return self._isTraining

    def output(self):
        return self._output

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

    def _outputSetter(self, value):
        self._output = value
        self._description += "\n" + str(value)

    def _convLayerWithoutNonLin(self, input_signal, filter_shape, input_channels, output_channels, stride, name,
                                padding="SAME"):
        assert(len(filter_shape) == 2), "filter must have 2 dimensions!"
        with tf.variable_scope(name, reuse=False):
            layers_filters = self._weight_variable([filter_shape[0], filter_shape[1], input_channels, output_channels])
            # layers_biases = self._bias_variable([output_channels])
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
            # layers_biases = self._bias_variable([output_channels])
            deconv = tf.nn.conv2d_transpose(input_signal, layers_filters, strides=strides, padding=padding,
                                            output_shape=output_shape)
            return deconv

    def _linearLayer(self, input_signal, input_size, output_size, name):
        with tf.variable_scope(name, reuse=False):
            weights = self._weight_variable([input_size, output_size])
            # biases = self._bias_variable(output_size)
            linear_function = tf.matmul(input_signal, weights)  # + biases
            return linear_function

    def _weight_variable(self, shape):
        return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable('b', shape, initializer=tf.contrib.layers.xavier_initializer())
