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
        self._description = "---------\n" + name + "\n---------"
        self._outputSetter(self._input)

    def input(self):
        return self._input

    def output(self):
        return self._output

    def setOutputTo(self, value):
        self._outputSetter(value)

    def description(self):
        return self._description

    def addSeveralConvLayers(self, filter_widths, input_channels, output_channels, strides, names, isTraining=True,
                             padding="SAME"):
        assert (len(filter_widths) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_width, input_channels, output_channels, stride, name in \
                zip(filter_widths, input_channels, output_channels, strides, names):
            self.addConvLayer(filter_width, input_channels, output_channels, stride, name, isTraining, padding)

    def addConvLayer(self, filter_width, input_channels, output_channels, stride, name, isTraining=True,
                     padding="SAME"):
        self._outputSetter(self._convLayer(self._output, filter_width, input_channels, output_channels,
                                           stride, name, isTraining, padding))

    def addConvLayerWithoutNonLin(self, filter_width, input_channels, output_channels, stride, name, isTraining=True,
                                  padding="SAME"):
        self._outputSetter(self._convLayerWithoutNonLin(self._output, filter_width, input_channels, output_channels,
                                                        stride, name, isTraining, padding))

    def addSeveralDeconvLayers(self, filter_widths, input_channels, output_channels, strides, names, isTraining=True,
                       padding="SAME"):
        assert (len(filter_widths) == len(input_channels) == len(output_channels) == len(strides) == len(names)),  \
            "filter_widths, input_channels, output_channels, strides, and names should all have the same length"
        for filter_width, input_channels, output_channels, stride, name in \
                zip(filter_widths, input_channels, output_channels, strides, names):
            self.addDeconvLayer(filter_width, input_channels, output_channels, stride, name, isTraining, padding)

    def addDeconvLayer(self, filter_width, input_channels, output_channels, stride, name, isTraining=True,
                       padding="SAME"):
        self._outputSetter(self._deconvLayer(self._output, filter_width, input_channels, output_channels,
                                             stride, name, isTraining, padding))

    def addDeconvLayerWithoutNonLin(self, filter_width, input_channels, output_channels, stride, name,
                                    isTraining=True, padding="SAME"):
        self._outputSetter(self._deconvLayerWithoutNonLin(self._output, filter_width, input_channels, output_channels,
                                                          stride, name, isTraining, padding))

    def addReshape(self, output_shape):
        self._outputSetter(tf.reshape(self._output, output_shape))

    def addFullyConnectedLayer(self, input_size, output_size, name, isTraining=True):
        self._outputSetter(self._linearLayer(self._output, input_size, output_size, name, isTraining))

    def addRelu(self):
        self._outputSetter(tf.nn.relu(self._output))

    def _outputSetter(self, value):
        self._output = value
        self._description += "\n" + str(value)

    def _convLayerWithoutNonLin(self, input_signal, filter_width, input_channels, output_channels, stride, name,
                                isTraining,
                                padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            layers_filters = self._weight_variable([1, filter_width, input_channels, output_channels])
            layers_biases = self._bias_variable([output_channels])
            conv = tf.nn.conv2d(input_signal, layers_filters, strides=stride, padding=padding)
            return conv + layers_biases

    def _convLayer(self, input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            conv = self._convLayerWithoutNonLin(input_signal, filter_width, input_channels, output_channels, stride,
                                                name, isTraining,
                                                padding)
            return tf.nn.relu(conv)

    def _deconvLayerWithoutNonLin(self, input_signal, filter_width, input_channels, output_channels, strides,
                                  name, isTraining,
                                  padding="SAME"):
        input_signal_shape = input_signal.get_shape().as_list()
        output_shape = input_signal_shape[:-1] + [output_channels]
        for index, stride in enumerate(strides):
            output_shape[index] = output_shape[index] * stride

        with tf.variable_scope(name, reuse=not isTraining):
            layers_filters = self._weight_variable([1, filter_width, output_channels, input_channels])
            layers_biases = self._bias_variable([output_channels])
            deconv = tf.nn.conv2d_transpose(input_signal, layers_filters, strides=strides, padding=padding,
                                            output_shape=output_shape)
            return deconv + layers_biases

    def _deconvLayer(self, input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                     padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            deconv = self._deconvLayerWithoutNonLin(input_signal, filter_width, input_channels, output_channels,
                                                    stride, name, isTraining)
            return tf.nn.relu(deconv)

    def _linearLayer(self, input_signal, input_size, output_size, name, isTraining):
        with tf.variable_scope(name, reuse=not isTraining):
            weights = self._weight_variable([input_size, output_size])
            biases = self._bias_variable(output_size)
            linear_function = tf.matmul(input_signal, weights) + biases
            return linear_function

    def _weight_variable(self, shape):
        return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable('b', shape, initializer=tf.contrib.layers.xavier_initializer())
