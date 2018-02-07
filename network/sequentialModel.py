import tensorflow as tf


__author__ = 'Andres'


class SequentialModel(object):
    """
    This class is meant to represent a Sequential Neural Network Model.
    It is initialized empty and one can add different types of layers to it.
    The output of the network is accessed with the output() function

    input_shape : Shape of the input (disregarding the batch size)
    """
    def __init__(self, train_input_data, name):
        self._name = name
        self.train_input_data = train_input_data
        self._description = "---------\n" + name + "\n---------"
        self._outputSetter(self.train_input_data)
        self._layerPrimitives = []

    def output(self):
        return self._output

    def description(self):
        return self._description

    def _outputSetter(self, value):
        self._output = value
        self._description += "\n" + str(value)

    def addConvLayer(self, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        self._outputSetter(self._convLayer(self._output, filter_width, input_channels, output_channels,
                                       stride, name, isTraining, padding))

    def addConvLayerWithoutNonLin(self, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        self._outputSetter(self._convLayerWithoutNonLin(self._output, filter_width, input_channels, output_channels,
                                                        stride, name, isTraining, padding))

    def addReshape(self, output_shape):
        self._outputSetter(tf.reshape(self._output, output_shape))

    def _convLayerWithoutNonLin(self, input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            layers_filters = self._weight_variable([filter_width, input_channels, output_channels])
            layers_biases = self._bias_variable([output_channels])
            conv = tf.nn.conv1d(input_signal, layers_filters, stride=stride, padding=padding)
            return conv + layers_biases

    def _convLayer(self, input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            conv = self._convLayerWithoutNonLin(input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding)
            return tf.nn.relu(conv)

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
