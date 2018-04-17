import tensorflow as tf
from network.tfGraph import TFGraph

__author__ = 'Andres'


class EmptyTfGraph(TFGraph):
    """
    This class is meant to represent a tensorflow graph.
    It is initialized empty and one can add different types of layers to it.
    The output of the network is accessed with output()
    The input of the function is a placeholder and can be set with input()

    input_shape : Shape of the input (with batch size)
    """

    def __init__(self, shapeOfInput, isTraining, name):
        inputSignal = tf.placeholder(tf.float32, shape=shapeOfInput, name='input_data')
        super().__init__(inputSignal=inputSignal, isTraining=isTraining, name=name)
