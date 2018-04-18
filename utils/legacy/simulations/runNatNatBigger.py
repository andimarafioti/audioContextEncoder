import tensorflow as tf

from network.emptyTFGraph import EmptyTfGraph
from utils.legacy.contextEncoder import ContextEncoderNetwork

__author__ = 'Andres'

tf.reset_default_graph()
train_filename = 'train_full_w5120_g1024_h512_ex18978619.tfrecords'
valid_filename = 'valid_full_w5120_g1024_h512_ex893971.tfrecords'

window_size = 5120
gap_length = 1024
batch_size = 256

aModel = EmptyTfGraph(shapeOfInput=(batch_size, window_size - gap_length), name="context encoder")

dataset = aModel.output()
first_half = dataset[:, :(window_size - gap_length) // 2]
second_half = dataset[:, (window_size - gap_length) // 2:]
stacked_halfs = tf.stack([first_half, second_half], axis=2)
aModel.setOutputTo(stacked_halfs)

with tf.variable_scope("Encoder"):
    aModel.addReshape((batch_size, 1, (window_size - gap_length) // 2, 2))
    filter_shapes = [(1, 129), (1, 65), (1, 33), (1, 17), (1, 17), (1, 17)]
    input_channels = [2, 32, 128, 512, 256, 128]
    output_channels = [*input_channels[1:], 64]
    strides = [[1, 1, 2, 1]] * len(input_channels)
    names = ['First_Conv', 'Second_Conv', 'Third_Conv', 'Fourth_Conv', 'Fifth_Conv', 'Six_Conv']
    aModel.addSeveralConvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                output_channels=output_channels, strides=strides, names=names)

aModel.addReshape((batch_size, 2048))
aModel.addFullyConnectedLayer(2048, 2048, 'Fully')
aModel.addRelu()
aModel.addReshape((batch_size, 1, 32, 64))

with tf.variable_scope("Decoder"):
    filter_shapes = [(1, 17), (1, 17), (1, 33), (1, 65), (1, 65)]
    input_channels = [64, 128, 512, 256, 128]
    output_channels = [*input_channels[1:], 16]
    strides = [[1, 1, 2, 1]] * len(input_channels)
    names = ['First_Deconv', 'Second_Deconv', 'Third_Deconv', 'Fourth_Deconv', 'Fifth_Deconv']
    aModel.addSeveralDeconvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                  output_channels=output_channels, strides=strides, names=names)
    aModel.addDeconvLayerWithoutNonLin(filter_shape=(1, 129), input_channels=16, output_channels=1,
                                       stride=(1, 1, 1, 1), name="Last_Deconv")
    aModel.addReshape((batch_size, gap_length))

aContextEncoderNetwork = ContextEncoderNetwork(model=aModel, batch_size=batch_size, window_size=window_size,
                                               gap_length=gap_length, learning_rate=1e-5, name='nat_sec_bigg')
aContextEncoderNetwork.train(train_filename, valid_filename, num_steps=1e6, restore_num=564425, per_process_gpu_memory_fraction=0.9)

