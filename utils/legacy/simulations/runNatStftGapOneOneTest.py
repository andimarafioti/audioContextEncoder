"""
This trained for 85k steps (24hs) with a learning rate of 1e-3 and didn't learn anything.

"""

import tensorflow as tf
from tensorflow.contrib import slim

from network.emptyTFGraph import EmptyTfGraph
from utils.legacy.stftGapContextEncoder import StftGapContextEncoder

__author__ = 'Andres'

tf.reset_default_graph()
train_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'
valid_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'

window_size = 5120
gap_length = 1024
batch_size = 256

fft_frame_length = 512
fft_frame_step = 128

aTargetModel = EmptyTfGraph(shapeOfInput=(batch_size, window_size), name="Target Model")

with tf.name_scope('Remove_unnecesary_sides_before_stft'):
    signal = aTargetModel.output()
    signal_without_unnecesary_sides = signal[:, 1664:3456]
    aTargetModel.setOutputTo(signal_without_unnecesary_sides)
aTargetModel.addSTFT(frame_length=fft_frame_length, frame_step=fft_frame_step)
aTargetModel.divideComplexOutputIntoRealAndImaginaryParts()  # (256, 11, 257, 2)

aModel = EmptyTfGraph(shapeOfInput=(batch_size, window_size), name="context encoder")

with tf.name_scope('Remove_gap_before_stft'):
    signal = aModel.output()
    left_side = signal[:, :2048]
    right_side = signal[:, 2048+1024:]
    
    # This is strange. The window is 5K samples long, the hole 1024 and the 0 pading 384.
    # Unless signal in in spectrogram. In that case, the code is not very clear. Maybe consider adding comments.
    left_side_padded = tf.concat((left_side, tf.zeros((batch_size, 384))), axis=1)
    right_side_padded = tf.concat((tf.zeros((batch_size, 384)), right_side), axis=1)

    # If you pad them with 0, maybe you also stack them allong axis 2 (one after the other.)
    signal_without_gap = tf.stack((left_side_padded, right_side_padded), axis=1)  # (256, 2, 2432)
    aModel.setOutputTo(signal_without_gap)

aModel.addSTFT(frame_length=fft_frame_length, frame_step=fft_frame_step)  # (256, 2, 16, 257)
aModel.addReshape((batch_size, 32, 257))
aModel.divideComplexOutputIntoRealAndImaginaryParts()  # (256, 32, 257, 2)
aModel.addReshape((batch_size, 16, 257, 4))

with tf.variable_scope("Encoder"):
    filter_shapes = [(7, 89), (3, 17), (2, 9), (1, 5), (2, 5), (2, 5)]
    input_channels = [4, 32, 128, 512, 256, 128]
    output_channels = [32, 128, 512, 256, 128, 256]
    strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1]]
    names = ['First_Conv', 'Second_Conv', 'Third_Conv', 'Fourth_Conv', 'Fifth_Conv', 'Sixth_Conv']
    aModel.addSeveralConvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                output_channels=output_channels, strides=strides, names=names)

aModel.addReshape((batch_size, 2560))
aModel.addFullyConnectedLayer(2560, 2048, 'Fully')
aModel.addRelu()
aModel.addBatchNormalization()
aModel.addReshape((batch_size, 8, 8, 32))

with tf.variable_scope("Decoder"):
    filter_shapes = [(5, 5), (3, 3), (3, 3), (11, 11)]
    input_channels = [32, 128, 512, 128]
    output_channels = [128, 512, 128, 32]
    strides = [[1, 2, 2, 1]] * len(input_channels)
    names = ['First_Deconv', 'Second_Deconv', 'Third_Deconv', 'Fourth_Deconv']
    aModel.addSeveralDeconvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                  output_channels=output_channels, strides=strides, names=names)
    aModel.addDeconvLayerWithoutNonLin(filter_shape=(13, 13), input_channels=32, output_channels=2,
                                       stride=(1, 1, 1, 1), name="Last_Deconv")

    aModel.addReshape((batch_size, 128, 2, 128))
    aModel.addConvLayer(filter_shape=(1, 1), input_channels=128, output_channels=11, stride=(1, 1, 1, 1),
                        name='first_1by1')
    aModel.addReshape((batch_size, 11, 2, 128))
    aModel.addConvLayer(filter_shape=(1, 1), input_channels=128, output_channels=257, stride=(1, 1, 1, 1),
                        name='second_1by1')
    aModel.addReshape((batch_size, 11, 257, 2))

print(aModel.description())

model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

aContextEncoderNetwork = StftGapContextEncoder(model=aModel, batch_size=batch_size, target_model=aTargetModel, window_size=window_size,
                                               gap_length=gap_length, learning_rate=1e-3, name='nat_stft_gap_1to1_1_')
aContextEncoderNetwork.train(train_filename, valid_filename, num_steps=1e6)
