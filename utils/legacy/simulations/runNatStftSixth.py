import functools

import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops

from network.tfGraph import TFGraph
from utils.legacy.contextEncoder import ContextEncoderNetwork

__author__ = 'Andres'

tf.reset_default_graph()
train_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'
valid_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'

window_size = 5120
gap_length = 1024
batch_size = 256

aModel = TFGraph(shapeOfInput=(batch_size, window_size - gap_length), name="context encoder")

dataset = aModel.output()
signal_length = window_size - gap_length
first_half = dataset[:, :signal_length // 2]
second_half = dataset[:, signal_length // 2:]
stacked_halfs = tf.stack([first_half, second_half], axis=1)

with tf.name_scope('Energy_Spectogram'):
    fft_frame_length = 512
    fft_frame_step = 128
    window_fn = functools.partial(window_ops.hann_window, periodic=True)

    stft = tf.contrib.signal.stft(signals=stacked_halfs, frame_length=fft_frame_length, frame_step=fft_frame_step,
                                  window_fn=window_fn)
    real_stft = tf.real(stft)
    imag_stft = tf.imag(stft)
    real_stft_left = real_stft[:, 0, :, :]
    real_stft_right = real_stft[:, 1, :, :]

    imag_stft_left = imag_stft[:, 0, :, :]
    imag_stft_right = imag_stft[:, 1, :, :]

    real_stft = tf.concat([real_stft_left, real_stft_right], 1)
    imag_stft = tf.concat([imag_stft_left, imag_stft_right], 1)
    print(real_stft)

    stacked = tf.stack([real_stft, imag_stft], axis=3)
    aModel.setOutputTo(stacked)

with tf.variable_scope("Encoder"):
    filter_widths = [(7, 89), (4, 43), (2, 11), (2, 3), (2, 5)]
    input_channels = [2, 16, 32, 128, 128]
    output_channels = [16, 32, 128, 128, 64]
    strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
    names = ['First_Conv', 'Second_Conv', 'Third_Conv', 'Fourth_Conv', 'Fifth_Conv']
    aModel.addSeveralConvLayers(filter_shapes=filter_widths, input_channels=input_channels,
                                output_channels=output_channels, strides=strides, names=names)
    print(aModel.output())

aModel.addReshape((batch_size, 2176))
aModel.addFullyConnectedLayer(2176, 1152, 'Fully')
aModel.addRelu()
aModel.addReshape((batch_size, 1, 9, 128))

with tf.variable_scope("Decoder"):
    filter_widths = [(1, 3), (2, 3), (2, 5)]
    input_channels = [128, 256, 64]
    output_channels = [256, 64, 128]
    strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]]
    names = ['First_Deconv', 'Second_Deconv', 'Third_Deconv']
    aModel.addSeveralDeconvLayers(filter_shapes=filter_widths, input_channels=input_channels,
                                  output_channels=output_channels, strides=strides, names=names)
    aModel.addReshape((batch_size, 2, 128, 288))

    aModel.addDeconvLayer(filter_shape=(1, 17), input_channels=288, output_channels=20, stride=(1, 2, 2, 1),
                          name='first_deconv_after_reshape')
    aModel.addDeconvLayerWithoutNonLin(filter_shape=(4, 129), input_channels=20, output_channels=1,
                                       stride=(1, 1, 1, 1), name="Last_Deconv")
    aModel.addReshape((batch_size, gap_length))

print(aModel.description())
aContextEncoderNetwork = ContextEncoderNetwork(model=aModel, batch_size=batch_size, window_size=window_size,
                                               gap_length=gap_length, learning_rate=1e-4, name='nat_full_stft_6_')
aContextEncoderNetwork.train(train_filename, valid_filename, num_steps=1e6)
