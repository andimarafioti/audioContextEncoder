import tensorflow as tf

from network.sequentialModel import SequentialModel
from network.stftTestContextEncoder import StftTestContextEncoder

__author__ = 'Andres'

tf.reset_default_graph()
train_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'
valid_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'

window_size = 5120
gap_length = 1024
batch_size = 256

aModel = SequentialModel(shapeOfInput=(batch_size, window_size), name="context encoder")

signal = aModel.output()

with tf.name_scope('Energy_Spectogram'):
    fft_frame_length = 512
    fft_frame_step = 128
    stft = tf.contrib.signal.stft(signals=signal, frame_length=fft_frame_length, frame_step=fft_frame_step)

    sides_stft = tf.stack((stft[:, :15, :], stft[:, 15+7:, :]), axis=3)

    mag_stft = tf.abs(sides_stft)    # (256, 15, 257, 2)
    aModel.setOutputTo(mag_stft)
	
with tf.variable_scope("Encoder"):
    filter_shapes = [(3, 33), (2, 9), (1, 3)]
    input_channels = [2, 32, 64]
    output_channels = [32, 64, 128]
    strides = [[1, 2, 4, 1], [1, 2, 4, 1], [1, 2, 4, 1]]
    names = ['First_Conv', 'Second_Conv', 'Third_Conv']
    aModel.addSeveralConvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                output_channels=output_channels, strides=strides, names=names)

aModel.addReshape((batch_size, 1280))
aModel.addFullyConnectedLayer(1280, 896, 'Fully')
aModel.addRelu()
aModel.addReshape((batch_size, 1, 7, 128))

with tf.variable_scope("Decoder"):
    filter_shapes = [(1, 5), (1, 9)]
    input_channels = [128, 256]
    output_channels = [256, 128]
    strides = [[1, 1, 2, 1]] * len(input_channels)
    names = ['First_Deconv', 'Second_Deconv']
    aModel.addSeveralDeconvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                  output_channels=output_channels, strides=strides, names=names)
    aModel.addReshape((batch_size, 1, 7, 512))
    aModel.addDeconvLayerWithoutNonLin(filter_shape=(1, 3), input_channels=512, output_channels=257,
                                       stride=(1, 1, 1, 1), name="Last_Deconv")
    aModel.addReshape((batch_size, 7, 257))

print(aModel.description())
aContextEncoderNetwork = StftTestContextEncoder(model=aModel, batch_size=batch_size, stft=stft, window_size=window_size,
                                               gap_length=gap_length, learning_rate=1e-4, name='nat_mag_stft_3_')
aContextEncoderNetwork.train(train_filename, valid_filename, num_steps=1e6)
