import os
import sys

from network.emptyTFGraph import EmptyTfGraph
from system.preAndPostProcessor import PreAndPostProcessor
from utils.legacy.stftPhaseContextEncoder import StftPhaseContextEncoder

sys.path.insert(0, '../')
import tensorflow as tf
from tensorflow.contrib import slim
import socket
if 'omenx' in socket.gethostname():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


__author__ = 'Andres'

tf.reset_default_graph()
train_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'
valid_filename = '../test_w5120_g1024_h512_ex63501.tfrecords'

signal_length = 5120
gap_length = 1024
batch_size = 256

fft_window_length = 512
fft_hop_size = 128

aTargetModel = EmptyTfGraph(shapeOfInput=(batch_size, signal_length), name="Target Model")
anStftForTheInpaintingSetting = PreAndPostProcessor(signalLength=signal_length,
                                                    gapLength=gap_length,
                                                    fftWindowLength=fft_window_length,
                                                    fftHopSize=fft_hop_size)
anStftForTheInpaintingSetting.addStftForGapTo(aTargetModel)
aTargetModel.divideComplexOutputIntoMagAndPhase()  # (256, 11, 257, 2)

aModel = EmptyTfGraph(shapeOfInput=(batch_size, signal_length), name="context encoder")

anStftForTheInpaintingSetting.addStftForTheContextTo(aModel)
aModel.divideComplexOutputIntoMagAndPhase()
aModel.addReshape((batch_size, 16, 257, 4))

with tf.variable_scope("Encoder"):
    filter_shapes = [(7, 89), (3, 17), (2, 6), (1, 5), (1, 3)]
    input_channels = [4, 32, 64, 128, 128]
    output_channels = [32, 64, 128, 128, 200]
    strides = [[1, 2, 2, 1], [1, 2, 3, 1], [1, 2, 3, 1], [1, 1, 2, 1], [1, 1, 1, 1]]
    names = ['First_Conv', 'Second_Conv', 'Third_Conv', 'Fourth_Conv', 'Fifth_Conv']
    aModel.addSeveralConvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                output_channels=output_channels, strides=strides, names=names)

aModel.addReshape((batch_size, 3200))
aModel.addFullyConnectedLayer(3200, 2048, 'Fully')
aModel.addRelu()
aModel.addBatchNormalization()
aModel.addReshape((batch_size, 8, 8, 32))

with tf.variable_scope("Decoder"):
    filter_shapes = [(5, 5), (3, 3)]
    input_channels = [32, 64]
    output_channels = [64, 257]
    strides = [[1, 2, 2, 1]] * len(input_channels)
    names = ['First_Deconv', 'Second_Deconv']
    aModel.addSeveralDeconvLayers(filter_shapes=filter_shapes, input_channels=input_channels,
                                  output_channels=output_channels, strides=strides, names=names)

    aModel.addReshape((batch_size, 8, 257, 128))
    aModel.addDeconvLayer(filter_shape=(3, 33), input_channels=128, output_channels=11, stride=(1, 2, 2, 1),
                          name='Third_deconv')
    aModel.addBatchNormalization()

    aModel.addReshape((batch_size, 11, 257, 32))

    aModel.addDeconvLayerWithoutNonLin(filter_shape=(5, 89), input_channels=32, output_channels=1,
                                       stride=(1, 1, 1, 1), name="Last_Deconv")

print(aModel.description())

model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

aContextEncoderNetwork = StftPhaseContextEncoder(model=aModel, batch_size=batch_size, target_model=aTargetModel, window_size=signal_length,
                                               gap_length=gap_length, learning_rate=1e-3, name='nat_mag_phase_times_mag_gap_')
aContextEncoderNetwork.train(train_filename, valid_filename, num_steps=1e6)
