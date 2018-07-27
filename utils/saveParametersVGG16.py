import pickle

import sys

from architecture.parameters.contextEncoderParameters import ContextEncoderParameters

sys.path.append('.')  # In case we launch this from the base folder

__author__ = 'Andres'

from architecture.parameters.convNetworkParams import ConvNetworkParams
from architecture.parameters.fullyLayerParams import FullyLayerParams

"Simple script to save parameters"

architecturesParametersFile = "VGG16.pkl"

batchSize = 256
signalLength = 5120
gapLength = 1024
fftWindowLength = 256
fftHopSize = 64

encoderParams = ConvNetworkParams(filterShapes=[(3, 3)] * 13,
                                  channels=[4,
                                            64, 64,
                                            128, 128,
                                            256, 256, 256,
                                            512, 512, 512,
                                            512, 512, 512],
                                  strides=[[1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1]],
                                  name='Encoder')

fullyParams = FullyLayerParams(inputShape=(batchSize, 1, 5, 512), outputShape=(batchSize, 5, 1, 512), name="Fully")

decoderParams = ConvNetworkParams(filterShapes=[(3, 3)] * 15,
                                  channels=[512, 512, 512,
                                            512, 512, 512,
                                            256, 256, 256,
                                            128, 128,
                                            64, 64,
                                            129, 19, 2],
                                  strides=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 2, 2, 1],
                                           [1, 1, 1, 1], [1, 1, 1, 1]],
                                  name='Decoder')

contextEncoderParameters = ContextEncoderParameters(batchSize, signalLength, gapLength, fftWindowLength, fftHopSize,
                                                    encoderParams, fullyParams, decoderParams)

with open(architecturesParametersFile, 'wb') as fiModel:
    pickle.dump(contextEncoderParameters, fiModel)
