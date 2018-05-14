import pickle

import sys
sys.path.append('.')  # In case we launch this from the base folder

__author__ = 'Andres'

from architecture.parameters.convNetworkParams import ConvNetworkParams
from architecture.parameters.fullyLayerParams import FullyLayerParams

"Simple script to save parameters"

architecturesParametersFile = "Papers_Context_Encoder_parameters.pkl"

batchSize = 256
signalLength = 5120
gapLength = 1024
fftWindowLength = 512
fftHopSize = 128

encoderParams = ConvNetworkParams(filterShapes=[(7, 89), (3, 17), (2, 11),
                                                (1, 9), (1, 5), (2, 5)],
                                  channels=[4, 32, 128, 512,
                                            256, 160, 128],
                                  strides=[[1, 2, 2, 1], [1, 2, 3, 1], [1, 2, 3, 1],
                                           [1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                                  name='Encoder')

fullyParams = FullyLayerParams(inputShape=(batchSize, 128, 2, 8), outputShape=(batchSize, 8, 8, 32), name="Fully")

decoderParams = ConvNetworkParams(filterShapes=[(8, 8), (5, 5), (3, 3), (5, 67), (11, 257)],
                                  channels=[32, 128, 512, 257, 11, 2],
                                  strides=[[1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1],
                                           [1, 2, 2, 1], [1, 1, 1, 1]],
                                  name='Decoder')

inputShape = (batchSize, 16, 257, 4)

dictToSave = {"Architecture Params": [inputShape, encoderParams, decoderParams, fullyParams],
              "PreProcessor Params": [signalLength, gapLength, fftWindowLength, fftHopSize],
              "batchSize": batchSize}

with open(architecturesParametersFile, 'wb') as fiModel:
    pickle.dump(dictToSave, fiModel)
