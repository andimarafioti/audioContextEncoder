from network.architecture.contextEncoder import ContextEncoder
from network.architecture.convNetworkParams import ConvNetworkParams
from network.architecture.fullyLayerParams import FullyLayerParams

batchSize = 256


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
targetShape = (batchSize, 11, 257, 2)
aContextEncoderArchitecture = ContextEncoder(inputShape, targetShape, encoderParams, fullyParams, decoderParams)