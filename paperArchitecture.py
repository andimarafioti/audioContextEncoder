from architecture.contextEncoderArchitecture import ContextEncoderArchitecture
from architecture.convNetworkParams import ConvNetworkParams
from architecture.fullyLayerParams import FullyLayerParams
from system.contextEncoderSystem import ContextEncoderSystem
from utils.stftForTheContextEncoder import StftForTheContextEncoder

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
aContextEncoderArchitecture = ContextEncoderArchitecture(inputShape, encoderParams, decoderParams, fullyParams)
anStftForTheContextEncoder = StftForTheContextEncoder(signalLength=5120, gapLength=1024, fftWindowLength=512,
                                                      fftHopSize=128)
aContextEncoderSystem = ContextEncoderSystem(aContextEncoderArchitecture, batchSize,
                                             anStftForTheContextEncoder, "Context_Encoder")



